"""
OralAgent batch runner: non-interactive execution of a job file (parameters + queries),
meant to be submitted to Slurm through slurm/run_batch.sbatch.

The agent is initialized once, every query runs in its own isolated thread, and one JSON
line per query is appended to <output_dir>/results.jsonl. Queries already answered in that
file are skipped, so a job killed by the time limit can simply be resubmitted.

Usage (from the repository root):
    python run_batch.py jobs/my_job.yaml           # run all queries
    python run_batch.py jobs/my_job.yaml --check   # only validate the job file (no models loaded)

On clusters without internet access the LLM can be served inside the same job: see the
`llm_server` parameter in jobs/template.yaml.
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent

REQUIRED_PARAMS = ("model", "model_dir", "output_dir")
DEFAULT_PARAMS: Dict[str, Any] = {
    "base_url": None,  # None -> OPENAI_BASE_URL / OPENAI_API_BASE from the environment or .env
    "temperature": 0.2,
    "send_images_to_llm": True,  # False for text-only LLMs: only the image path is passed (tools still read the file)
    "tools": None,  # None -> DEFAULT_SELECTED_TOOL_NAMES in oralagent/tools/get_tools.py
    "temp_dir": None,  # None -> <output_dir>/temp
    "prompt_file": str(REPO_ROOT / "oralagent" / "docs" / "system_prompts.txt"),
    "rag": {},  # overrides for RAG_DEFAULTS, only used when OralRAG is selected
    "llm_server": None,  # {"command": ..., "startup_timeout": ...}: LLM server started and stopped with the job
}
LLM_SERVER_DEFAULTS: Dict[str, Any] = {"command": None, "startup_timeout": 1800}
# ${name} placeholders of the job file; a bare $name is left untouched (e.g. for the shell in llm_server.command)
VAR_PATTERN = re.compile(r"\$\{(\w+)\}")
# Same values as the RAGConfig in main.py
RAG_DEFAULTS: Dict[str, Any] = {
    "model": "command-a-03-2025",
    "embedding_model": "Qwen/Qwen3-Embedding-8B",
    "rerank_model": "Qwen/Qwen3-Reranker-8B",
    "temperature": 0.7,
    "retriever_k": 7,
    "persist_dir": "oralagent/rag/",
    "use_OralCorpus": True,
    "corpus_language": "chinese",
    "local_docs_dir": "",
    "chunk_size": 1000,
    "chunk_overlap": 100,
}


def expand_vars(value: Any, variables: Dict[str, Any], undefined: Set[str]) -> Any:
    """Replace every ${name} in the strings of a (nested) job value; unknown names are collected in `undefined`."""
    if isinstance(value, str):
        def replace(match):
            if match.group(1) not in variables:
                undefined.add(match.group(1))
                return match.group(0)
            return str(variables[match.group(1)])
        return VAR_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: expand_vars(v, variables, undefined) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_vars(v, variables, undefined) for v in value]
    return value


def load_job(job_file: str) -> Dict[str, Any]:
    """Load and validate a YAML job file. Exits with the full list of problems if it is invalid."""
    with open(job_file, "r", encoding="utf-8") as f:
        raw_job = yaml.safe_load(f) or {}

    errors: List[str] = []
    # ${name} is looked up in the `vars` block first, then in the environment
    variables: Dict[str, Any] = dict(os.environ)
    undefined: Set[str] = set()
    job_vars = raw_job.pop("vars", None) or {}
    if not isinstance(job_vars, dict):
        errors.append("vars must be a block of `name: value` lines")
        job_vars = {}
    for name, value in job_vars.items():
        variables[name] = expand_vars(value, variables, undefined)  # a variable can use the ones above it
    raw_job = expand_vars(raw_job, variables, undefined)
    for name in sorted(undefined):
        errors.append(f"undefined variable: ${{{name}}} (define it in the `vars` block or in the environment)")

    known = set(REQUIRED_PARAMS) | set(DEFAULT_PARAMS) | {"queries"}
    for key in sorted(set(raw_job) - known):
        errors.append(f"unknown parameter: {key}")
    for key in REQUIRED_PARAMS:
        if not raw_job.get(key):
            errors.append(f"missing required parameter: {key}")
    for key in sorted(set(raw_job.get("rag") or {}) - set(RAG_DEFAULTS)):
        errors.append(f"unknown rag parameter: {key}")

    job = {**DEFAULT_PARAMS, **{k: v for k, v in raw_job.items() if v is not None}}
    if job["llm_server"]:
        for key in sorted(set(job["llm_server"]) - set(LLM_SERVER_DEFAULTS)):
            errors.append(f"unknown llm_server parameter: {key}")
        job["llm_server"] = {**LLM_SERVER_DEFAULTS, **job["llm_server"]}
        if not job["llm_server"]["command"]:
            errors.append("llm_server requires 'command'")
        if not job["base_url"]:
            errors.append("llm_server requires 'base_url' (e.g. http://localhost:8000/v1)")
    if not job["temp_dir"] and job.get("output_dir"):
        job["temp_dir"] = str(Path(job["output_dir"]) / "temp")
    if job.get("model_dir") and not Path(job["model_dir"]).is_dir():
        errors.append(f"model_dir is not a directory: {job['model_dir']}")
    if not Path(job["prompt_file"]).is_file():
        errors.append(f"prompt_file not found: {job['prompt_file']}")

    queries: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    for i, raw in enumerate(raw_job.get("queries") or [], start=1):
        if isinstance(raw, str):
            raw = {"query": raw}
        query_id = str(raw.get("id") or f"q{i:04d}")
        if query_id in seen_ids:
            errors.append(f"duplicate query id: {query_id}")
        seen_ids.add(query_id)

        text = (raw.get("query") or "").strip()
        if not text:
            errors.append(f"query {query_id}: empty 'query'")

        images = raw.get("images") or raw.get("image") or []
        if isinstance(images, str):
            images = [images]
        # Tools open the image from disk, so paths must be absolute and readable from the compute node
        images = [str(Path(p).expanduser().resolve()) for p in images]
        for path in images:
            if not Path(path).is_file():
                errors.append(f"query {query_id}: image not found: {path}")

        queries.append({"id": query_id, "query": text, "images": images, "index": raw.get("index")})

    if not queries:
        errors.append("no queries defined")
    if errors:
        sys.exit(f"Invalid job file {job_file}:\n" + "\n".join(f"  - {e}" for e in errors))

    job["queries"] = queries
    return job


def build_agent(job: Dict[str, Any]):
    """Initialize the agent (LLM + selected tools) from the job parameters."""
    from main import initialize_agent  # heavy import; also loads .env
    from oralagent.rag import RAGConfig
    from oralagent.tools.get_tools import DEFAULT_SELECTED_TOOL_NAMES, get_all_tools_factories

    tools = job["tools"] or DEFAULT_SELECTED_TOOL_NAMES
    rag_config = RAGConfig(**{**RAG_DEFAULTS, **job["rag"]}) if "OralRAG" in tools else None

    available = get_all_tools_factories(job["model_dir"], job["temp_dir"], rag_config=rag_config)
    unknown = [t for t in tools if t not in available]
    if unknown:
        sys.exit(f"Unknown tools: {unknown}\nAvailable tools: {sorted(available)}")
    if rag_config is not None:
        # The RAG tool builds a Cohere chat client it never calls, and that client refuses to start without a key
        os.environ.setdefault("COHERE_API_KEY", "unused")

    # An empty value (e.g. `OPENAI_API_BASE=` in .env) would be used as the endpoint and break every request
    for var in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        if os.environ.get(var) == "":
            del os.environ[var]
    # Locally served models ignore the key, but the client refuses to start without one
    openai_kwargs = {"api_key": os.getenv("OPENAI_API_KEY") or "not-needed"}
    if job["base_url"]:
        openai_kwargs["base_url"] = job["base_url"]

    agent, _ = initialize_agent(
        job["prompt_file"],
        tools_to_use=tools,
        model_dir=job["model_dir"],
        temp_dir=job["temp_dir"],
        model=job["model"],
        temperature=job["temperature"],
        rag_config=rag_config,
        openai_kwargs=openai_kwargs,
        log_tools=False,  # tool calls are already recorded in results.jsonl
    )
    return agent


def start_llm_server(command: str, log_path: Path) -> subprocess.Popen:
    """Launch the LLM server declared in the job file; its output goes to log_path."""
    print(f"Starting LLM server (log: {log_path}): {command}", flush=True)
    # Own process group, so that the whole tree (shell, conda run, workers...) can be stopped at the end
    return subprocess.Popen(
        command, shell=True, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True
    )


def wait_for_llm_server(server: subprocess.Popen, base_url: str, timeout: float, log_path: Path) -> None:
    """Block until the OpenAI-compatible endpoint answers on <base_url>/models."""
    url = base_url.rstrip("/") + "/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            sys.exit(f"LLM server exited with code {server.returncode} during startup, see {log_path}")
        try:
            with urllib.request.urlopen(url, timeout=5):
                print(f"LLM server ready at {base_url}", flush=True)
                return
        except OSError:
            time.sleep(5)
    sys.exit(f"LLM server not ready after {timeout}s ({url}), see {log_path}")


def stop_llm_server(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)


def build_messages(query: Dict[str, Any], send_images_to_llm: bool) -> List[Dict[str, Any]]:
    """Build the single user message for a query, in the same format produced by interface.py."""
    content: List[Dict[str, Any]] = []
    for path in query["images"]:
        if send_images_to_llm:
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            with open(path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            image_url = {"url": f"data:{mime};base64,{img_base64}", "image_path": path}
            if query["index"] is not None:
                image_url["index"] = query["index"]
            content.append({"type": "text", "text": f"image_path: {path}"})
            content.append({"type": "image_url", "image_url": image_url})
        else:
            content.append({"type": "text", "text": f"image_path: {path}"})
    content.append({"type": "text", "text": query["query"]})
    # Without an image_url block the agent reads the benchmark index from an `index:` text block
    if query["index"] is not None and not (query["images"] and send_images_to_llm):
        content.append({"type": "text", "text": f"index: {query['index']}"})
    return [{"role": "user", "content": content}]


def run_query(agent, messages: List[Dict[str, Any]]) -> Tuple[Any, List[Dict[str, Any]], Dict[str, int]]:
    """Run one query in a fresh thread. Returns (final response, tool calls, token usage)."""
    from langchain_core.messages import AIMessage, ToolMessage

    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    response: Any = ""
    tool_calls: Dict[str, Dict[str, Any]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0}

    for event in agent.workflow.stream({"messages": messages}, thread):
        for update in event.values():
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage):
                    response = msg.content
                    for call in msg.tool_calls:
                        tool_calls[call["id"]] = {"name": call["name"], "args": call["args"]}
                    for key in usage:
                        usage[key] += (msg.usage_metadata or {}).get(key) or 0
                elif isinstance(msg, ToolMessage):
                    tool_calls.setdefault(msg.tool_call_id, {"name": msg.name})["output"] = msg.content

    if isinstance(response, str):
        response = response.strip()
    return response, list(tool_calls.values()), usage


def load_done_ids(results_path: Path) -> Set[str]:
    """Ids of the queries already answered successfully in a previous run."""
    done: Set[str] = set()
    if not results_path.exists():
        return done
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated line left by a job killed while writing
            if record.get("status") == "ok":
                done.add(record["id"])
    return done


def run_queries(agent, job: Dict[str, Any], pending: List[Dict[str, Any]], results_path: Path) -> int:
    """Run the pending queries, appending one record each to results_path. Returns the number of failures."""
    n_errors = 0
    with open(results_path, "a", encoding="utf-8") as results_file:
        for i, query in enumerate(pending, start=1):
            record: Dict[str, Any] = {
                "id": query["id"],
                "query": query["query"],
                "images": query["images"],
                "index": query["index"],
                "model": job["model"],
                "timestamp": datetime.now().isoformat(),
            }
            start = time.time()
            try:
                messages = build_messages(query, job["send_images_to_llm"])
                response, tool_calls, usage = run_query(agent, messages)
                record.update(status="ok", response=response, tool_calls=tool_calls, token_usage=usage)
            except Exception as e:
                n_errors += 1
                record.update(status="error", error=f"{type(e).__name__}: {e}")
            record["duration_sec"] = round(time.time() - start, 3)

            results_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            results_file.flush()

            print(f"\n===== [{i}/{len(pending)}] {query['id']} ({record['status']}, {record['duration_sec']}s) =====")
            print(record.get("response") if record["status"] == "ok" else record["error"], flush=True)
    return n_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OralAgent non-interactively on a job file.")
    parser.add_argument("job_file", help="YAML job file with parameters and queries (see jobs/template.yaml)")
    parser.add_argument("--check", action="store_true", help="Validate the job file and exit without loading models")
    parser.add_argument("--overwrite", action="store_true", help="Discard previous results instead of resuming")
    args = parser.parse_args()

    job = load_job(args.job_file)
    queries = job["queries"]
    if args.check:
        n_images = sum(len(q["images"]) for q in queries)
        print(f"Job file OK: {len(queries)} queries, {n_images} images, model={job['model']}, output_dir={job['output_dir']}")
        return 0

    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(job["temp_dir"]).mkdir(parents=True, exist_ok=True)  # the tools only create the last level
    results_path = output_dir / "results.jsonl"
    if args.overwrite and results_path.exists():
        results_path.unlink()

    done_ids = load_done_ids(results_path)
    pending = [q for q in queries if q["id"] not in done_ids]
    print(f"{len(queries)} queries: {len(done_ids & {q['id'] for q in queries})} already done, {len(pending)} to run")
    if not pending:
        return 0

    server = None
    try:
        if job["llm_server"]:
            server_log = output_dir / "llm_server.log"
            server = start_llm_server(job["llm_server"]["command"], server_log)
        agent = build_agent(job)  # the tools load while the LLM server starts
        if server:
            wait_for_llm_server(server, job["base_url"], job["llm_server"]["startup_timeout"], server_log)
        n_errors = run_queries(agent, job, pending, results_path)
    finally:
        if server:
            stop_llm_server(server)

    print(f"\nDone: {len(pending) - n_errors} ok, {n_errors} failed. Results: {results_path.resolve()}")
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
