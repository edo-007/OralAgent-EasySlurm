"""
Download everything OralAgent needs to run on compute nodes without internet access.
Run it where internet is available (typically the login node, with <dest> on a filesystem
shared with the compute nodes); only `huggingface_hub` is required.

OralGPT/OralAgent-ToolHub and OralGPT/OralCorpus are gated: request access on their Hugging Face
pages and log in first (`huggingface-cli login`, or set HF_TOKEN).

    python slurm/download_assets.py /path/to/oralagent_assets
    python slurm/download_assets.py /path/to/oralagent_assets --rag english --llm Qwen/Qwen2.5-VL-7B-Instruct

Resulting layout:
    <dest>/hf_cache/            HF_HOME      (set by slurm/run_batch.sbatch from ASSETS_DIR)
    <dest>/torch_cache/         TORCH_HOME   (set by slurm/run_batch.sbatch from ASSETS_DIR)
    <dest>/OralAgent-ToolHub/   `model_dir` in the job file
    <dest>/OralCorpus/          `rag.persist_dir` in the job file (only with --rag)
    <dest>/llm/<name>/          local LLM weights for `llm_server.command` (only with --llm)
"""
import argparse
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

# Loaded with from_pretrained() at every start: intent recognition (agent.py) and modality classifier backbone
BASE_MODELS = ["Qwen/Qwen3-0.6B", "chuhac/BiomedCLIP-vit-bert-hf"]
# RAG_DEFAULTS in run_batch.py
RAG_MODELS = ["Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Reranker-8B"]
# torchvision weights fetched by the DINO r50 tools when they build their backbone
RESNET50_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the OralAgent assets for offline use.")
    parser.add_argument("dest", help="Destination directory, readable from the compute nodes")
    parser.add_argument("--rag", choices=["english", "chinese"], help="Also download what OralRAG needs: "
                        "embedding/rerank models and the pre-built OralCorpus vector DB in this language")
    parser.add_argument("--llm", help="Hugging Face repo id of the LLM to serve locally (e.g. with vLLM)")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    hub_cache = dest / "hf_cache" / "hub"

    for repo_id in BASE_MODELS + (RAG_MODELS if args.rag else []):
        print(f"== {repo_id}")
        snapshot_download(repo_id, cache_dir=hub_cache)

    resnet_path = dest / "torch_cache" / "hub" / "checkpoints" / Path(RESNET50_URL).name
    if not resnet_path.exists():
        print(f"== {RESNET50_URL}")
        resnet_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RESNET50_URL, resnet_path)

    if args.llm:
        print(f"== {args.llm}")
        snapshot_download(args.llm, local_dir=dest / "llm" / args.llm.split("/")[-1])

    print("== OralGPT/OralAgent-ToolHub")
    snapshot_download("OralGPT/OralAgent-ToolHub", local_dir=dest / "OralAgent-ToolHub")

    if args.rag:
        vector_db = f"vectorDB_OralCorpus_{args.rag.capitalize()}"
        print(f"== OralGPT/OralCorpus ({vector_db})")
        snapshot_download(
            "OralGPT/OralCorpus",
            repo_type="dataset",
            local_dir=dest / "OralCorpus",
            allow_patterns=[f"{vector_db}/**"],
        )

    print(f"\nDone. Point ASSETS_DIR (slurm/run_batch.sbatch) and the job file to {dest}.")


if __name__ == "__main__":
    main()
