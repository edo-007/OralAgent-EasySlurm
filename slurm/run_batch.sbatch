#!/bin/bash
# Non-interactive OralAgent run on Slurm.
#
# Usage (from the repository root; jobs/template.yaml documents the job file):
#   python run_batch.py jobs/my_job.yaml --check      # validate the job file on the login node
#   sbatch slurm/run_batch.sbatch jobs/my_job.yaml
#
# Any resource below can be overridden at submission time, e.g.:
#   sbatch --partition=gpu --time=08:00:00 slurm/run_batch.sbatch jobs/my_job.yaml
#
# Answers are written to <output_dir>/results.jsonl (output_dir is set in the job file)
# and are also printed in the Slurm log.
#
# Slurm log: by default Slurm writes slurm-<jobid>.out in the directory you submit from, i.e.
# the repository. To keep it elsewhere, pass --output at submission time or enable the
# directive below. The directory must already exist (Slurm does not create it, and the job
# fails without any message), and #SBATCH lines cannot use environment variables.
#   sbatch --output=/path/to/slurm_logs/%x-%j.out slurm/run_batch.sbatch jobs/my_job.yaml

#SBATCH --job-name=oralagent
##SBATCH --output=/path/to/slurm_logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
##SBATCH --partition=<your_partition>
##SBATCH --account=<your_account>

set -euo pipefail

JOB_FILE="${1:?Usage: sbatch slurm/run_batch.sbatch <job.yaml>}"
cd "$SLURM_SUBMIT_DIR"

# ---- Environment: adapt to your cluster ----
# module load cuda
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate oralagent

# ---- Offline assets ----
# Compute nodes have no internet access: every model is read from a directory downloaded
# on the login node with slurm/download_assets.py. Edit the path below or export
# ORALAGENT_ASSETS_DIR before submitting. (If the compute nodes have internet access, remove
# this block to let models download on demand.)
ASSETS_DIR="${ORALAGENT_ASSETS_DIR:-/path/to/oralagent_assets}"
if [ ! -d "$ASSETS_DIR/hf_cache" ]; then
  echo "Assets not found in $ASSETS_DIR: set ORALAGENT_ASSETS_DIR or edit slurm/run_batch.sbatch" >&2
  exit 1
fi
export HF_HOME="$ASSETS_DIR/hf_cache"
export TORCH_HOME="$ASSETS_DIR/torch_cache"
export HF_HUB_OFFLINE=1

python run_batch.py "$JOB_FILE"
