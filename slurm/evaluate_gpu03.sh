#!/bin/bash
# ---------------------------------------------------------------------------
# FireWatch-AI evaluation job (partition gpu03, 1 GPU). Runs scripts/evaluate.py.
#
#   slurm/submit.sh evaluate --weights ~/firewatch_runs/<run>/weights/best.pt \
#       --data ~/datasets/dfire_view_v2/data.yaml               # val split
#   slurm/submit.sh evaluate --weights ... --data ... --split test --final-test
#
# All arguments are passed to scripts/evaluate.py. Resource values are
# PROVISIONAL (see slurm/train_gpu03.sh). No internet access is needed.
# ---------------------------------------------------------------------------
#SBATCH --job-name=firewatch-eval
#SBATCH --partition=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

fail() { echo "ERROR: $*" >&2; exit 2; }

PYTHON="${FIREWATCH_PYTHON:-$HOME/envs/firewatch/bin/python}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f pyproject.toml && -f scripts/evaluate.py ]] \
    || fail "submit from the FireWatch-AI repository root (current dir: $PWD)"
[[ -x "$PYTHON" ]] || fail "Python interpreter not found: $PYTHON (set FIREWATCH_PYTHON)"
[[ $# -gt 0 ]] || fail "pass scripts/evaluate.py arguments (at least --weights and --data)"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export YOLO_OFFLINE=1

echo "job ${SLURM_JOB_ID:-n/a} on $(hostname) at $(date -Is)"
exec "$PYTHON" scripts/evaluate.py --device 0 --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
