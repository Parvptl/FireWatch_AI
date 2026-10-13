#!/bin/bash
# ---------------------------------------------------------------------------
# FireWatch-AI single-GPU training job for the Bhavani cluster (partition gpu03,
# NVIDIA A30 24 GB).
#
# STATUS: TEMPLATE. The training entry point scripts/train.py does NOT exist yet.
# This script checks for it and exits with a clear error until it is added.
# Everything else (environment, logging, run directories, provenance) is ready.
#
# Submit from the repository root on the login node:
#   cd ~/projects/FireWatch-AI
#   mkdir -p slurm_logs
#   sbatch slurm/train_gpu03.sh                       # uses configs/baseline.yaml
#   sbatch slurm/train_gpu03.sh configs/other.yaml    # custom config
#
# Overridable environment variables (pass with: sbatch --export=ALL,VAR=value ...):
#   FIREWATCH_PYTHON     Python interpreter      (default: $HOME/envs/firewatch/bin/python)
#   FIREWATCH_DATA_ROOT  Dataset root, REQUIRED  (e.g. $HOME/datasets/dfire)
#   FIREWATCH_RUNS_ROOT  Run output root         (default: $HOME/firewatch_runs)
#   FIREWATCH_SEED       Random seed             (default: 42)
#
# PROVISIONAL resource values: --time=12:00:00, --mem=32G, --cpus-per-task=8 and
# the --gres=gpu:1 syntax have NOT been verified against the gpu03 partition
# policy. Before the first real job, check and adjust them:
#   scontrol show partition gpu03      # MaxTime, DefMemPerCPU/MaxMemPerNode
#   scontrol show node node003         # CPUTot, RealMemory, Gres (e.g. gpu:a30:1)
# Some clusters require a typed request such as --gres=gpu:a30:1.
#
# No step in this script needs internet access. The trainer must not download
# weights at runtime either (see docs/cluster-setup.md).
# ---------------------------------------------------------------------------
#SBATCH --job-name=firewatch-train
#SBATCH --partition=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# Paths are relative to the submission directory; slurm_logs/ must exist
# before submission (SLURM does not create it) and is ignored by Git.
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

CONFIG="${1:-configs/baseline.yaml}"
PYTHON="${FIREWATCH_PYTHON:-$HOME/envs/firewatch/bin/python}"
RUNS_ROOT="${FIREWATCH_RUNS_ROOT:-$HOME/firewatch_runs}"
SEED="${FIREWATCH_SEED:-42}"
TRAIN_SCRIPT="scripts/train.py"

fail() { echo "ERROR: $*" >&2; exit 1; }

# --- Working directory: the repo root the job was submitted from -------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f pyproject.toml && -d src/firewatch ]] \
    || fail "submit from the FireWatch-AI repository root (current dir: $PWD)"

# --- Preconditions -------------------------------------------------------------
[[ -x "$PYTHON" ]] || fail "Python interpreter not found: $PYTHON (set FIREWATCH_PYTHON)"
[[ -f "$CONFIG" ]] || fail "config file not found: $CONFIG"
[[ -n "${FIREWATCH_DATA_ROOT:-}" ]] \
    || fail "FIREWATCH_DATA_ROOT is not set; export it before sbatch (see docs/dataset-guide.md)"
[[ -d "$FIREWATCH_DATA_ROOT" ]] || fail "dataset root does not exist: $FIREWATCH_DATA_ROOT"
[[ -f "$TRAIN_SCRIPT" ]] || fail "$TRAIN_SCRIPT is not implemented yet. \
Training is not available in this version of FireWatch-AI; see README 'Implementation status'."

# --- Run directory outside the repository -----------------------------------
RUN_NAME="${SLURM_JOB_NAME:-firewatch}-${SLURM_JOB_ID:-local}"
RUN_DIR="$RUNS_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"/{checkpoints,logs}

# --- Reproducibility / provenance -------------------------------------------
export PYTHONHASHSEED="$SEED"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cp "$CONFIG" "$RUN_DIR/config.yaml"
{
    echo "job_id:      ${SLURM_JOB_ID:-n/a}"
    echo "node:        $(hostname)"
    echo "started:     $(date -Is)"
    echo "git_commit:  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty:   $([[ -n "$(git status --porcelain 2>/dev/null)" ]] && echo yes || echo no)"
    echo "config:      $CONFIG"
    echo "data_root:   $FIREWATCH_DATA_ROOT"
    echo "seed:        $SEED"
    echo "python:      $PYTHON"
    "$PYTHON" -c 'import sys, torch, torchvision; print("versions:   ", sys.version.split()[0], torch.__version__, torchvision.__version__); print("cuda:       ", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")'
} | tee "$RUN_DIR/run_info.txt"
nvidia-smi > "$RUN_DIR/nvidia-smi.txt" 2>&1 || true
git diff > "$RUN_DIR/uncommitted.diff" 2>/dev/null || true

# --- Train ----------------------------------------------------------------------
# NOTE: the CLI below is the intended contract for scripts/train.py; adjust it
# when the trainer is implemented.
"$PYTHON" "$TRAIN_SCRIPT" \
    --config "$CONFIG" \
    --data-root "$FIREWATCH_DATA_ROOT" \
    --output-dir "$RUN_DIR" \
    --seed "$SEED"

echo "finished: $(date -Is)  outputs: $RUN_DIR"
