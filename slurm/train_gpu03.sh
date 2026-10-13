#!/bin/bash
# ---------------------------------------------------------------------------
# FireWatch-AI single-GPU YOLO training job (Bhavani cluster, partition gpu03,
# 1x NVIDIA A30 24 GB). Runs scripts/train.py; see docs/training.md.
#
# Submit from the repository root on the login node (preferred: the wrapper
# creates the log directory outside the repo before SLURM opens the log files):
#   slurm/submit.sh train configs/smoke_test.yaml
#   slurm/submit.sh train configs/baseline.yaml [extra scripts/train.py args]
# Direct submission also works if ./slurm_logs exists (gitignored):
#   mkdir -p slurm_logs && sbatch slurm/train_gpu03.sh configs/baseline.yaml
#
# Environment (exported to the job by sbatch --export=ALL, the default):
#   FIREWATCH_VIEW_DIR   prepared dataset view   (default: $HOME/datasets/dfire_view)
#   FIREWATCH_DATA_ROOT  raw D-Fire dataset, only needed if the view does not
#                        exist yet and must be built by this job
#   FIREWATCH_RUNS_ROOT  parent of run dirs      (default: $HOME/firewatch_runs)
#   FIREWATCH_PYTHON     interpreter             (default: $HOME/envs/firewatch/bin/python)
#   FIREWATCH_RESUME     path to <run_dir>/weights/last.pt to continue a run
#   FIREWATCH_LINK_MODE  symlink|hardlink|copy when building the view (default: config)
#
# PROVISIONAL resources: --time, --mem, --cpus-per-task and the --gres syntax
# are not verified against the gpu03 policy. Check before the first real job:
#   scontrol show partition gpu03 ; scontrol show node node003
# Some clusters require a typed request such as --gres=gpu:a30:1.
#
# Nothing in this job needs internet access: weights must already exist at the
# configured path, and the trainer forces Ultralytics offline mode.
# ---------------------------------------------------------------------------
#SBATCH --job-name=firewatch-train
#SBATCH --partition=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

fail() { echo "ERROR: $*" >&2; exit 2; }

CONFIG="${1:-configs/baseline.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
PYTHON="${FIREWATCH_PYTHON:-$HOME/envs/firewatch/bin/python}"
RUNS_ROOT="${FIREWATCH_RUNS_ROOT:-$HOME/firewatch_runs}"
VIEW_DIR="${FIREWATCH_VIEW_DIR:-$HOME/datasets/dfire_view}"

# --- Working directory: the repository root the job was submitted from --------
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f pyproject.toml && -f scripts/train.py ]] \
    || fail "submit from the FireWatch-AI repository root (current dir: $PWD)"
[[ -x "$PYTHON" ]] || fail "Python interpreter not found: $PYTHON (set FIREWATCH_PYTHON)"
[[ -f "$CONFIG" ]] || fail "config file not found: $CONFIG"

# --- Reproducibility-related environment ---------------------------------------
export PYTHONHASHSEED=0          # fixed hash seed for this process tree
export PYTHONUNBUFFERED=1        # timely log output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export YOLO_OFFLINE=1            # also set by the trainer; fail fast on downloads

echo "job ${SLURM_JOB_ID:-n/a} on $(hostname) at $(date -Is)"
echo "config=$CONFIG python=$PYTHON runs_root=$RUNS_ROOT view=$VIEW_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

# --- Resume an existing run ------------------------------------------------------
if [[ -n "${FIREWATCH_RESUME:-}" ]]; then
    [[ -f "$FIREWATCH_RESUME" ]] || fail "FIREWATCH_RESUME checkpoint not found: $FIREWATCH_RESUME"
    echo "resuming from $FIREWATCH_RESUME"
    exec "$PYTHON" scripts/train.py --config "$CONFIG" --resume "$FIREWATCH_RESUME" "$@"
fi

# --- Dataset view: reuse, or build once from the raw dataset -------------------
if [[ ! -f "$VIEW_DIR/data.yaml" ]]; then
    [[ -n "${FIREWATCH_DATA_ROOT:-}" ]] || fail "no dataset view at $VIEW_DIR and \
FIREWATCH_DATA_ROOT is not set to build one (see docs/training.md)"
    echo "building dataset view $VIEW_DIR from $FIREWATCH_DATA_ROOT"
    "$PYTHON" scripts/prepare_data.py --config "$CONFIG" \
        --data-root "$FIREWATCH_DATA_ROOT" --output-dir "$VIEW_DIR" \
        --report "$VIEW_DIR.validation.json" \
        ${FIREWATCH_LINK_MODE:+--link-mode "$FIREWATCH_LINK_MODE"}
fi

# --- Train ------------------------------------------------------------------------
# Run name <config stem>-<job id>; a --name in the extra args takes precedence.
RUN_NAME="$(basename "$CONFIG" .yaml)-${SLURM_JOB_ID:-local}"
exec "$PYTHON" scripts/train.py \
    --config "$CONFIG" \
    --data "$VIEW_DIR/data.yaml" \
    --project "$RUNS_ROOT" \
    --name "$RUN_NAME" \
    --device 0 \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"
