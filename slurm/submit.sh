#!/bin/bash
# ---------------------------------------------------------------------------
# Submit a FireWatch-AI job with SLURM logs OUTSIDE the repository.
#
# SLURM opens --output/--error before the job script runs and does not create
# missing directories, so this wrapper creates the log directory first and
# passes absolute log paths to sbatch (they override the #SBATCH defaults).
#
# Usage (from the repository root on the login node):
#   slurm/submit.sh train    [CONFIG] [extra scripts/train.py args...]
#   slurm/submit.sh evaluate [scripts/evaluate.py args...]
#
# Extra sbatch options can be given through SBATCH_* environment variables,
# e.g.  SBATCH_TIMELIMIT=02:00:00 slurm/submit.sh train configs/smoke_test.yaml
# All FIREWATCH_* variables in your shell are exported to the job.
# ---------------------------------------------------------------------------
set -euo pipefail

usage() { sed -n '9,11p' "$0" >&2; exit 2; }
[[ $# -ge 1 ]] || usage
kind="$1"; shift
case "$kind" in
    train)    script=slurm/train_gpu03.sh ;;
    evaluate) script=slurm/evaluate_gpu03.sh ;;
    *)        usage ;;
esac

[[ -f "$script" && -f pyproject.toml ]] || { echo "ERROR: run from the repository root" >&2; exit 2; }
command -v sbatch >/dev/null || { echo "ERROR: sbatch not found (run on the cluster login node)" >&2; exit 2; }

LOG_DIR="${FIREWATCH_RUNS_ROOT:-$HOME/firewatch_runs}/slurm_logs"
mkdir -p "$LOG_DIR"

exec sbatch --export=ALL \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    "$script" "$@"
