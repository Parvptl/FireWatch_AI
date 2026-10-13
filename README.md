# FireWatch-AI

Fire and smoke detection for CCTV footage and images, being developed with the
public [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset) dataset.
The intended use is early visual warning of fire and smoke in camera feeds. It is
a research and student project, not a certified safety system.

> **Naming:** the GitHub repository is `Parvptl/FireWatch_AI` (underscore), the
> project is called FireWatch-AI, and the Python package is `firewatch`. The
> cluster clone lives at `~/projects/FireWatch-AI` (hyphen), cloned there explicitly.
> Local clones may use any directory name.

## Implementation status

| Component | Status |
| --- | --- |
| YOLO label parsing and coordinate conversion | Implemented, tested |
| D-Fire dataset validation and statistics (`scripts/prepare_data.py`) | Implemented, tested on synthetic data |
| Config loading (`configs/baseline.yaml`) | Implemented, tested |
| SLURM job template (`slurm/train_gpu03.sh`) | Ready, but waits for a trainer and exits with an error until then |
| Model / training (`scripts/train.py`) | **Not implemented**. The detection framework is not chosen yet |
| Evaluation | **Not implemented** |
| Video / CCTV inference | **Not implemented** |

No models have been trained yet, and there are no results or benchmark numbers.

## Repository structure

```
configs/          experiment configs (baseline.yaml)
docs/             architecture, workflow, cluster setup, dataset guide
scripts/          command-line entry points (prepare_data.py)
slurm/            SLURM job scripts for the Bhavani cluster
src/firewatch/    Python package (data/, utils/)
tests/            pytest suite
```

These paths are local only and gitignored: `data/`, `outputs/`, `checkpoints/`,
`slurm_logs/`, `DFireDataset/` (reference clone of the upstream metadata repo),
and `.venv/`.

## Environment

Requires Python ≥ 3.10. The cluster uses 3.10.22 with torch 2.5.1+cu121. Torch is
**not** a declared dependency, because the current code does not need it and the
cluster already has a verified CUDA build.

Local (Windows, Git Bash):

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv -e ".[dev]"
```

Cluster:

```bash
~/envs/firewatch/bin/python -m pip install -e ".[dev]"
```

Offline installation is covered in [docs/cluster-setup.md](docs/cluster-setup.md).

## Dataset

D-Fire images and YOLO labels (`0 = smoke`, `1 = fire`) are downloaded separately
and kept outside the repo. See [docs/dataset-guide.md](docs/dataset-guide.md) for
sources, layout and licence. Validate a download with:

```bash
python scripts/prepare_data.py --config configs/baseline.yaml --data-root /path/to/dfire \
    --report outputs/dfire_report.json --write-data-yaml outputs/dfire_data.yaml
```

## Training and evaluation

Not available yet. `slurm/train_gpu03.sh` defines how training will be launched
on the cluster once `scripts/train.py` exists:

```bash
mkdir -p slurm_logs
export FIREWATCH_DATA_ROOT=$HOME/datasets/dfire
sbatch slurm/train_gpu03.sh configs/baseline.yaml
```

## Cluster workflow

Develop locally, push to GitHub, run `git pull --ff-only` on the login node,
submit with `sbatch`, and train on `gpu03` (1 × A30). See
[docs/development-workflow.md](docs/development-workflow.md) and
[docs/cluster-setup.md](docs/cluster-setup.md).

## Experiment artifacts

Each SLURM run writes to `~/firewatch_runs/<job-name>-<jobid>/`. That includes
checkpoints, logs, a config copy, the git commit, the seed and environment info.
Nothing is written inside the repository except gitignored SLURM logs. Copy back
only the metrics and checkpoints you need.

## Tests

With the project environment active (`source .venv/Scripts/activate` in Git Bash
on Windows, `source .venv/bin/activate` on Linux):

```bash
python -m pytest
ruff check . && ruff format --check .
bash -n slurm/train_gpu03.sh
```

The same checks run on GitHub Actions (`.github/workflows/ci.yml`).

## Known limitations and next steps

- Choose the detection framework and implement `scripts/train.py` and
  `scripts/evaluate.py`, following the CLI contract in `slurm/train_gpu03.sh`.
- Download D-Fire to the cluster, run `prepare_data.py` on it, and confirm the
  split directory names in `configs/baseline.yaml`.
- Confirm the `gpu03` partition limits (time and memory) and tune the `#SBATCH`
  values.
- Pre-download any pretrained weights, because compute nodes may be offline.
- Video/CCTV inference is not implemented yet.

## Acknowledgements

D-Fire dataset © Gaia, solutions on demand, CC0-1.0. Cite de Venâncio et al.,
*Neural Computing and Applications* (2022) when using it.
