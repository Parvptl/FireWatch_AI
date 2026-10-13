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
| YOLO label parsing, coordinate conversion | Implemented, tested |
| D-Fire validation and statistics (`scripts/prepare_data.py`) | Implemented, tested; run on the full local D-Fire copy |
| Sanitised training view (`prepare_data.py --output-dir`) | Implemented, tested; built from the full local D-Fire copy |
| Training (`scripts/train.py`, Ultralytics YOLO26n) | Implemented; options, preflight and provenance are tested. **Not yet run with Ultralytics/GPU** |
| Evaluation (`scripts/evaluate.py`) | Implemented; CLI and metrics extraction tested with a stub. **Not yet run with Ultralytics/GPU** |
| SLURM jobs (`slurm/`) | Implemented; scripts tested with bash outside SLURM. **No job submitted yet** |
| Video / CCTV inference | **Not implemented** |

No model has been trained yet. There are no results, accuracy figures or
benchmark numbers.

## Repository structure

```
configs/          baseline.yaml, smoke_test.yaml (extends baseline)
constraints/      pip constraints protecting the cluster's torch build
docs/             training, dataset, cluster setup, workflow, architecture
scripts/          prepare_data.py, train.py, evaluate.py
slurm/            submit.sh, train_gpu03.sh, evaluate_gpu03.sh
src/firewatch/    data/ (labels, validation, view), training/, evaluation/, utils/
tests/            pytest suite (no downloads, no real training)
```

These paths are local only and gitignored: `data/`, `outputs/`, `checkpoints/`,
`runs/`, `weights/`, `slurm_logs/`, `DFireDataset/` (upstream clone plus your local
data) and `.venv/`.

## Environment

Requires Python ≥ 3.10. The cluster uses 3.10.22 with torch 2.5.1+cu121. Torch is
**not** a declared dependency and must not be reinstalled. Training needs the
optional `train` extra (`ultralytics==8.4.176`, AGPL-3.0; see
[Licensing](docs/training.md#licensing)).

Cluster (the login node has internet):

```bash
~/envs/firewatch/bin/python -m pip install -e ".[train,dev]" -c constraints/cluster-torch.txt
```

Local development and tests need no torch or Ultralytics:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv -e ".[dev]"
```

Offline installation, model weights and headless OpenCV are covered in
[docs/cluster-setup.md](docs/cluster-setup.md).

## Dataset

D-Fire images and YOLO labels (`0 = smoke`, `1 = fire`) are downloaded separately
and kept outside the repo ([docs/dataset-guide.md](docs/dataset-guide.md)).
Validate the download and build the sanitised training view:

```bash
python scripts/prepare_data.py --config configs/baseline.yaml --data-root /path/to/dfire \
    --output-dir /path/to/dfire_view
```

The source dataset is never modified.

## Training and evaluation

See [docs/training.md](docs/training.md). Pretrained weights must be downloaded
beforehand, because the trainer never downloads anything. Run the smoke test first:

```bash
python scripts/train.py --config configs/smoke_test.yaml --data /path/to/dfire_view/data.yaml --dry-run
bash slurm/submit.sh train configs/smoke_test.yaml        # on the cluster
bash slurm/submit.sh train configs/baseline.yaml
python scripts/evaluate.py --weights <run_dir>/weights/best.pt --data /path/to/dfire_view/data.yaml
```

Model selection uses val. Test evaluation needs `--split test --final-test`, and
every test evaluation is logged.

## Cluster workflow

Develop locally, push to GitHub, run `git pull --ff-only` on the login node,
submit with `slurm/submit.sh`, and train on `gpu03` (1 × A30). See
[docs/development-workflow.md](docs/development-workflow.md) and
[docs/cluster-setup.md](docs/cluster-setup.md).

## Experiment artifacts

Each run writes to `$FIREWATCH_RUNS_ROOT/<config>-<jobid>/` (default
`~/firewatch_runs`). A run directory holds:

- checkpoints, `results.csv` and plots (from Ultralytics);
- `firewatch_run.json`: git commit and dirty flag, versions, GPU, seed, options
  and weight hashes;
- the merged config;
- `eval/…/metrics.json`.

SLURM logs go to `$FIREWATCH_RUNS_ROOT/slurm_logs/`. Nothing is written inside
the repository.

## Tests

With the project environment active (`source .venv/Scripts/activate` in Git Bash
on Windows, `source .venv/bin/activate` on Linux):

```bash
python -m pytest
ruff check . && ruff format --check .
for f in slurm/*.sh; do bash -n "$f"; done
```

The tests never download weights, never import Ultralytics, and never train.
Ultralytics is replaced by small stubs that check the calling contract. The same
checks run on GitHub Actions (`.github/workflows/ci.yml`).

## Known limitations and next steps

- Run the smoke test on `gpu03`. It is the first real test of Ultralytics 8.4.176
  with the cluster's torch, OpenCV and dependencies, and of GPU memory at
  `batch: 32`.
- Confirm the `gpu03` limits and tune the provisional `#SBATCH` values (time,
  memory, CPUs, `--gres` syntax).
- Choose a LICENSE for the repository that is compatible with AGPL-3.0 if it will
  be published with Ultralytics.
- Bit-exact reproducibility is not guaranteed (see
  [Reproducibility](docs/training.md#reproducibility)).
- Video/CCTV inference is not implemented yet.

## Acknowledgements

D-Fire dataset © Gaia, solutions on demand, CC0-1.0. Cite de Venâncio et al.,
*Neural Computing and Applications* (2022) when using it.
