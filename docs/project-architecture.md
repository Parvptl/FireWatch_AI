# Project architecture

## Goal

Detect fire and smoke in CCTV frames and still images: an object detector
trained on D-Fire (YOLO-format boxes, classes `smoke` and `fire`), later applied
to video streams.

## Current layout

```
configs/baseline.yaml          dataset configuration (used by prepare_data.py)
src/firewatch/
  data/yolo.py                 YOLO label parsing, coordinate conversion
  data/dfire.py                D-Fire split scanning / validation, data-YAML export
  utils/config.py              YAML config loading, dataset-root resolution
scripts/prepare_data.py        CLI: validate a dataset download, write report
slurm/train_gpu03.sh           SLURM template for single-GPU training (gpu03)
tests/                         pytest suite (synthetic fixtures, no real data)
docs/                          workflow, cluster and dataset docs
```

The code uses a `src/` layout and is installed with `pip install -e .`, so
imports are always `firewatch.*` and never depend on the working directory.

## Design rules

- **Config-driven.** Experiments are defined by YAML in `configs/`. CLI flags
  override paths only.
- **No data or artifacts in Git.** Dataset roots and output directories are
  passed in through `--data-root` / `FIREWATCH_DATA_ROOT` and the SLURM run
  directory.
- **Cluster-compatible.** Python 3.10, and the existing torch 2.5.1+cu121 is
  assumed rather than installed. Nothing may download at runtime on compute
  nodes.
- **Provenance.** Every SLURM run records the git commit, dirty state, config
  copy, seed and environment versions.

## Planned (not implemented)

These subpackages will be added only together with working code:

| Subpackage | Purpose |
| --- | --- |
| `firewatch.models` | Detector construction (framework still to be chosen) |
| `firewatch.training` | Training loop / trainer wrapper → `scripts/train.py` |
| `firewatch.evaluation` | mAP, precision/recall per class → `scripts/evaluate.py` |
| `firewatch.inference` | Image / video / RTSP inference for CCTV |

Choosing the detection framework is the next decision. Ultralytics YOLO
(AGPL-3.0) and torchvision detectors (BSD) are the main candidates. Whatever is
chosen, `scripts/train.py` should accept the CLI that `slurm/train_gpu03.sh`
already passes: `--config`, `--data-root`, `--output-dir`, `--seed`.
