# Project architecture

## Goal

Detect fire and smoke in CCTV frames and still images: an object detector
trained on D-Fire (YOLO-format boxes, classes `smoke` and `fire`), later applied
to video streams.

## Current layout

```
configs/baseline.yaml            dataset, prepare and train settings
configs/smoke_test.yaml          extends baseline: 2 epochs, full train split
constraints/cluster-torch.txt    pins torch/torchvision during pip installs
src/firewatch/
  data/yolo.py                   YOLO label parsing, coordinate conversion
  data/dfire.py                  split scanning / validation
  data/yolo_view.py              sanitised, linked training view + manifest
  training/options.py            CLI/config resolution and validation
  training/preflight.py          pre-import checks: data YAML, weights, output paths
  training/metadata.py           provenance: git, versions, GPU, hashes
  training/runner.py             Ultralytics training (lazy import, offline mode)
  evaluation/evaluate.py         Ultralytics validation, per-class metrics, test guard
  utils/config.py                YAML loading with `extends:`
  utils/paths.py                 path expansion, "outside the repo" checks
scripts/prepare_data.py          validate; --output-dir builds the view
scripts/train.py                 train / resume / --dry-run
scripts/evaluate.py              evaluate val (default) or test (--final-test)
slurm/submit.sh                  creates log dir, calls sbatch
slurm/train_gpu03.sh             1-GPU training job (builds view if missing)
slurm/evaluate_gpu03.sh          1-GPU evaluation job
tests/                           pytest; stubs replace Ultralytics, no downloads
```

The code uses a `src/` layout and is installed with `pip install -e .`, so
imports are always `firewatch.*` and never depend on the working directory.

## Design rules

- **Config-driven.** Experiments are YAML files in `configs/`, and command-line
  options override them. Defaults are written down, not implicit.
- **Raw data is read-only.** Training reads a derived view. Every label change it
  makes is recorded in the view manifest.
- **No data or artifacts in Git.** Views, runs and logs live outside the
  repository; the code refuses output paths inside it.
- **Offline by construction.** Weights must be local files. Ultralytics runs with
  `YOLO_OFFLINE=1` and `YOLO_AUTOINSTALL=false`. Torch is never installed by this
  project.
- **Fail before the GPU.** All checks on options, data, weights and paths run
  before Ultralytics is imported (`--dry-run` stops there).
- **Validation and test are separate.** Model selection uses val; test needs an
  explicit flag and is logged.
- **Provenance.** Every run records the commit, dirty diff, merged config, seed,
  versions, GPU and hashes.

## Planned (not implemented)

| Subpackage | Purpose |
| --- | --- |
| `firewatch.inference` | Image / video / RTSP inference for CCTV |

The detection framework is Ultralytics YOLO (AGPL-3.0); see
[training.md](training.md#licensing). It is isolated in
`training/runner.py` and `evaluation/evaluate.py` and imported lazily, so the
data tooling does not depend on it.
