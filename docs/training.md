# Training and evaluation

FireWatch-AI trains a single-GPU **Ultralytics YOLO** detector (baseline:
**YOLO26n**, COCO-pretrained) on D-Fire with two classes, `0 = smoke` and
`1 = fire`. No model has been trained yet, so this document describes the
procedure and reports no results.

## Pipeline

```
raw D-Fire (never modified)
   │  scripts/prepare_data.py --output-dir VIEW      (validate, then build view)
   ▼
dataset view: linked images + sanitised labels + data.yaml + firewatch_view.json
   │  scripts/train.py --data VIEW/data.yaml         (train; select on val)
   ▼
<runs_root>/<name>/  weights/best.pt, last.pt, results.csv, firewatch_run.json, …
   │  scripts/evaluate.py --split val                (model analysis)
   │  scripts/evaluate.py --split test --final-test  (once, at the end)
   ▼
<run_dir>/eval/<split>-<ckpt>-<timestamp>/metrics.json
```

## 1. Dataset view (why training never reads the raw dataset)

Two behaviours of Ultralytics 8.4.176 were verified in its source:

- It writes `labels.cache` next to the label directory, which would modify the
  dataset tree.
- It discards a **whole image** as "corrupt" when any label value exceeds 1.01,
  but it keeps zero-size boxes.

`scripts/prepare_data.py --output-dir` therefore builds a separate view.
`src/firewatch/data/yolo_view.py` documents the full policy, and
`tests/test_yolo_view.py` tests it:

| Case | Action |
| --- | --- |
| Box corner outside [0, 1] | Clip the corners to the image and write the clipped box |
| Clipped width or height ≤ `min_box_size` (default 0) | Drop the box |
| Every box of an image dropped | Exclude the image (it is not turned into a "background" image) |
| Empty label file | Keep as background |
| Missing label file / duplicate image stem | Exclude the image |
| Malformed line / unknown class id | Abort |
| Unchanged box | Copy the box text verbatim (no float reformatting) |

Every label file in the view is written with LF line endings, so CRLF source
files differ byte-wise from their view copies even when no box changed. This is
not recorded as a change in `firewatch_view.json`.

Every change (file, line, before, after) and every exclusion is listed in
`firewatch_view.json`. Images are per-file links to the source by default
(`prepare.link_mode`):

- `symlink` (Linux default);
- `hardlink` (needs the same filesystem; works on Windows without extra rights);
- `copy` (duplicates about 3 GB).

A *directory* symlink would not work: Ultralytics resolves the split directory
path back to the raw labels.

**JPEG repair inside the view.** When Ultralytics first scans a split, it
re-encodes any JPEG whose file does not end with the `FF D9` end marker (log:
`corrupt JPEG restored and saved`). It writes a temporary file and
`os.replace`s the *view entry*, so the link becomes a regular file and the raw
image is not touched. In D-Fire, 21 train, 5 val and 11 test JPEGs carry an
18-byte `MediaType=default` trailer and are replaced this way on first use;
`firewatch_view.json` does not record it. Later runs replay the message from
`labels.cache` without rewriting the file. The raw dataset can be checked
against a SHA-256 manifest under `~/firewatch_runs/integrity/`.

Result on the local D-Fire copy (2026-10-13, hard links, about 40 s): all 21,527
images kept, 0 excluded. Boxes changed:

| Split | Clipped | Dropped (zero size) |
| --- | --- | --- |
| train | 255 | 13 |
| val | 62 | 1 |
| test | 62 | 4 |

All source label files were byte-identical afterwards. The clip count is larger
than the validator's 26 warnings, because the validator checks
centre/width/height while the view checks box *corners*. Every clipped box
extended at least about 0.00075 of the image size (about 1 px) past an edge, and
up to 0.094. These are real overshoots, not floating-point noise.

```bash
python scripts/prepare_data.py --config configs/baseline.yaml \
    --data-root ~/datasets/dfire --output-dir ~/datasets/dfire_view_v2
```

`data.yaml` contains absolute paths, so build the view **on the machine that
trains**. Rebuild it with `--overwrite`, which deletes only a directory that
carries a FireWatch manifest. The trainer refuses YAMLs that:

- are not a view (`--allow-unprepared-data` overrides this);
- have relative or missing paths, or dangling image links;
- have a class order other than `[smoke, fire]`;
- reuse the same directory for train/val/test;
- contain a `download:` key, which Ultralytics would execute.

## 2. Pretrained weights

`train.weights` (default `~/firewatch_weights/yolo26n.pt`) must already exist.
The trainer never downloads weights. It sets `YOLO_OFFLINE=1` and
`YOLO_AUTOINSTALL=false` before importing Ultralytics, and if the file is
missing it exits with code 2 and prints the download URL. To fetch the weights,
see [cluster-setup.md](cluster-setup.md#model-weights).

## 3. Experimental design

- **Smoke test first**: `configs/smoke_test.yaml` extends the baseline. It runs
  2 epochs on the full train split and validates on the full val split (about
  45 s per epoch on an A30). It checks the environment, GPU memory at
  `batch: 32`, checkpointing and evaluation. Its metrics are not results.
- **`fraction` is not a random sample.** Ultralytics 8.4 keeps the first N
  images in sorted file order. On D-Fire, `fraction: 0.1` selected 1,407
  background images out of 1,412, so avoid `--fraction` for anything that
  should measure learning.
- **Baseline**: `configs/baseline.yaml`. Settings: 640 px, up to 100 epochs,
  early stopping after 20 epochs without val improvement, batch 32 (provisional
  for an A30), seed 42, `optimizer: auto`, AMP on.
- **Model selection uses val only.** Ultralytics keeps `best.pt` = the epoch with
  the highest val fitness, which is mAP50-95 in 8.4.176. Training never passes a
  test split.
- **Test is held out.** `scripts/evaluate.py --split test` refuses to run without
  `--final-test`. Every test evaluation of a run is appended to
  `<run_dir>/test_set_evaluations.jsonl`, so repeated looks at the test set stay
  visible. Tune hyperparameters on val, then evaluate the chosen checkpoint on
  test once.
- **Label policy and test metrics**: test labels are sanitised by the same rules
  (62 clipped and 4 dropped boxes). Report this alongside test metrics.

## 4. Running

Locally or on the login node, validate without training (Ultralytics is not
imported):

```bash
python scripts/train.py --config configs/smoke_test.yaml --data ~/datasets/dfire_view_v2/data.yaml --dry-run
```

On the cluster, always go through SLURM:

```bash
bash slurm/submit.sh train configs/smoke_test.yaml
bash slurm/submit.sh train configs/baseline.yaml
```

The run directory is `$FIREWATCH_RUNS_ROOT/<config-stem>-<jobid>`. Any
`scripts/train.py` option can follow the config, e.g.
`bash slurm/submit.sh train configs/baseline.yaml --batch 64`.

The trainer refuses to reuse a run directory that already contains a run.

### Resume

Ultralytics resumes from `<run_dir>/weights/last.pt` with the **checkpoint's**
settings, into the same run directory. FireWatch makes this explicit:

- `--resume` cannot be combined with `--epochs`, `--seed`, `--batch`, `--name`
  or similar options; this fails rather than being silently ignored;
- only `--device` and `--workers` may change;
- each resume writes its own `firewatch_resume_<timestamp>.json`.

```bash
FIREWATCH_RESUME=$HOME/firewatch_runs/baseline-12345/weights/last.pt \
    bash slurm/submit.sh train configs/baseline.yaml
```

## 5. What a run directory contains

| File | Written by | Content |
| --- | --- | --- |
| `firewatch_run.json` | FireWatch | status, options, exact Ultralytics kwargs, git commit/branch/dirty, package versions, CUDA/GPU, SLURM job, weights SHA-256, seed, final val metrics, best and last checkpoints + SHA-256 |
| `firewatch_config.yaml` | FireWatch | fully merged config (after `extends`) |
| `uncommitted_<ts>.diff` | FireWatch | only if the working tree was dirty |
| `args.yaml`, `results.csv`, plots | Ultralytics | training arguments and per-epoch metrics |
| `weights/best.pt`, `weights/last.pt` | Ultralytics | checkpoints |
| `eval/<split>-…/metrics.json` | evaluate.py | overall and per-class P, R, mAP50, mAP75, mAP50-95, speed |

## 6. Evaluation

```bash
python scripts/evaluate.py --weights ~/firewatch_runs/<run>/weights/best.pt \
    --data ~/datasets/dfire_view_v2/data.yaml                   # val
python scripts/evaluate.py --weights ~/firewatch_runs/<run>/weights/best.pt \
    --data ~/datasets/dfire_view_v2/data.yaml --split test --final-test
```

Defaults: `conf=0.001` (standard for mAP), `iou=0.7` (Ultralytics default), and
`imgsz` taken from the run's `args.yaml`. The script checks that the checkpoint's
class names are `[smoke, fire]`. On the cluster, use
`bash slurm/submit.sh evaluate <same args>`.

## Reproducibility

Recorded per run: seed, the `deterministic` flag, the full config, Ultralytics
kwargs, git commit and dirty state (plus the diff), package versions, the CUDA
build, the GPU name, and SHA-256 hashes of the start weights and `best.pt`.

What is **not** guaranteed:

- Ultralytics seeds with `seed + 1 + RANK` and uses
  `torch.use_deterministic_algorithms(True, warn_only=True)`. Some CUDA
  kernels have no deterministic version, so two runs on the same A30 can still
  differ slightly.
- Runs on different GPUs or with different library versions will differ.
- Dataloader worker scheduling and AMP add further small variation.

Treat run-to-run differences of this size as noise. Compare models with
repeated seeds when differences are small.

## Licensing

Ultralytics (the package and the YOLO26 weights) is licensed under **AGPL-3.0**,
unless you hold an Ultralytics Enterprise licence. In practice:

- Training and evaluating privately for coursework or research is not affected.
- If you **distribute** FireWatch-AI with Ultralytics, or software containing
  the trained model, the combined work must be released under AGPL-3.0, with
  source.
- If you offer it to users **over a network** (e.g. a monitoring web service),
  the same applies. AGPL-3.0 §13 covers network use.
- The FireWatch-AI repository currently has **no LICENSE file**. Choose one that
  is compatible (e.g. AGPL-3.0) before publishing code that depends on
  Ultralytics, or keep the `train` extra separate from any non-AGPL
  distribution.

This is a summary, not legal advice. Ultralytics' licensing page and your
institution are the authorities. Ultralytics is an optional extra
(`.[train]`); the data tooling has no AGPL dependency.
