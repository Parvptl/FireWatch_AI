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
| Training (`scripts/train.py`, Ultralytics YOLO26n) | Implemented and run on the cluster (YOLO26n, 960 px baseline) |
| Evaluation (`scripts/evaluate.py`) | Implemented and run on the cluster |
| SLURM jobs (`slurm/`) | Implemented and used |
| Video analysis application (`scripts/serve.py`, `scripts/analyze_video.py`) | Implemented, tested; end-to-end smoke test with the real checkpoint on CPU |
| Live streams (RTSP/HTTP) | Implemented via OpenCV/FFmpeg; tested with an HTTP-served video, **not yet with a real RTSP camera** |

D-Fire is a development resource, not an independent benchmark: its splits share
incidents, videos and cameras, so D-Fire scores do not predict performance on new
CCTV cameras or sites. Software tests check the application, not detection quality.

## Repository structure

```
configs/          baseline.yaml, smoke_test.yaml (extends baseline), app.yaml (application)
constraints/      pip constraints protecting the cluster's torch build
docs/             training, dataset, cluster setup, workflow, architecture
scripts/          prepare_data.py, train.py, evaluate.py, serve.py, analyze_video.py
slurm/            submit.sh, train_gpu03.sh, evaluate_gpu03.sh
src/firewatch/    data/ (labels, validation, view), training/, evaluation/, utils/,
                  app/ (inference, events, jobs, storage, HTTP API, dashboard)
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

## Running the application

The application analyses recorded videos (and optionally live streams) with the trained
checkpoint, groups per-frame detections into fire/smoke events, raises one alert per
confirmed event, and serves the results through a JSON API and a browser dashboard.
It uses only the Python standard library plus OpenCV/Ultralytics already in the
cluster environment (no web framework). All settings are in `configs/app.yaml`.

```bash
# 1. Put authorised videos in the input directory (default ~/firewatch_app/input)
mkdir -p ~/firewatch_app/input && cp /path/to/video.mp4 ~/firewatch_app/input/

# 2. Start the API and dashboard (Ctrl-C or SIGTERM stops it cleanly)
~/envs/firewatch/bin/python scripts/serve.py --config configs/app.yaml

# 3. Open http://127.0.0.1:8000/  (from a laptop: ssh -L 8000:127.0.0.1:8000 <cluster>)
```

Use `--device 0` on a GPU node, `--port` to change the port and `--weights` (or
`FIREWATCH_MODEL_PATH`) to use another checkpoint. A non-loopback `--host` requires
`FIREWATCH_API_TOKEN`. One server per data directory.

Command-line analysis of one video (stored in the same database, so it also appears in
the dashboard):

```bash
~/envs/firewatch/bin/python scripts/analyze_video.py --config configs/app.yaml --video video.mp4
```

API examples:

```bash
curl -s localhost:8000/api/health
curl -s -X POST -H 'Content-Type: application/json' -d '{"video":"video.mp4"}' localhost:8000/api/jobs
curl -s localhost:8000/api/jobs/<job_id>             # status, progress, summary, error
curl -s localhost:8000/api/jobs/<job_id>/events      # events with start/alert/end times
curl -s localhost:8000/api/jobs/<job_id>/detections  # per-frame boxes, classes, confidences
curl -s localhost:8000/api/alerts                    # alert history
curl -s -X POST --data-binary @video.mp4 'localhost:8000/api/uploads?filename=video.mp4&submit=1'
curl -s -X POST -H 'Content-Type: application/json' -d '{"url":"rtsp://camera/stream"}' localhost:8000/api/streams
```

Event logic (`events:` in the config): a class becomes an event after `activate_hits`
analysed frames with a detection of at least `min_confidence` within
`activate_window_s`; the alert is raised once at that moment; the event ends after
`resolve_gap_s` without detections. Jobs, events and alerts are stored in SQLite under
`storage.data_dir`, with key frames, `detections.jsonl` and an annotated MP4 per job.

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
- The video application has not been validated on a real RTSP camera, and its
  detection quality on unseen CCTV cameras is unknown; an independent, rights-cleared
  CCTV evaluation is needed before any operational use.
- The commercial-use rights of individual D-Fire images are unresolved.

## Acknowledgements

D-Fire dataset © Gaia, solutions on demand, CC0-1.0. Cite de Venâncio et al.,
*Neural Computing and Applications* (2022) when using it.
