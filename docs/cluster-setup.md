# Cluster setup: Bhavani GPU cluster

## Known facts

| Item | Value |
| --- | --- |
| Login host | `master` |
| GPU partition | `gpu03` |
| Observed GPU node | `node003` |
| GPU | NVIDIA A30, 24 GB VRAM |
| Python env | `~/envs/firewatch`, Python 3.10.22 |
| PyTorch | 2.5.1+cu121, torchvision 0.20.1+cu121 (CUDA verified) |
| Repo location | `~/projects/FireWatch-AI` |
| Run outputs | `~/firewatch_runs/` (outside the repo) |
| Datasets | `~/datasets/` (outside the repo) |

Compute nodes may have **no internet access**. Training runs **only** on compute
nodes through SLURM, never on the login node.

Things to confirm on the cluster (not verified by this repo):

```bash
scontrol show partition gpu03     # MaxTime, default memory, allowed accounts
sinfo -p gpu03 -N -l              # nodes, CPUs, memory per node
scontrol show node node003        # CPUTot, RealMemory, Gres string (e.g. gpu:a30:1)
```

The `#SBATCH` values in `slurm/train_gpu03.sh` (`--time=12:00:00`, `--mem=32G`,
`--cpus-per-task=8`, `--gres=gpu:1`) are **provisional placeholders** that have
not been checked against the `gpu03` policy. Adjust them to the output above
before the first real job.

## First-time setup

```bash
ssh <user>@<bhavani-login-address>        # lands on host "master"
mkdir -p ~/projects ~/datasets ~/firewatch_runs
git clone https://github.com/Parvptl/FireWatch_AI.git ~/projects/FireWatch-AI
cd ~/projects/FireWatch-AI
```

The GitHub repo is named `FireWatch_AI` (underscore). Cloning into
`FireWatch-AI` (hyphen) is intentional and matches the paths in these docs.

### Install the project into the existing environment

Do **not** reinstall or upgrade torch or torchvision. `pyproject.toml` does not
list them, and `constraints/cluster-torch.txt` pins the installed builds
(`torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`). If any package would need
a different torch, pip **fails** instead of replacing the working CUDA build.
Use the constraints file for every install into this environment.

On the login node (it has internet; the environment lives in your home directory,
so compute nodes see the same packages):

```bash
cd ~/projects/FireWatch-AI
~/envs/firewatch/bin/python -m pip install -e ".[train,dev]" -c constraints/cluster-torch.txt
~/envs/firewatch/bin/python -c "import torch, ultralytics, cv2; print(torch.__version__, torch.version.cuda, ultralytics.__version__, cv2.__version__)"
```

Expected: `2.5.1+cu121 12.1 8.4.176 …`. If torch shows another version, stop:
the constraints were not applied.

The `train` extra installs `ultralytics==8.4.176` and its dependencies (OpenCV,
matplotlib, polars, …). A resolution check on 2026-10-13 against Linux,
Python 3.10 and the pinned torch picked `opencv-python 5.0.0.93`,
`polars 2.0.0` and `numpy 2.2.6`. That was a resolution check only. These versions
have **not** been installed or run with this pipeline yet; the smoke test is the
first real check.

**Headless OpenCV.** `opencv-python` needs `libGL.so.1`, which headless compute
nodes often lack. Test it on a compute node:

```bash
srun -p gpu03 --gres=gpu:1 --time=00:05:00 ~/envs/firewatch/bin/python -c "import cv2; print(cv2.__version__)"
```

If that fails with `libGL.so.1: cannot open shared object file`, swap in the
headless build of the **same version** (it provides the same `cv2` module):

```bash
V=$(~/envs/firewatch/bin/python -c "import importlib.metadata as m; print(m.version('opencv-python'))")
~/envs/firewatch/bin/python -m pip uninstall -y opencv-python
~/envs/firewatch/bin/python -m pip install "opencv-python-headless==$V" -c constraints/cluster-torch.txt
```

The project sets `YOLO_AUTOINSTALL=false`, so Ultralytics will not try to
reinstall packages at runtime.

If `~/envs/firewatch` is a conda environment rather than a venv, the
`bin/python` path is the same, and the commands above work unchanged.

### Offline installation (wheelhouse)

Use this only if the target machine cannot reach PyPI. Let pip compute exactly
what is missing from **your** environment (torch already satisfied), then
download only those wheels.

On a machine with internet and the same platform (the login node):

```bash
cd ~/projects/FireWatch-AI
PY=~/envs/firewatch/bin/python
$PY -m pip install --dry-run --report /tmp/fw-report.json \
    -e ".[train,dev]" -c constraints/cluster-torch.txt >/dev/null
$PY -c "import json; r=json.load(open('/tmp/fw-report.json')); print('\n'.join(f\"{i['metadata']['name']}=={i['metadata']['version']}\" for i in r['install'] if i['metadata']['name'].lower() not in {'firewatch','torch','torchvision','triton'} and not i['metadata']['name'].lower().startswith('nvidia-')))" > /tmp/fw-reqs.txt
$PY -m pip download --no-deps -d wheelhouse -r /tmp/fw-reqs.txt "setuptools>=64" wheel
```

Copy `wheelhouse/` to the target (e.g. with `scp -r`), then install without an index:

```bash
$PY -m pip install --no-index --find-links wheelhouse "setuptools>=64" wheel
$PY -m pip install --no-index --find-links wheelhouse --no-build-isolation \
    -e ".[train,dev]" -c constraints/cluster-torch.txt
```

If you must download on Windows instead, add
`--platform manylinux2014_x86_64 --python-version 3.10 --only-binary=:all:` to
`pip download`. `wheelhouse/` is gitignored.

Record the exact working environment for reproducibility. The trainer also
records key package versions in every `firewatch_run.json`.

```bash
~/envs/firewatch/bin/python -m pip freeze > ~/firewatch_runs/env-freeze-$(date +%F).txt
```

### Model weights

Training uses COCO-pretrained **YOLO26n** (5,544,453 bytes, AGPL-3.0). The
source is the Ultralytics assets release `v8.4.0`, the same URL Ultralytics
8.4.176 itself uses:

```
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
```

Download it where there is internet access (login node or laptop) and put it at
the path in `configs/baseline.yaml` (`train.weights: ~/firewatch_weights/yolo26n.pt`):

```bash
mkdir -p ~/firewatch_weights
curl -fL -o ~/firewatch_weights/yolo26n.pt \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
ls -l ~/firewatch_weights/yolo26n.pt        # expect 5544453 bytes
sha256sum ~/firewatch_weights/yolo26n.pt    # recorded in firewatch_run.json for every run
```

If you downloaded it on the laptop, copy it with
`scp yolo26n.pt <user>@<cluster>:~/firewatch_weights/`.

The trainer **never downloads**. If the file is missing, it exits with code 2
and prints this URL.

**AMP self-check.** Before training, Ultralytics 8.4 runs an AMP sanity check
that loads `yolo26n.pt` from *its own* settings `weights_dir`. It does not use
the training weights path. Offline, it logs "checks skipped" and keeps AMP on,
which is fine on an A30. `firewatch_run.json` records where it looked
(`amp_check.weights`). To let the check run, copy the same file there once:

```bash
~/envs/firewatch/bin/python -c "from ultralytics.utils import WEIGHTS_DIR; print(WEIGHTS_DIR)"
```

## Running jobs

See [training.md](training.md) for the experimental procedure and
[development-workflow.md](development-workflow.md) for the Git loop.

```bash
cd ~/projects/FireWatch-AI
export FIREWATCH_DATA_ROOT=$HOME/datasets/dfire        # raw D-Fire (to build the view)
export FIREWATCH_VIEW_DIR=$HOME/datasets/dfire_view    # sanitised view (built once)
export FIREWATCH_RUNS_ROOT=$HOME/firewatch_runs
bash slurm/submit.sh train configs/smoke_test.yaml
```

`slurm/submit.sh` creates `$FIREWATCH_RUNS_ROOT/slurm_logs/` and submits with
absolute `--output`/`--error` paths. SLURM opens log files *before* the job
starts and does not create directories, so a direct `sbatch` needs
`mkdir -p slurm_logs` in the repo root first.

`slurm/train_gpu03.sh`:

- requests 1 GPU on `gpu03`, with 8 CPUs, 32 GB RAM and 12 h. These values are
  **provisional** (see above);
- uses `~/envs/firewatch/bin/python` explicitly (override with `FIREWATCH_PYTHON`);
- builds the dataset view from `FIREWATCH_DATA_ROOT` if `$FIREWATCH_VIEW_DIR/data.yaml`
  does not exist yet (`FIREWATCH_LINK_MODE` overrides `prepare.link_mode`), and
  otherwise reuses it;
- runs `scripts/train.py --config … --data $FIREWATCH_VIEW_DIR/data.yaml
  --project $FIREWATCH_RUNS_ROOT --name <config-stem>-<jobid> --device 0`. Extra
  arguments after the config are passed on and take precedence;
- resumes instead when `FIREWATCH_RESUME=<run_dir>/weights/last.pt` is set;
- needs no internet access (`YOLO_OFFLINE=1`).

`slurm/evaluate_gpu03.sh` passes all its arguments to `scripts/evaluate.py`:
`bash slurm/submit.sh evaluate --weights … --data …`.

Building the view (about 21.5k label files plus links) and data validation need no
GPU. Run them in a short allocation rather than on the login node:

```bash
srun -p gpu03 --cpus-per-task=2 --mem=4G --time=00:20:00 \
    ~/envs/firewatch/bin/python scripts/prepare_data.py --config configs/baseline.yaml \
    --data-root ~/datasets/dfire --output-dir ~/datasets/dfire_view
```
