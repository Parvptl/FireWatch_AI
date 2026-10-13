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

Do **not** reinstall or upgrade torch or torchvision. `pyproject.toml` deliberately
omits them, so pip will not touch the working CUDA build.

On the login node (has internet):

```bash
~/envs/firewatch/bin/python -m pip install -e ".[dev]"
~/envs/firewatch/bin/python -c "import firewatch, torch; print(firewatch.__version__, torch.__version__, torch.version.cuda)"
```

If `~/envs/firewatch` is a conda environment rather than a venv, the
`bin/python` path is the same, and the commands above work unchanged.

### Offline installation (if the login node also lacks internet, or for a compute node)

Build a wheelhouse where internet is available, then install without an index.
Run all of this with the **cluster's** Python (3.10, Linux), so the wheels match
the platform:

```bash
cd ~/projects/FireWatch-AI
~/envs/firewatch/bin/python -m pip download -d wheelhouse "PyYAML>=6.0,<7" "pytest>=8,<9" "ruff>=0.17,<0.18" "setuptools>=64" wheel
# later, possibly on a node without internet:
~/envs/firewatch/bin/python -m pip install --no-index --find-links wheelhouse "setuptools>=64" wheel
~/envs/firewatch/bin/python -m pip install --no-index --find-links wheelhouse --no-build-isolation -e ".[dev]"
```

If you have to download on Windows instead, add
`--platform manylinux2014_x86_64 --python-version 3.10 --only-binary=:all:`.
`wheelhouse/` is gitignored.

Record the exact working environment for reproducibility. Keep this file outside
the repo, or commit it deliberately as `requirements-lock.txt`:

```bash
~/envs/firewatch/bin/python -m pip freeze > ~/firewatch_runs/env-freeze-$(date +%F).txt
```

### Pretrained weights on offline nodes

Any future trainer that downloads pretrained weights at runtime (torchvision
`weights=...`, Ultralytics `*.pt`) will fail on an offline compute node.
Download them on the login node first, e.g. into the torch hub cache
`~/.cache/torch/hub/checkpoints/` or a dedicated `~/weights/` directory, and
point the config at the local file.

## Running jobs

See [development-workflow.md](development-workflow.md) for the full loop. In
short:

```bash
cd ~/projects/FireWatch-AI
mkdir -p slurm_logs
export FIREWATCH_DATA_ROOT=$HOME/datasets/dfire
sbatch slurm/train_gpu03.sh configs/baseline.yaml
```

> `slurm/train_gpu03.sh` currently exits with an error because
> `scripts/train.py` is not implemented yet. It is a ready template for when it is.

The job script:

- requests 1 GPU on `gpu03`, with 8 CPUs, 32 GB RAM and 12 h. These values are
  provisional (see above);
- writes SLURM stdout and stderr to `slurm_logs/<job-name>-<jobid>.{out,err}`
  (gitignored). SLURM opens these files before the script runs and does **not**
  create the directory. If `slurm_logs/` is missing, the job fails immediately
  with no log, so always run `mkdir -p slurm_logs` first;
- needs no internet access on the compute node;
- uses `~/envs/firewatch/bin/python` explicitly, with no `activate` needed
  (override with `FIREWATCH_PYTHON`);
- creates `~/firewatch_runs/<job-name>-<jobid>/` with `checkpoints/`, `logs/`, a
  copy of the config, `run_info.txt` (git commit, dirty flag, node, versions,
  seed), `nvidia-smi.txt` and any uncommitted diff.

Lightweight data validation (no GPU needed) can run in a short interactive
allocation rather than on the login node:

```bash
srun -p gpu03 --cpus-per-task=2 --mem=4G --time=00:20:00 --pty \
    ~/envs/firewatch/bin/python scripts/prepare_data.py --data-root ~/datasets/dfire
```
