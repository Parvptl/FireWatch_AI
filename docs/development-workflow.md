# Development workflow

Code is written and tested on the Windows laptop, versioned on GitHub, and
trained on the Bhavani GPU cluster through SLURM (once a trainer exists).

```
laptop (Claude Code, tests) --push--> GitHub --pull--> cluster login node --sbatch--> gpu03 compute node
                                                                                           |
                       selected metrics / checkpoints <------- scp/rsync ------- ~/firewatch_runs/
```

## 1. Edit and test locally

Claude Code edits source code in the local clone. Run the lightweight checks
before committing:

```bash
.venv/Scripts/python -m pytest          # Windows Git Bash; use .venv/bin/python on Linux/macOS
.venv/Scripts/ruff check .
.venv/Scripts/ruff format --check .
bash -n slurm/train_gpu03.sh
```

Create the local environment once. Python 3.10 matches the cluster.

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv -e ".[dev]"
```

(`python -m venv .venv` followed by `pip install -e ".[dev]"` works too.)

## 2. Review changes

```bash
git status
git diff            # unstaged
git diff --staged   # staged
```

Make sure no data, weights or outputs are staged. `.gitignore` covers the usual
paths, but check anyway.

## 3. Commit and push

```bash
git add <files>
git commit -m "Describe the change"
git push origin main
```

## 4. Connect to the cluster login node

```bash
ssh <user>@<bhavani-login-address>    # host "master"
```

## 5. Pull safely (fast-forward only)

```bash
cd ~/projects/FireWatch-AI
git status                  # should be clean; never edit tracked files here without committing
git pull --ff-only origin main
```

`--ff-only` refuses to create merge commits or silently combine diverged
histories. If it fails, inspect with `git status` and `git log --oneline --graph --all`
instead of forcing anything.

## 6. Lightweight checks on the login node

Cheap checks only, with no training and no heavy data processing:

```bash
~/envs/firewatch/bin/python -m pip install -e ".[dev]"   # only when dependencies change
~/envs/firewatch/bin/python -m pytest -q
bash -n slurm/train_gpu03.sh
```

## 7. Submit jobs

Training **must** run on an allocated compute node through `sbatch` (or `srun`),
never directly on the login node. The login node is shared and has no allocated GPU.

```bash
mkdir -p slurm_logs
export FIREWATCH_DATA_ROOT=$HOME/datasets/dfire
sbatch slurm/train_gpu03.sh configs/baseline.yaml
```

(The training entry point is not implemented yet. See the README.)

## 8. Monitor

```bash
squeue -u $USER
tail -f slurm_logs/firewatch-train-<jobid>.out
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
scancel <jobid>            # if needed
```

## 9. Keep data and artifacts out of the repo

| What | Where |
| --- | --- |
| Datasets | `~/datasets/` |
| Run outputs and checkpoints | `~/firewatch_runs/<job-name>-<jobid>/` |
| SLURM stdout/stderr | `slurm_logs/` (in repo dir, gitignored) |

## 10. Retrieve results

From the laptop, copy back only what you need:

```bash
scp <user>@<cluster>:~/firewatch_runs/<run>/run_info.txt ./outputs/<run>/
rsync -av --include='*/' --include='*.json' --include='*.csv' --include='*.yaml' --exclude='*' \
    <user>@<cluster>:~/firewatch_runs/<run>/ ./outputs/<run>/
scp <user>@<cluster>:~/firewatch_runs/<run>/checkpoints/best.pt ./checkpoints/<run>/
```

`outputs/` and `checkpoints/` are gitignored locally. (`rsync` may not be
available in Git Bash on Windows. Use `scp` or WSL.)

## Alternative: develop directly on the cluster over SSH

You can also edit on the cluster itself, either with VS Code Remote-SSH or with
an editor in an SSH session. That avoids the push and pull round-trip. Things to
keep in mind:

- Edit and run quick checks on the login node, but still submit training with `sbatch`.
- Commit from the cluster clone too, and push to GitHub so the laptop stays in sync
  (`git pull --ff-only` on the laptop).
- Never keep diverging uncommitted edits in both places.
- Running Claude Code on the cluster is a possible future step. It is not set up
  here, and it would need the login node's internet access and the cluster's usage
  policy to be checked first.
