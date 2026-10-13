"""Run the SLURM job scripts with bash (outside SLURM) to exercise their logic.

Training itself is never started: the scripts are driven into failure paths or
into scripts/train.py --dry-run.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import DEFAULT_SPLITS, REPO_ROOT

BASH = shutil.which("bash")
if BASH and "system32" in BASH.lower():  # WSL launcher, not a usable POSIX bash here
    BASH = None
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def posix(p: Path | str) -> str:
    return Path(p).as_posix()


def run_script(script: str, args: list[str], env_extra: dict, tmp_path: Path):
    env = {
        **os.environ,
        "FIREWATCH_PYTHON": posix(sys.executable),
        "FIREWATCH_RUNS_ROOT": posix(tmp_path / "runs"),
        "SLURM_SUBMIT_DIR": posix(REPO_ROOT),
        "SLURM_JOB_ID": "12345",
        **env_extra,
    }
    for var in ("FIREWATCH_DATA_ROOT", "FIREWATCH_RESUME"):
        if var not in env_extra:
            env.pop(var, None)
    return subprocess.run(
        [BASH, posix(REPO_ROOT / script), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_train_job_requires_data_when_view_missing(tmp_path):
    r = run_script(
        "slurm/train_gpu03.sh",
        ["configs/smoke_test.yaml"],
        {"FIREWATCH_VIEW_DIR": posix(tmp_path / "no_view")},
        tmp_path,
    )
    assert r.returncode == 2
    assert "FIREWATCH_DATA_ROOT is not set" in r.stderr


def test_train_job_builds_view_then_dry_runs(tmp_path, make_dataset, fake_weights):
    raw = make_dataset(DEFAULT_SPLITS)
    view = tmp_path / "view"
    r = run_script(
        "slurm/train_gpu03.sh",
        [
            "configs/smoke_test.yaml",
            "--weights",
            posix(fake_weights),
            "--device",
            "cpu",
            "--dry-run",
        ],
        {
            "FIREWATCH_VIEW_DIR": posix(view),
            "FIREWATCH_DATA_ROOT": posix(raw),
            "FIREWATCH_LINK_MODE": "hardlink",  # works without symlink rights
        },
        tmp_path,
    )
    assert r.returncode == 0, r.stderr
    assert "building dataset view" in r.stdout
    assert (view / "data.yaml").is_file()
    plan = json.loads(r.stdout[r.stdout.index("\n{\n") :])  # the dry-run plan block
    kw = plan["ultralytics_train_kwargs"]
    assert kw["name"] == "smoke_test-12345"
    assert kw["device"] == "cpu"  # extra args override the script defaults
    assert kw["epochs"] == 2
    assert not (tmp_path / "runs" / "smoke_test-12345").exists()


def test_train_job_missing_resume_checkpoint(tmp_path):
    r = run_script(
        "slurm/train_gpu03.sh",
        ["configs/baseline.yaml"],
        {"FIREWATCH_RESUME": posix(tmp_path / "none" / "last.pt")},
        tmp_path,
    )
    assert r.returncode == 2 and "FIREWATCH_RESUME checkpoint not found" in r.stderr


def test_train_job_bad_python(tmp_path):
    r = run_script(
        "slurm/train_gpu03.sh",
        [],
        {"FIREWATCH_PYTHON": posix(tmp_path / "nope" / "python")},
        tmp_path,
    )
    assert r.returncode == 2 and "Python interpreter not found" in r.stderr


def test_evaluate_job_requires_arguments(tmp_path):
    r = run_script("slurm/evaluate_gpu03.sh", [], {}, tmp_path)
    assert r.returncode == 2 and "--weights" in r.stderr


def test_submit_wrapper_usage_and_sbatch_check(tmp_path):
    r = run_script("slurm/submit.sh", ["bogus"], {}, tmp_path)
    assert r.returncode == 2 and "Usage" in r.stderr
    if shutil.which("sbatch") is None:
        r = run_script("slurm/submit.sh", ["train"], {}, tmp_path)
        assert r.returncode == 2 and "sbatch not found" in r.stderr
