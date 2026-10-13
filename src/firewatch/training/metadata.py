"""Experiment provenance: Git state, software versions, hardware, hashes."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from firewatch import __version__
from firewatch.utils.paths import project_repo_root

PACKAGES = (
    "ultralytics",
    "torch",
    "torchvision",
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "PyYAML",
)
SLURM_VARS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
    "CUDA_VISIBLE_DEVICES",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def git_state(repo: Path | None = None) -> dict:
    """Commit, branch and dirty flag of the FireWatch checkout (None values if unavailable)."""
    repo = repo or project_repo_root()
    if repo is None:
        return {"commit": None, "branch": None, "dirty": None, "diff": None}
    commit = _git(["rev-parse", "HEAD"], repo)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo)
    status = _git(["status", "--porcelain"], repo)
    dirty = None if status is None else bool(status.strip())
    diff = _git(["diff", "HEAD"], repo) if dirty else None
    return {
        "commit": commit.strip() if commit else None,
        "branch": branch.strip() if branch else None,
        "dirty": dirty,
        "diff": diff,
    }


def package_versions(packages: tuple[str, ...] = PACKAGES) -> dict:
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_runtime() -> dict:
    """CUDA details from torch; safe to call when torch is missing."""
    try:
        import torch
    except ImportError:
        return {"available": False, "error": "torch not importable"}
    info = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def collect_metadata(include_torch: bool = True) -> dict:
    return {
        "firewatch_version": __version__,
        "timestamp_utc": utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "argv": sys.argv,
        "packages": package_versions(),
        "torch_runtime": torch_runtime() if include_torch else None,
        "slurm": {k: os.environ[k] for k in SLURM_VARS if k in os.environ},
        "env": {
            k: os.environ.get(k) for k in ("YOLO_OFFLINE", "YOLO_AUTOINSTALL", "PYTHONHASHSEED")
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def to_jsonable(value):
    """Convert numpy scalars/arrays and Paths in nested metrics to plain JSON types."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):  # numpy arrays / scalars, torch tensors
        return to_jsonable(value.tolist())
    if isinstance(value, float) and value != value:  # NaN -> null
        return None
    return value
