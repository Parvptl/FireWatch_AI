"""Ultralytics integration for training. Ultralytics (and torch) are imported lazily."""

from __future__ import annotations

import os
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

import yaml

from firewatch.training.metadata import (
    collect_metadata,
    git_state,
    sha256_file,
    to_jsonable,
    utc_now,
    write_json,
)
from firewatch.training.options import TrainOptions, ultralytics_train_kwargs
from firewatch.training.preflight import RUN_METADATA_NAME

# Version this integration was written and checked against (pyproject pins it).
ULTRALYTICS_VERSION = "8.4.176"
# Ultralytics 8.4's AMP self-check loads this file from its settings weights_dir
# and tries to download it if absent (it does not use the training weights).
AMP_CHECK_WEIGHTS = "yolo26n.pt"

REPRODUCIBILITY_NOTE = (
    "Seeds python/numpy/torch with seed+1 (Ultralytics adds 1 + RANK) and, with "
    "deterministic=true, calls torch.use_deterministic_algorithms(True, warn_only=True). "
    "Some CUDA kernels have no deterministic implementation, so repeated runs on the "
    "same GPU can still differ slightly; runs on different GPU models or library "
    "versions are not expected to match exactly."
)


class PrerequisiteError(RuntimeError):
    """Ultralytics or another runtime prerequisite is missing."""


def configure_ultralytics_env(allow_network: bool) -> None:
    """Must run before ``import ultralytics`` (it reads these at import time)."""
    if "ultralytics" in sys.modules:
        raise RuntimeError(
            "configure_ultralytics_env() must be called before importing ultralytics"
        )
    os.environ["YOLO_AUTOINSTALL"] = "false"  # never pip-install packages at runtime
    if not allow_network:
        os.environ["YOLO_OFFLINE"] = "1"  # download attempts fail fast instead of hanging


def load_yolo_class():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise PrerequisiteError(
            "Ultralytics is not installed in this environment. Install the training extra "
            "without touching torch: pip install -e '.[train]' -c constraints/cluster-torch.txt "
            "(see docs/cluster-setup.md)."
        ) from exc
    return YOLO


def ultralytics_version() -> str | None:
    try:
        return importlib_metadata.version("ultralytics")
    except importlib_metadata.PackageNotFoundError:
        return None


def amp_check_status() -> dict:
    """Where Ultralytics' AMP self-check will look for its weights (call after import)."""
    try:
        from ultralytics.utils import WEIGHTS_DIR
    except ImportError:
        return {"weights": None, "present": False}
    path = Path(WEIGHTS_DIR) / AMP_CHECK_WEIGHTS
    return {
        "weights": str(path),
        "present": path.is_file(),
        "note": "if absent and offline, Ultralytics skips the AMP check and keeps amp=True",
    }


def run_training(opts: TrainOptions, config: dict) -> dict:
    """Train (or resume) and return the final run record, also written to the run dir."""
    run_dir = opts.run_dir
    fresh = opts.resume is None
    stamp = utc_now().replace(":", "").replace("-", "")
    record_path = run_dir / (RUN_METADATA_NAME if fresh else f"firewatch_resume_{stamp}.json")

    configure_ultralytics_env(opts.allow_network)
    yolo_cls = load_yolo_class()
    installed = ultralytics_version()

    kwargs = ultralytics_train_kwargs(opts)
    git = git_state()
    start_weights = opts.weights if fresh else opts.resume
    record = {
        "status": "running",
        "mode": "fresh" if fresh else "resume",
        "started_utc": utc_now(),
        "options": opts.to_dict(),
        "ultralytics_train_kwargs": kwargs,
        "ultralytics_version_expected": ULTRALYTICS_VERSION,
        "start_weights": str(start_weights),
        "start_weights_sha256": sha256_file(start_weights),
        "git": {k: v for k, v in git.items() if k != "diff"},
        "environment": collect_metadata(),
        "amp_check": amp_check_status(),
        "reproducibility": {
            "seed": opts.seed,
            "deterministic": opts.deterministic,
            "note": REPRODUCIBILITY_NOTE,
        },
    }
    if installed != ULTRALYTICS_VERSION:
        record["warnings"] = [
            f"ultralytics {installed} installed; this pipeline was verified with "
            f"{ULTRALYTICS_VERSION}"
        ]
        print(f"WARNING: {record['warnings'][0]}", file=sys.stderr)

    run_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        (run_dir / "firewatch_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    if git.get("diff"):
        (run_dir / f"uncommitted_{stamp}.diff").write_text(git["diff"], encoding="utf-8")
    write_json(record_path, record)

    try:
        model = yolo_cls(str(start_weights))
        model.train(**kwargs)
    except BaseException as exc:  # record interruptions (e.g. SLURM time limit) too
        record.update(status="failed", finished_utc=utc_now(), error=repr(exc))
        write_json(record_path, record)
        raise

    trainer = model.trainer
    save_dir = Path(trainer.save_dir).resolve()
    best, last = Path(trainer.best), Path(trainer.last)
    record.update(
        status="completed",
        finished_utc=utc_now(),
        save_dir=str(save_dir),
        best_checkpoint=str(best) if best.is_file() else None,
        best_checkpoint_sha256=sha256_file(best) if best.is_file() else None,
        last_checkpoint=str(last) if last.is_file() else None,
        last_checkpoint_sha256=sha256_file(last) if last.is_file() else None,
        final_val_metrics=to_jsonable(getattr(trainer, "metrics", None)),
        model_selection="best.pt = epoch with the highest val fitness (mAP50-95 in 8.4)",
    )
    if save_dir != run_dir.resolve():
        record.setdefault("warnings", []).append(
            f"Ultralytics saved to {save_dir}, not the expected {run_dir}"
        )
    write_json(record_path, record)
    return record
