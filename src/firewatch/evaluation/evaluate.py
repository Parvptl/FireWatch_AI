"""Evaluate a trained checkpoint with Ultralytics' validator and save metrics as JSON."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
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
from firewatch.training.options import validate_device
from firewatch.training.preflight import PreflightError, check_data_yaml
from firewatch.training.runner import configure_ultralytics_env, load_yolo_class
from firewatch.utils.paths import ensure_outside_repo, expand_path

SPLITS = ("val", "test")
TEST_LOG_NAME = "test_set_evaluations.jsonl"
METRICS_NAME = "metrics.json"


@dataclass(frozen=True)
class EvalOptions:
    weights: Path
    data: Path
    split: str
    imgsz: int
    batch: int
    conf: float
    iou: float
    device: str
    workers: int
    output_dir: Path
    plots: bool
    final_test: bool
    allow_network: bool
    class_names: tuple[str, ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}


def run_dir_of(weights: Path) -> Path | None:
    """<run_dir> for checkpoints stored as <run_dir>/weights/<name>.pt, else None."""
    if weights.parent.name == "weights" and (weights.parent.parent / "args.yaml").is_file():
        return weights.parent.parent
    return None


def training_imgsz(weights: Path) -> int | None:
    run_dir = run_dir_of(weights)
    if run_dir is None:
        return None
    args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8")) or {}
    value = args.get("imgsz")
    return value if isinstance(value, int) else None


def resolve_eval_options(args, class_names: list[str]) -> EvalOptions:
    """Validate CLI arguments; raises ``ValueError``/``PreflightError`` with a clear message."""
    if args.split not in SPLITS:
        raise ValueError(f"--split must be one of {SPLITS}")
    if args.split == "test" and not args.final_test:
        raise ValueError(
            "--split test requires --final-test. The test set is reserved for the final, "
            "one-off evaluation of a model already selected on the val split."
        )
    weights = expand_path(args.weights)
    if not weights.is_file() or weights.suffix != ".pt":
        raise PreflightError(f"checkpoint not found or not a .pt file: {weights}")
    data = expand_path(args.data)
    imgsz = args.imgsz or training_imgsz(weights) or 640
    if imgsz < 32 or imgsz % 32:
        raise ValueError(f"--imgsz must be a positive multiple of 32, got {imgsz}")
    if args.batch < 1:
        raise ValueError("--batch must be >= 1")
    if not 0.0 <= args.conf < 1.0 or not 0.0 < args.iou <= 1.0:
        raise ValueError("--conf must be in [0, 1) and --iou in (0, 1]")

    if args.output_dir:
        output_dir = expand_path(args.output_dir)
    else:
        run_dir = run_dir_of(weights)
        if run_dir is None:
            raise ValueError("--output-dir is required when the checkpoint is not in a run dir")
        stamp = re.sub(r"[^0-9T]", "", utc_now())
        output_dir = run_dir / "eval" / f"{args.split}-{weights.stem}-{stamp}"
    ensure_outside_repo(output_dir, "evaluation output directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory already exists and is not empty: {output_dir}")

    return EvalOptions(
        weights=weights,
        data=data,
        split=args.split,
        imgsz=imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=validate_device(args.device),
        workers=args.workers,
        output_dir=output_dir,
        plots=args.plots,
        final_test=args.final_test,
        allow_network=args.allow_network,
        class_names=tuple(class_names),
    )


def metrics_to_dict(metrics) -> dict:
    """Extract overall and per-class detection metrics from Ultralytics ``DetMetrics``."""
    box = metrics.box
    per_class = metrics.summary()  # one row per class present in the evaluated split
    return to_jsonable(
        {
            "overall": {
                "precision": box.mp,
                "recall": box.mr,
                "mAP50": box.map50,
                "mAP75": box.map75,
                "mAP50-95": box.map,
                "fitness": metrics.fitness,
            },
            "per_class": per_class,
            "results_dict": metrics.results_dict,
            "speed_ms_per_image": metrics.speed,
        }
    )


def run_evaluation(opts: EvalOptions, require_view: bool = True) -> dict:
    check_data_yaml(
        opts.data, opts.class_names, require_view=require_view, required_splits=(opts.split,)
    )
    configure_ultralytics_env(opts.allow_network)
    yolo_cls = load_yolo_class()
    model = yolo_cls(str(opts.weights))
    names = [model.names[i] for i in sorted(model.names)]
    if names != list(opts.class_names):
        raise PreflightError(
            f"checkpoint classes {names} do not match the config {list(opts.class_names)}"
        )

    opts.output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    metrics = model.val(
        data=str(opts.data),
        split=opts.split,
        imgsz=opts.imgsz,
        batch=opts.batch,
        conf=opts.conf,
        iou=opts.iou,
        device=opts.device,
        workers=opts.workers,
        project=str(opts.output_dir.parent),
        name=opts.output_dir.name,
        exist_ok=True,
        plots=opts.plots,
        save_json=False,
        verbose=True,
    )
    result = {
        "split": opts.split,
        "final_test": opts.final_test,
        "started_utc": started,
        "finished_utc": utc_now(),
        "checkpoint": str(opts.weights),
        "checkpoint_sha256": sha256_file(opts.weights),
        "options": opts.to_dict(),
        "metrics": metrics_to_dict(metrics),
        "git": {k: v for k, v in git_state().items() if k != "diff"},
        "environment": collect_metadata(),
    }
    write_json(opts.output_dir / METRICS_NAME, result)

    run_dir = run_dir_of(opts.weights)
    if opts.split == "test" and run_dir is not None:
        # Append-only audit trail of every look at the test set for this run.
        entry = {
            "time_utc": result["finished_utc"],
            "checkpoint": str(opts.weights),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "output": str(opts.output_dir),
            "mAP50-95": result["metrics"]["overall"]["mAP50-95"],
        }
        with (run_dir / TEST_LOG_NAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    return result
