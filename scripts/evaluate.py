#!/usr/bin/env python
"""Evaluate a trained checkpoint and write overall and per-class detection metrics.

Reports precision, recall, mAP50, mAP75 and mAP50-95 overall and per class
(smoke, fire) from Ultralytics' validator, plus inference speed. Results go to
<output-dir>/metrics.json (default: <run_dir>/eval/<split>-<ckpt>-<timestamp>/).

The default split is val. Evaluating on test requires --final-test and is
appended to <run_dir>/test_set_evaluations.jsonl, so every look at the test
set leaves a record. Use test only once a model has been chosen on val.

Examples:
    python scripts/evaluate.py --weights ~/firewatch_runs/baseline-001/weights/best.pt \
        --data ~/datasets/dfire_view/data.yaml
    python scripts/evaluate.py --weights ~/firewatch_runs/baseline-001/weights/best.pt \
        --data ~/datasets/dfire_view/data.yaml --split test --final-test

Exit codes: 0 = success, 2 = invalid options or missing prerequisites, 1 = evaluation failed.
"""

from __future__ import annotations

import argparse
import json
import sys

from firewatch.training.preflight import PreflightError
from firewatch.utils.config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/baseline.yaml", help="config (for class names)")
    p.add_argument("--weights", required=True, help="checkpoint to evaluate (e.g. best.pt)")
    p.add_argument("--data", required=True, help="dataset YAML of the prepared view")
    p.add_argument("--split", default="val", choices=("val", "test"))
    p.add_argument("--final-test", action="store_true", help="confirm a final test-set evaluation")
    p.add_argument("--imgsz", type=int, help="default: the run's training imgsz, else 640")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--conf", type=float, default=0.001, help="confidence threshold for mAP")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--device", default="0", help="single GPU index or cpu")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output-dir", help="results directory (outside the repository)")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-network", action="store_true")
    p.add_argument("--allow-unprepared-data", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="validate and print the plan only")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from firewatch.evaluation.evaluate import resolve_eval_options
    from firewatch.training.preflight import check_data_yaml

    try:
        cfg = load_config(args.config)
        opts = resolve_eval_options(args, cfg["dataset"]["class_names"])
        check_data_yaml(
            opts.data,
            opts.class_names,
            require_view=not args.allow_unprepared_data,
            required_splits=(opts.split,),
        )
    except (ConfigError, PreflightError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps(opts.to_dict(), indent=2))
        print("dry run OK: prerequisites satisfied; nothing was written", file=sys.stderr)
        return 0

    from firewatch.evaluation.evaluate import run_evaluation
    from firewatch.training.runner import PrerequisiteError

    try:
        result = run_evaluation(opts, require_view=not args.allow_unprepared_data)
    except (PrerequisiteError, PreflightError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: evaluation failed: {exc!r}", file=sys.stderr)
        return 1
    overall = result["metrics"]["overall"]
    print(json.dumps({"split": opts.split, **overall}, indent=2))
    print(f"metrics written to {opts.output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
