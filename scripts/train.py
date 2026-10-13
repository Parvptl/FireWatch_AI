#!/usr/bin/env python
"""Train a YOLO fire/smoke detector with Ultralytics (single GPU).

Settings come from the config's ``train:`` section; any command-line option
overrides it. Pretrained weights must already exist locally: nothing is
downloaded (compute nodes are offline). Model selection uses the val split
only; the test split is never touched here (see scripts/evaluate.py).

Examples:
    # check everything without importing Ultralytics or training
    python scripts/train.py --config configs/smoke_test.yaml \
        --data ~/datasets/dfire_view/data.yaml --dry-run

    python scripts/train.py --config configs/baseline.yaml \
        --data ~/datasets/dfire_view/data.yaml --name baseline-001

    # continue an interrupted run (settings come from the checkpoint)
    python scripts/train.py --config configs/baseline.yaml \
        --resume ~/firewatch_runs/baseline-001/weights/last.pt

Exit codes: 0 = success (or dry run passed), 2 = invalid options or missing
prerequisites, 1 = training failed.
"""

from __future__ import annotations

import argparse
import json
import sys

from firewatch.training.options import OptionsError, resolve_train_options
from firewatch.training.preflight import PreflightError, run_preflight
from firewatch.utils.config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Unset options fall back to the config's train: section, then built-in defaults.",
    )
    p.add_argument("--config", default="configs/baseline.yaml", help="experiment YAML")
    p.add_argument("--data", help="dataset YAML of a prepared view (env FIREWATCH_DATA_YAML)")
    p.add_argument("--weights", help="local pretrained .pt file (never downloaded)")
    p.add_argument("--imgsz", type=int, help="training image size (multiple of 32)")
    p.add_argument("--epochs", type=int, help="maximum number of epochs")
    p.add_argument("--patience", type=int, help="early-stop after N epochs without val gain")
    p.add_argument("--batch", type=int, help="batch size (fixed; AutoBatch is not used)")
    p.add_argument("--seed", type=int, help="random seed")
    p.add_argument("--device", help="single GPU index such as 0, or cpu")
    p.add_argument("--workers", type=int, help="dataloader worker processes")
    p.add_argument("--fraction", type=float, help="fraction of the train split to use (0, 1]")
    p.add_argument("--optimizer", help="Ultralytics optimizer name (default: auto)")
    p.add_argument("--project", help="parent directory for runs (env FIREWATCH_RUNS_ROOT)")
    p.add_argument("--name", help="run name; the run directory is <project>/<name>")
    for flag, text in (
        ("amp", "mixed precision"),
        ("deterministic", "deterministic algorithms where available"),
        ("plots", "training plots"),
    ):
        p.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction, default=None, help=text)
    p.add_argument("--resume", metavar="LAST_PT", help="resume <run_dir>/weights/last.pt")
    p.add_argument(
        "--allow-network",
        action="store_true",
        help="do not force Ultralytics offline mode (YOLO_OFFLINE)",
    )
    p.add_argument(
        "--allow-unprepared-data",
        action="store_true",
        help="accept a data YAML that is not a prepared FireWatch view (not recommended)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="validate options and prerequisites, print the plan, and exit without training",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        opts = resolve_train_options(cfg, args, args.config)
        run_preflight(
            opts,
            weights_url=(cfg.get("train") or {}).get("weights_url"),
            require_view=not args.allow_unprepared_data,
        )
    except (ConfigError, OptionsError, PreflightError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        from firewatch.training.options import ultralytics_train_kwargs

        plan = {
            "run_dir": str(opts.run_dir),
            "options": opts.to_dict(),
            "ultralytics_train_kwargs": ultralytics_train_kwargs(opts),
        }
        print(json.dumps(plan, indent=2))
        print("dry run OK: prerequisites satisfied; nothing was written", file=sys.stderr)
        return 0

    from firewatch.training.runner import PrerequisiteError, run_training

    try:
        record = run_training(opts, cfg)
    except PrerequisiteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # recorded in firewatch_run.json by run_training
        print(f"error: training failed: {exc!r}", file=sys.stderr)
        return 1
    print(f"training {record['status']}: best={record.get('best_checkpoint')}")
    print(f"run directory: {opts.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
