#!/usr/bin/env python
"""Validate a D-Fire (YOLO-format) dataset on disk and report statistics.

Checks every configured split for image/label pairing, malformed label lines,
unknown class ids and non-normalised boxes. It does not modify the dataset and
does not train anything.

Examples:
    python scripts/prepare_data.py --config configs/baseline.yaml --data-root ~/datasets/dfire
    python scripts/prepare_data.py --config configs/baseline.yaml \
        --report outputs/dfire_report.json --write-data-yaml outputs/dfire_data.yaml

Exit codes: 0 = no errors, 1 = dataset errors found (or warnings with --strict),
2 = configuration / missing-directory problem.

Errors: malformed or unreadable label files, class ids outside class_names, and
splits with no images. Warnings: images without a label file, labels without an
image, duplicate image stems, and boxes outside [0, 1].
The JSON report (--report) is written whenever the scan ran (exit 0 or 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from firewatch.data.dfire import DEFAULT_IMAGE_EXTENSIONS, scan_split, ultralytics_data_yaml
from firewatch.utils.config import ConfigError, load_config, resolve_data_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/baseline.yaml", help="YAML config path")
    parser.add_argument("--data-root", help="dataset root (overrides config / env var)")
    parser.add_argument("--report", type=Path, help="write the full JSON report here")
    parser.add_argument(
        "--write-data-yaml",
        type=Path,
        help="write an Ultralytics/YOLOv5-style data YAML for the splits that were found",
    )
    parser.add_argument(
        "--strict", action="store_true", help="also fail on warnings (unpaired files, bad boxes)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
        root = resolve_data_root(cfg, args.data_root)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ds = cfg["dataset"]
    if not root.is_dir():
        print(f"error: dataset root does not exist: {root}", file=sys.stderr)
        return 2

    class_names = ds["class_names"]
    reports = []
    found_splits: dict[str, str] = {}
    missing_splits: list[str] = []
    for split, split_dir in ds["splits"].items():
        try:
            report = scan_split(
                root,
                split,
                split_dir,
                class_names,
                image_dir_name=ds.get("image_dir", "images"),
                label_dir_name=ds.get("label_dir", "labels"),
                image_extensions=tuple(ds.get("image_extensions", DEFAULT_IMAGE_EXTENSIONS)),
            )
        except FileNotFoundError as exc:
            print(f"warning: {exc} (split skipped)", file=sys.stderr)
            missing_splits.append(split)
            continue
        reports.append(report)
        found_splits[split] = split_dir

    if "train" not in found_splits:
        print("error: no usable 'train' split found", file=sys.stderr)
        return 2

    errors = sum(r.num_errors for r in reports)
    warnings = sum(r.num_warnings for r in reports)
    exit_code = 1 if errors or (args.strict and warnings) else 0

    for r in reports:
        print(
            f"[{r.split}] images={r.num_images} labels={r.num_label_files} "
            f"background={r.num_background_images} boxes={r.boxes_per_class} "
            f"images_with_class={r.images_per_class}"
        )
        print(
            f"[{r.split}] images_missing_label={len(r.images_missing_label)} "
            f"labels_missing_image={len(r.labels_missing_image)} "
            f"duplicate_image_stems={len(r.duplicate_image_stems)} "
            f"parse_errors={len(r.parse_errors)} unknown_class_boxes={r.unknown_class_boxes} "
            f"non_normalised_boxes={r.non_normalised_boxes}"
        )
        if r.num_images == 0:
            print(f"    error: split '{r.split}' contains no images")
        for msg in r.parse_errors[:5]:
            print(f"    {msg}")
    print(f"errors={errors} warnings={warnings} missing_splits={missing_splits}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "data_root": str(root),
            "class_names": class_names,
            "missing_splits": missing_splits,
            "strict": args.strict,
            "errors": errors,
            "warnings": warnings,
            "exit_code": exit_code,
            "splits": [r.to_dict() for r in reports],
        }
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"report written to {args.report}")

    if args.write_data_yaml:
        args.write_data_yaml.parent.mkdir(parents=True, exist_ok=True)
        data_yaml = ultralytics_data_yaml(
            root, found_splits, class_names, ds.get("image_dir", "images")
        )
        args.write_data_yaml.write_text(yaml.safe_dump(data_yaml, sort_keys=False), "utf-8")
        print(f"data yaml written to {args.write_data_yaml}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
