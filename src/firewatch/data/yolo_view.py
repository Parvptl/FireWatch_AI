"""Build a sanitised, Ultralytics-ready *view* of a YOLO dataset.

The source dataset is never modified. The view is a new directory::

    <out>/data.yaml                 absolute-path data config for this machine
    <out>/firewatch_view.json       manifest: source, policy, every change made
    <out>/<split>/images/<file>     link (or copy) to the source image
    <out>/<split>/labels/<stem>.txt sanitised label

Training on a view rather than the raw dataset matters for Ultralytics 8.4:
it writes ``labels.cache`` next to the label directory (which would modify the
source tree), and it silently discards an entire image as "corrupt" when any
coordinate exceeds 1.01, but keeps zero-size boxes. Sanitising explicitly makes
both behaviours visible and deterministic.

Sanitisation policy (per box, in normalised coordinates):

1. Convert (xc, yc, w, h) to corners and clip them to [0, 1].
2. If the clipped width or height is <= ``min_box_size``, drop the box
   ("dropped_degenerate"). With the default ``min_box_size=0`` only boxes with
   zero or negative extent are dropped.
3. If clipping changed the box, write the clipped box ("clipped").
4. Unchanged boxes are copied verbatim (original text, no float reformatting).
   Label files are always written with LF line endings.

Per image:

- label file empty in the source -> kept as a background image;
- every box dropped -> image excluded ("excluded_all_boxes_dropped"), because
  keeping it as a background image would teach the model that a scene that
  does contain fire/smoke is negative;
- no label file -> excluded ("excluded_missing_label");
- a second image with the same stem -> excluded ("excluded_duplicate_stem");
- malformed label lines or class ids outside ``class_names`` -> the whole
  preparation fails (``ViewError``); these are errors, not quirks to repair.

Images are linked, not copied, by default. Ultralytics resolves the split
*directory* path (``Path.resolve()``), so a directory symlink would lead it back
to the source labels; per-file links inside a real directory do not.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from firewatch import __version__
from firewatch.data.dfire import DEFAULT_IMAGE_EXTENSIONS, find_image_label_pairs
from firewatch.data.yolo import LabelParseError, YoloBox, parse_label_line
from firewatch.utils.paths import is_within

MANIFEST_NAME = "firewatch_view.json"
DATA_YAML_NAME = "data.yaml"
LINK_MODES = ("symlink", "hardlink", "copy")
VIEW_FORMAT_VERSION = 1


class ViewError(RuntimeError):
    """Raised when a dataset view cannot be built safely."""


@dataclass(frozen=True)
class SanitizedBox:
    box: YoloBox | None  # None when dropped
    action: str | None  # None (unchanged), "clipped" or "dropped_degenerate"


def sanitize_box(box: YoloBox, min_box_size: float = 0.0) -> SanitizedBox:
    """Apply the clip / drop policy described in the module docstring to one box."""
    x1 = box.x_center - box.width / 2
    y1 = box.y_center - box.height / 2
    x2 = box.x_center + box.width / 2
    y2 = box.y_center + box.height / 2
    cx1, cy1, cx2, cy2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
    w, h = cx2 - cx1, cy2 - cy1
    if w <= min_box_size or h <= min_box_size:
        return SanitizedBox(None, "dropped_degenerate")
    if (cx1, cy1, cx2, cy2) == (x1, y1, x2, y2):
        return SanitizedBox(box, None)
    clipped = YoloBox(box.class_id, (cx1 + cx2) / 2, (cy1 + cy2) / 2, w, h)
    return SanitizedBox(clipped, "clipped")


def format_box(box: YoloBox) -> str:
    return f"{box.class_id} {box.x_center:.8f} {box.y_center:.8f} {box.width:.8f} {box.height:.8f}"


def sanitize_label_text(
    text: str, num_classes: int, min_box_size: float = 0.0, source: str = "<label>"
) -> tuple[list[str], list[dict], int]:
    """Sanitise the content of one label file.

    Returns (output_lines, changes, num_input_boxes). ``changes`` holds one
    record per clipped or dropped box. Raises ``ViewError`` on malformed lines
    or unknown class ids.
    """
    out: list[str] = []
    changes: list[dict] = []
    n_in = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        n_in += 1
        try:
            box = parse_label_line(line)
        except LabelParseError as exc:
            raise ViewError(f"{source}:{lineno}: {exc}") from exc
        if box.class_id >= num_classes:
            raise ViewError(f"{source}:{lineno}: class id {box.class_id} >= {num_classes}")
        result = sanitize_box(box, min_box_size)
        if result.action is None:
            out.append(line)
            continue
        record = {"file": source, "line": lineno, "action": result.action, "before": line}
        if result.box is not None:
            record["after"] = format_box(result.box)
            out.append(record["after"])
        changes.append(record)
    return out, changes, n_in


def _link(src: Path, dst: Path, mode: str) -> None:
    try:
        if mode == "symlink":
            dst.symlink_to(src)
        elif mode == "hardlink":
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
    except OSError as exc:
        hint = ""
        if mode == "symlink":
            hint = " (on Windows symlinks need Developer Mode; try --link-mode hardlink)"
        elif mode == "hardlink":
            hint = " (hard links need source and view on the same filesystem; try symlink)"
        raise ViewError(f"cannot {mode} {src} -> {dst}: {exc}{hint}") from exc


def _prepare_output_dir(out_dir: Path, source_root: Path, overwrite: bool) -> None:
    if is_within(out_dir, source_root) or is_within(source_root, out_dir):
        raise ViewError(
            f"view directory {out_dir} must not overlap the source dataset {source_root}"
        )
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise ViewError(f"{out_dir} exists and is not a directory")
    if not any(out_dir.iterdir()):
        out_dir.rmdir()
        return
    if not (out_dir / MANIFEST_NAME).is_file():
        raise ViewError(f"{out_dir} is not empty and is not a FireWatch dataset view; refusing")
    if not overwrite:
        raise ViewError(f"{out_dir} already contains a dataset view; pass --overwrite to rebuild")
    # Only links, copies and generated labels live here; rmtree removes links,
    # never the files they point to.
    shutil.rmtree(out_dir)


def build_yolo_view(
    source_root: str | Path,
    out_dir: str | Path,
    splits: dict[str, str],
    class_names: list[str],
    image_dir_name: str = "images",
    label_dir_name: str = "labels",
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
    link_mode: str = "symlink",
    min_box_size: float = 0.0,
    overwrite: bool = False,
) -> dict:
    """Build the view and return its manifest. See the module docstring for the policy."""
    if link_mode not in LINK_MODES:
        raise ViewError(f"link_mode must be one of {LINK_MODES}, got {link_mode!r}")
    if not 0.0 <= min_box_size < 1.0:
        raise ViewError(f"min_box_size must be in [0, 1), got {min_box_size}")
    source_root = Path(source_root).resolve()
    out_dir = Path(out_dir).absolute()
    if not source_root.is_dir():
        raise ViewError(f"source dataset not found: {source_root}")
    if "train" not in splits or "val" not in splits:
        raise ViewError("a view needs at least 'train' and 'val' splits")
    _prepare_output_dir(out_dir, source_root, overwrite)

    # Build next to the destination, then rename, so a failure never leaves a
    # half-built directory that looks valid.
    staging = out_dir.with_name(f".{out_dir.name}.partial-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        split_reports = {}
        for split, split_dir in splits.items():
            split_reports[split] = _build_split(
                source_root / split_dir,
                staging / split,
                class_names,
                image_dir_name,
                label_dir_name,
                image_extensions,
                link_mode,
                min_box_size,
            )
        data_yaml = {
            "path": str(out_dir),
            **{split: f"{split}/images" for split in splits},
            "nc": len(class_names),
            "names": dict(enumerate(class_names)),
        }
        manifest = {
            "format_version": VIEW_FORMAT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "firewatch_version": __version__,
            "source_root": str(source_root),
            "view_root": str(out_dir),
            "class_names": list(class_names),
            "link_mode": link_mode,
            "min_box_size": min_box_size,
            "policy": {
                "out_of_bounds_box": "clip corners to [0, 1]",
                "degenerate_box": f"drop if clipped width or height <= {min_box_size}",
                "all_boxes_dropped": "exclude image",
                "empty_label_file": "keep as background image",
                "missing_label_file": "exclude image",
                "duplicate_image_stem": "exclude all but the first (sorted) image",
                "malformed_line_or_unknown_class": "abort",
            },
            "splits": split_reports,
        }
        (staging / DATA_YAML_NAME).write_text(
            yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
        )
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.rename(out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _build_split(
    src_split: Path,
    dst_split: Path,
    class_names: list[str],
    image_dir_name: str,
    label_dir_name: str,
    image_extensions: tuple[str, ...],
    link_mode: str,
    min_box_size: float,
) -> dict:
    src_images, src_labels = src_split / image_dir_name, src_split / label_dir_name
    for d in (src_images, src_labels):
        if not d.is_dir():
            raise ViewError(f"source directory not found: {d}")
    images, labels, duplicates = find_image_label_pairs(src_images, src_labels, image_extensions)
    (dst_split / "images").mkdir(parents=True)
    (dst_split / "labels").mkdir(parents=True)

    excluded: dict[str, list[str]] = defaultdict(list)
    excluded["excluded_duplicate_stem"] = list(duplicates)
    changes: list[dict] = []
    counts = {"images_in": len(images), "images_out": 0, "background": 0}
    boxes_in = boxes_out = 0
    for stem in sorted(images):
        if stem not in labels:
            excluded["excluded_missing_label"].append(stem)
            continue
        text = labels[stem].read_text(encoding="utf-8-sig")
        lines, file_changes, n_in = sanitize_label_text(
            text, len(class_names), min_box_size, source=str(labels[stem])
        )
        changes.extend(file_changes)
        if n_in and not lines:
            excluded["excluded_all_boxes_dropped"].append(stem)
            continue
        boxes_in += n_in
        boxes_out += len(lines)
        if not lines:
            counts["background"] += 1
        _link(images[stem].resolve(), dst_split / "images" / images[stem].name, link_mode)
        (dst_split / "labels" / f"{stem}.txt").write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8"
        )
        counts["images_out"] += 1

    return {
        "source": str(src_split),
        **counts,
        "boxes_in_kept_images": boxes_in,
        "boxes_out": boxes_out,
        "num_clipped": sum(c["action"] == "clipped" for c in changes),
        "num_dropped_degenerate": sum(c["action"] == "dropped_degenerate" for c in changes),
        "excluded": {k: v for k, v in excluded.items() if v},
        "changes": changes,
    }


def load_view_manifest(view_root: str | Path) -> dict:
    """Read the manifest of a built view; raises ``ViewError`` if it is missing/invalid."""
    path = Path(view_root) / MANIFEST_NAME
    if not path.is_file():
        raise ViewError(f"no {MANIFEST_NAME} in {view_root}: not a FireWatch dataset view")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ViewError(f"{path}: invalid JSON: {exc}") from exc
    if manifest.get("format_version") != VIEW_FORMAT_VERSION:
        raise ViewError(f"{path}: unsupported view format {manifest.get('format_version')!r}")
    return manifest
