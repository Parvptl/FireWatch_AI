"""YOLO-format bounding-box labels.

Each label file holds one object per line::

    <class_id> <x_center> <y_center> <width> <height>

with coordinates normalised to [0, 1] relative to the image size. An empty
label file means the image contains no objects (a background image).

The coordinate conversions are adapted from ``utils/utils.py`` in the upstream
D-Fire repository (https://github.com/gaia-solutions-on-demand/DFireDataset,
CC0-1.0), with the same clamping of negative values that upstream applies to
work around labelling-tool bugs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


class LabelParseError(ValueError):
    """Raised when a YOLO label line is malformed."""


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def is_normalised(self) -> bool:
        """True if all coordinates lie in [0, 1] and the box has positive size."""
        coords = (self.x_center, self.y_center, self.width, self.height)
        return all(0.0 <= c <= 1.0 for c in coords) and self.width > 0 and self.height > 0


def parse_label_line(line: str) -> YoloBox:
    """Parse a single non-empty YOLO label line."""
    parts = line.split()
    if len(parts) != 5:
        raise LabelParseError(f"expected 5 fields, got {len(parts)}: {line!r}")
    try:
        class_value = float(parts[0])
        coords = [float(p) for p in parts[1:]]
    except ValueError as exc:
        raise LabelParseError(f"non-numeric field in {line!r}") from exc
    if not all(math.isfinite(c) for c in coords):
        raise LabelParseError(f"non-finite coordinate in {line!r}")
    if not class_value.is_integer() or class_value < 0:
        raise LabelParseError(f"class id must be a non-negative integer: {parts[0]!r}")
    return YoloBox(int(class_value), *coords)


def read_label_file(path: str | Path) -> list[YoloBox]:
    """Read every box in a label file. Blank lines are ignored.

    Raises ``LabelParseError`` (with file and line number) on the first bad line,
    or if the file is not valid UTF-8 text. A leading UTF-8 BOM is tolerated.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LabelParseError(f"{path}: not valid UTF-8 text ({exc.reason})") from exc
    boxes: list[YoloBox] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            boxes.append(parse_label_line(line))
        except LabelParseError as exc:
            raise LabelParseError(f"{path}:{lineno}: {exc}") from exc
    return boxes


def _non_negative(value: float) -> float:
    return value if value > 0 else 0


def yolo_to_pixel(image_size: tuple[int, int], box: tuple[float, float, float, float]) -> list[int]:
    """Convert normalised (xc, yc, w, h) to pixel (xmin, ymin, xmax, ymax).

    ``image_size`` is (width, height).
    """
    width, height = image_size
    xc, yc, w, h = box
    xmin = _non_negative(round(width * (xc - w / 2)))
    xmax = _non_negative(round(width * (xc + w / 2)))
    ymin = _non_negative(round(height * (yc - h / 2)))
    ymax = _non_negative(round(height * (yc + h / 2)))
    return [xmin, ymin, xmax, ymax]


def pixel_to_yolo(
    image_size: tuple[int, int], box: tuple[float, float, float, float]
) -> list[float]:
    """Convert pixel (xmin, ymin, xmax, ymax) to normalised (xc, yc, w, h).

    ``image_size`` is (width, height).
    """
    width, height = image_size
    xmin, ymin, xmax, ymax = box
    return [
        _non_negative((xmin + xmax) / 2 / width),
        _non_negative((ymin + ymax) / 2 / height),
        _non_negative((xmax - xmin) / width),
        _non_negative((ymax - ymin) / height),
    ]
