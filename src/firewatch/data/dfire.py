"""Validation and statistics for a D-Fire style YOLO dataset on disk.

Expected layout for each split (directory names are configurable)::

    <root>/<split>/images/<stem>.jpg
    <root>/<split>/labels/<stem>.txt

D-Fire class ids: 0 = smoke, 1 = fire (as in the dataset authors'
``data/dfire.yaml`` at https://github.com/pedbrgs/Fire-Detection).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from firewatch.data.yolo import LabelParseError, read_label_file

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Cap on how many example paths are kept per problem category in a report.
MAX_EXAMPLES = 20


@dataclass
class SplitReport:
    split: str
    image_dir: str
    label_dir: str
    num_images: int = 0
    num_label_files: int = 0
    num_background_images: int = 0
    images_missing_label: list[str] = field(default_factory=list)
    labels_missing_image: list[str] = field(default_factory=list)
    duplicate_image_stems: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    unknown_class_boxes: int = 0
    non_normalised_boxes: int = 0
    boxes_per_class: dict[str, int] = field(default_factory=dict)
    images_per_class: dict[str, int] = field(default_factory=dict)

    @property
    def num_errors(self) -> int:
        """Problems that make the split unusable for training as-is."""
        empty = 1 if self.num_images == 0 else 0
        return len(self.parse_errors) + self.unknown_class_boxes + empty

    @property
    def num_warnings(self) -> int:
        """Suspicious but non-fatal problems (reported; fatal only with --strict)."""
        return (
            len(self.images_missing_label)
            + len(self.labels_missing_image)
            + len(self.duplicate_image_stems)
            + self.non_normalised_boxes
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in (
            "images_missing_label",
            "labels_missing_image",
            "duplicate_image_stems",
            "parse_errors",
        ):
            data[f"num_{key}"] = len(data[key])
            data[key] = data[key][:MAX_EXAMPLES]
        data["num_errors"] = self.num_errors
        data["num_warnings"] = self.num_warnings
        return data


def find_image_label_pairs(
    image_dir: Path,
    label_dir: Path,
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
) -> tuple[dict[str, Path], dict[str, Path], list[str]]:
    """Return ({stem: image_path}, {stem: label_path}, duplicate_stems) for a split.

    Images sharing a stem (e.g. ``a.jpg`` and ``a.png``) would share one label
    file; such stems are returned in ``duplicate_stems`` and only the first path
    in sorted order is kept.
    """
    exts = {e.lower() for e in image_extensions}
    images: dict[str, Path] = {}
    duplicates: set[str] = set()
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            if p.stem in images:
                duplicates.add(p.stem)
            else:
                images[p.stem] = p
    labels = {p.stem: p for p in label_dir.iterdir() if p.is_file() and p.suffix == ".txt"}
    return images, labels, sorted(duplicates)


def scan_split(
    root: Path,
    split: str,
    split_dir: str,
    class_names: list[str],
    image_dir_name: str = "images",
    label_dir_name: str = "labels",
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
) -> SplitReport:
    """Scan one split and return counts plus any consistency problems.

    Raises ``FileNotFoundError`` if the split's image or label directory is absent.
    """
    image_dir = root / split_dir / image_dir_name
    label_dir = root / split_dir / label_dir_name
    for d in (image_dir, label_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"{split}: directory not found: {d}")

    report = SplitReport(split=split, image_dir=str(image_dir), label_dir=str(label_dir))
    images, labels, report.duplicate_image_stems = find_image_label_pairs(
        image_dir, label_dir, image_extensions
    )
    report.num_images = len(images)
    report.num_label_files = len(labels)
    report.images_missing_label = sorted(str(images[s]) for s in images.keys() - labels.keys())
    report.labels_missing_image = sorted(str(labels[s]) for s in labels.keys() - images.keys())

    box_counts: Counter[int] = Counter()
    image_counts: Counter[int] = Counter()
    for stem in sorted(images.keys() & labels.keys()):
        try:
            boxes = read_label_file(labels[stem])
        except LabelParseError as exc:
            report.parse_errors.append(str(exc))
            continue
        except OSError as exc:
            report.parse_errors.append(f"{labels[stem]}: unreadable ({exc})")
            continue
        if not boxes:
            report.num_background_images += 1
            continue
        for box in boxes:
            if box.class_id >= len(class_names):
                report.unknown_class_boxes += 1
            if not box.is_normalised():
                report.non_normalised_boxes += 1
            box_counts[box.class_id] += 1
        image_counts.update({b.class_id for b in boxes})

    def named(counter: Counter[int]) -> dict[str, int]:
        return {
            class_names[c] if c < len(class_names) else f"unknown_{c}": n
            for c, n in sorted(counter.items())
        }

    report.boxes_per_class = named(box_counts)
    report.images_per_class = named(image_counts)
    return report


def ultralytics_data_yaml(
    root: Path,
    splits: dict[str, str],
    class_names: list[str],
    image_dir_name: str = "images",
) -> dict:
    """Build an Ultralytics/YOLOv5-style data config dict for the given splits."""
    data: dict = {"path": str(root.resolve())}
    for split, split_dir in splits.items():
        data[split] = f"{split_dir}/{image_dir_name}"
    data["nc"] = len(class_names)
    data["names"] = list(class_names)
    return data
