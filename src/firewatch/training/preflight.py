"""Checks that run before Ultralytics is imported, so a job fails fast and clearly."""

from __future__ import annotations

from pathlib import Path

import yaml

from firewatch.data.yolo_view import MANIFEST_NAME, ViewError, load_view_manifest
from firewatch.training.options import TrainOptions
from firewatch.utils.paths import ensure_outside_repo

WEIGHTS_DOC = "docs/cluster-setup.md#model-weights"
RUN_METADATA_NAME = "firewatch_run.json"


class PreflightError(RuntimeError):
    """A prerequisite for training or evaluation is missing or invalid."""


def _names_list(names, path: Path) -> list[str]:
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        try:
            keys = sorted(int(k) for k in names)
        except (TypeError, ValueError) as exc:
            raise PreflightError(f"{path}: 'names' keys must be class ids") from exc
        if keys != list(range(len(keys))):
            raise PreflightError(f"{path}: 'names' ids must be 0..{len(keys) - 1}, got {keys}")
        return [str(names[k] if k in names else names[str(k)]) for k in keys]
    raise PreflightError(f"{path}: 'names' must be a list or an id -> name mapping")


def check_data_yaml(
    path: str | Path,
    class_names: list[str] | tuple[str, ...],
    require_view: bool = True,
    required_splits: tuple[str, ...] = ("train", "val"),
) -> dict:
    """Validate an Ultralytics data YAML for use on *this* machine.

    Checks: absolute existing ``path``; required split image dirs exist and are
    non-empty with readable (non-dangling) images; class order equals
    ``class_names``; ``test`` is distinct from train/val; no ``download`` key
    (Ultralytics would execute it); and, by default, that it belongs to a
    FireWatch dataset view built by ``scripts/prepare_data.py --output-dir``.
    """
    path = Path(path)
    if not path.is_file():
        raise PreflightError(f"dataset YAML not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PreflightError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise PreflightError(f"{path}: top level must be a mapping")
    if "download" in data:
        raise PreflightError(
            f"{path}: 'download' key is not allowed (Ultralytics would run it to fetch data)"
        )

    root = data.get("path")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise PreflightError(
            f"{path}: 'path' must be an absolute directory on this machine, got {root!r}. "
            "Regenerate the YAML on this machine with scripts/prepare_data.py --output-dir."
        )
    root_path = Path(root)
    if not root_path.is_dir():
        raise PreflightError(f"{path}: dataset root does not exist on this machine: {root}")

    names = _names_list(data.get("names"), path)
    if names != list(class_names):
        raise PreflightError(
            f"{path}: class order {names} does not match the config {list(class_names)}"
        )
    if "nc" in data and data["nc"] != len(names):
        raise PreflightError(f"{path}: nc={data['nc']} but {len(names)} names")

    split_dirs: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        value = data.get(split)
        if value is None:
            if split in required_splits:
                raise PreflightError(f"{path}: missing required split '{split}'")
            continue
        if not isinstance(value, str):
            raise PreflightError(f"{path}: '{split}' must be a single directory path")
        d = (root_path / value).resolve()
        if not d.is_dir():
            raise PreflightError(f"{path}: '{split}' images directory not found: {d}")
        first = next((p for p in sorted(d.iterdir()) if p.is_file() or p.is_symlink()), None)
        if first is None:
            raise PreflightError(f"{path}: '{split}' images directory is empty: {d}")
        if not first.exists():
            raise PreflightError(
                f"{path}: '{split}' image {first} is a dangling link; the source dataset "
                "has moved. Rebuild the view with scripts/prepare_data.py --overwrite."
            )
        split_dirs[split] = d
    if "test" in split_dirs and split_dirs["test"] in (split_dirs["train"], split_dirs["val"]):
        raise PreflightError(f"{path}: 'test' must not be the same directory as train/val")
    if split_dirs["train"] == split_dirs["val"]:
        raise PreflightError(f"{path}: 'train' and 'val' must be different directories")

    if require_view:
        try:
            manifest = load_view_manifest(root_path)
        except ViewError as exc:
            raise PreflightError(
                f"{exc}. Train on a sanitised view (scripts/prepare_data.py --output-dir) or "
                "pass --allow-unprepared-data to use a raw dataset (Ultralytics then writes "
                "labels.cache into it and drops images with out-of-bounds boxes)."
            ) from exc
        if manifest.get("class_names") != list(class_names):
            raise PreflightError(
                f"{root_path / MANIFEST_NAME}: built for classes {manifest.get('class_names')}"
            )
    return data


def check_weights(path: Path | None, url_hint: str | None = None) -> Path:
    """The pretrained weights must already be a local file; nothing is downloaded."""
    if path is None:
        raise PreflightError("no weights configured")
    if not path.is_file():
        source = f" from {url_hint}" if url_hint else ""
        raise PreflightError(
            f"weights file not found: {path}. Download it{source} on a machine with internet "
            f"access and copy it to that path (see {WEIGHTS_DOC}). Training never downloads "
            "weights because compute nodes are offline."
        )
    if path.suffix != ".pt":
        raise PreflightError(f"weights must be a PyTorch .pt checkpoint, got {path}")
    return path


def check_run_dir(opts: TrainOptions) -> None:
    """Output location rules for fresh and resumed runs."""
    try:
        ensure_outside_repo(opts.run_dir, "training output directory")
    except ValueError as exc:
        raise PreflightError(str(exc)) from exc
    if opts.resume is not None:
        if not opts.resume.is_file():
            raise PreflightError(f"resume checkpoint not found: {opts.resume}")
        if opts.resume.parent.name != "weights":
            raise PreflightError(f"--resume expects <run_dir>/weights/last.pt, got {opts.resume}")
        if not (opts.run_dir / "args.yaml").is_file():
            raise PreflightError(f"{opts.run_dir} has no args.yaml; not an Ultralytics run dir")
        return
    for marker in (RUN_METADATA_NAME, "weights", "args.yaml", "results.csv"):
        if (opts.run_dir / marker).exists():
            raise PreflightError(
                f"run directory already holds a training run ({opts.run_dir / marker}). "
                "Choose a new --name, or continue it with --resume <run_dir>/weights/last.pt."
            )


def run_preflight(opts: TrainOptions, weights_url: str | None, require_view: bool) -> dict:
    """All pre-import checks for ``scripts/train.py``. Returns the parsed data YAML (or {})."""
    check_run_dir(opts)
    if opts.resume is not None:
        return {}
    check_weights(opts.weights, weights_url)
    return check_data_yaml(opts.data, opts.class_names, require_view=require_view)
