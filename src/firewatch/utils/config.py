"""YAML experiment configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

DATA_ROOT_ENV = "FIREWATCH_DATA_ROOT"


class ConfigError(ValueError):
    """Raised for missing or invalid configuration values."""


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and check the sections the code relies on."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        try:
            cfg = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    dataset = cfg.get("dataset")
    if not isinstance(dataset, dict):
        raise ConfigError(f"{path}: missing 'dataset' section")
    names = dataset.get("class_names")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ConfigError(f"{path}: dataset.class_names must be a non-empty list of strings")
    splits = dataset.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ConfigError(f"{path}: dataset.splits must be a non-empty mapping")
    for split, split_dir in splits.items():
        if not isinstance(split_dir, str) or not split_dir:
            raise ConfigError(f"{path}: dataset.splits.{split} must be a directory name")
    for key in ("image_dir", "label_dir"):
        if key in dataset and (not isinstance(dataset[key], str) or not dataset[key]):
            raise ConfigError(f"{path}: dataset.{key} must be a directory name")
    exts = dataset.get("image_extensions")
    if exts is not None and (
        not isinstance(exts, list) or not exts or not all(isinstance(e, str) for e in exts)
    ):
        raise ConfigError(f"{path}: dataset.image_extensions must be a list such as [.jpg, .png]")
    return cfg


def resolve_data_root(cfg: dict, cli_value: str | None = None) -> Path:
    """Pick the dataset root: CLI argument, then config value, then $FIREWATCH_DATA_ROOT."""
    value = cli_value or cfg["dataset"].get("root") or os.environ.get(DATA_ROOT_ENV)
    if not value:
        raise ConfigError(
            "dataset root not set: pass --data-root, set dataset.root in the config, "
            f"or export {DATA_ROOT_ENV}"
        )
    return Path(os.path.expandvars(str(value))).expanduser()
