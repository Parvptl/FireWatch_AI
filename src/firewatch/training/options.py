"""Resolve and validate training options from config + command line.

Precedence for every option: command line > ``train:`` section of the config >
environment variable (where noted) > built-in default.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from firewatch.utils.paths import expand_path

DATA_YAML_ENV = "FIREWATCH_DATA_YAML"
RUNS_ROOT_ENV = "FIREWATCH_RUNS_ROOT"
DEFAULT_RUNS_ROOT = "~/firewatch_runs"

# Options that a resumed run takes from its checkpoint; passing them with
# --resume is rejected rather than silently ignored.
FRESH_RUN_ONLY = (
    "data",
    "weights",
    "imgsz",
    "epochs",
    "patience",
    "batch",
    "seed",
    "deterministic",
    "fraction",
    "amp",
    "plots",
    "optimizer",
    "project",
    "name",
)
# Options Ultralytics allows to change on resume and that this CLI exposes.
RESUME_OVERRIDABLE = ("device", "workers")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_DEVICE_RE = re.compile(r"^(cpu|\d+)$")


class OptionsError(ValueError):
    """Raised for invalid or inconsistent training options."""


@dataclass(frozen=True)
class TrainOptions:
    config_path: Path
    class_names: tuple[str, ...]
    data: Path | None
    weights: Path | None
    imgsz: int
    epochs: int
    patience: int
    batch: int
    seed: int
    deterministic: bool
    device: str
    workers: int
    fraction: float
    amp: bool
    plots: bool
    optimizer: str
    project: Path | None
    name: str | None
    resume: Path | None
    allow_network: bool

    @property
    def run_dir(self) -> Path:
        if self.resume is not None:
            return self.resume.parent.parent  # <run_dir>/weights/last.pt
        assert self.project is not None and self.name is not None
        return self.project / self.name

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        d["class_names"] = list(self.class_names)
        d["run_dir"] = str(self.run_dir)
        return d


def _int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OptionsError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise OptionsError(f"{name} must be >= {minimum}, got {value}")
    return value


def _bool(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise OptionsError(f"{name} must be true or false, got {value!r}")
    return value


def validate_device(value) -> str:
    device = str(value).strip().lower()
    if "," in device:
        raise OptionsError(
            f"device={value!r}: multi-GPU training is not supported by this pipeline; "
            "request one GPU and use device 0"
        )
    if not _DEVICE_RE.match(device):
        raise OptionsError(f"device must be 'cpu' or a single GPU index such as '0', got {value!r}")
    return device


def resolve_train_options(cfg: dict, args, config_path: str | Path) -> TrainOptions:
    """Combine the config's ``train:`` section with parsed CLI ``args``.

    ``args`` is an ``argparse.Namespace`` whose option attributes default to
    ``None`` (meaning "not given on the command line").
    """
    train = cfg.get("train") or {}
    if not isinstance(train, dict):
        raise OptionsError("'train' section of the config must be a mapping")
    config_path = Path(config_path)

    def pick(key: str, default=None):
        cli = getattr(args, key, None)
        if cli is not None:
            return cli
        return train.get(key, default)

    resume = getattr(args, "resume", None)
    if resume is not None:
        given = [k for k in FRESH_RUN_ONLY if getattr(args, k, None) is not None]
        if given:
            flags = ", ".join(f"--{k.replace('_', '-')}" for k in given)
            raise OptionsError(
                f"{flags} cannot be combined with --resume: a resumed run reuses the "
                f"checkpoint's settings. Only {', '.join('--' + k for k in RESUME_OVERRIDABLE)} "
                "may be changed. Start a new run (new --name) to change anything else."
            )

    imgsz = _int("imgsz", pick("imgsz", 640), 32)
    if imgsz % 32:
        raise OptionsError(f"imgsz must be a multiple of 32 (model stride), got {imgsz}")
    fraction = pick("fraction", 1.0)
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0 < fraction <= 1
    ):
        raise OptionsError(f"fraction must be in (0, 1], got {fraction!r}")
    optimizer = pick("optimizer", "auto")
    if not isinstance(optimizer, str) or not optimizer:
        raise OptionsError(f"optimizer must be a non-empty string, got {optimizer!r}")

    data = weights = project = name = None
    if resume is None:
        data_value = pick("data") or os.environ.get(DATA_YAML_ENV)
        if not data_value:
            raise OptionsError(
                f"no dataset YAML: pass --data, set train.data, or export {DATA_YAML_ENV}. "
                "Build one with scripts/prepare_data.py --output-dir (docs/training.md)."
            )
        data = expand_path(data_value)
        weights_value = pick("weights")
        if not weights_value:
            raise OptionsError("no model weights: pass --weights or set train.weights")
        weights = expand_path(weights_value)
        project = expand_path(pick("project") or os.environ.get(RUNS_ROOT_ENV) or DEFAULT_RUNS_ROOT)
        name = pick("name") or config_path.stem
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise OptionsError(
                f"name must match {_NAME_RE.pattern} (letters, digits, '.', '_', '-'), got {name!r}"
            )

    class_names = cfg.get("dataset", {}).get("class_names") or []
    return TrainOptions(
        config_path=config_path,
        class_names=tuple(class_names),
        data=data,
        weights=weights,
        imgsz=imgsz,
        epochs=_int("epochs", pick("epochs", 100), 1),
        patience=_int("patience", pick("patience", 20), 0),
        batch=_int("batch", pick("batch", 16), 1),
        seed=_int("seed", pick("seed", 0), 0),
        deterministic=_bool("deterministic", pick("deterministic", True)),
        device=validate_device(pick("device", "0")),
        workers=_int("workers", pick("workers", 8), 0),
        fraction=float(fraction),
        amp=_bool("amp", pick("amp", True)),
        plots=_bool("plots", pick("plots", True)),
        optimizer=optimizer,
        project=project,
        name=name,
        resume=expand_path(resume) if resume is not None else None,
        allow_network=bool(getattr(args, "allow_network", False)),
    )


def ultralytics_train_kwargs(opts: TrainOptions) -> dict:
    """Keyword arguments for ``YOLO(...).train`` (names verified against Ultralytics 8.4.176)."""
    if opts.resume is not None:
        return {"resume": True, "device": opts.device, "workers": opts.workers}
    return {
        "data": str(opts.data),
        "epochs": opts.epochs,
        "imgsz": opts.imgsz,
        "batch": opts.batch,
        "seed": opts.seed,
        "deterministic": opts.deterministic,
        "device": opts.device,
        "workers": opts.workers,
        "fraction": opts.fraction,
        "patience": opts.patience,
        "amp": opts.amp,
        "plots": opts.plots,
        "optimizer": opts.optimizer,
        "project": str(opts.project),
        "name": opts.name,
        "exist_ok": True,  # the run dir is created and checked by FireWatch beforehand
        "pretrained": True,
        "cache": False,
        "val": True,
        "verbose": True,
    }
