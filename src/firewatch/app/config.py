"""Application configuration: one YAML file (configs/app.yaml) plus a few environment overrides.

Every setting has a default here; the YAML only needs to list what differs. Environment variables
override the file for values that are machine- or secret-specific:

    FIREWATCH_APP_CONFIG   path of the YAML file (used by scripts when --config is not given)
    FIREWATCH_MODEL_PATH   model.weights
    FIREWATCH_APP_HOME     base directory for video.input_dir and storage.data_dir defaults
    FIREWATCH_API_TOKEN    api.token
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from firewatch.utils.config import ConfigError, deep_merge
from firewatch.utils.paths import ensure_outside_repo, expand_path

CONFIG_ENV = "FIREWATCH_APP_CONFIG"
DEFAULT_HOME = "~/firewatch_app"


@dataclass
class ModelConfig:
    weights: str = "~/firewatch_runs/baseline_imgsz960/weights/best.pt"
    imgsz: int = 960
    device: str = "cpu"  # "cpu" or a CUDA index such as "0"
    class_names: list[str] = field(default_factory=lambda: ["smoke", "fire"])
    conf_threshold: float = 0.25  # detections below are not reported at all
    iou_threshold: float = 0.7
    max_det: int = 100
    cpu_threads: int = 4  # torch threads when device == "cpu" (0 = library default)


@dataclass
class VideoConfig:
    input_dir: str = f"{DEFAULT_HOME}/input"  # only videos under this directory may be analysed
    allowed_extensions: list[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mkv", ".mov", ".m4v"]
    )
    sample_fps: float = 2.0  # analysed frames per second of video; 0 = every frame
    max_upload_mb: int = 500
    max_duration_s: float = 3 * 3600.0  # longer videos are rejected
    write_annotated_video: bool = True


@dataclass
class EventConfig:
    min_confidence: float = 0.35  # detections below this do not count towards events
    activate_hits: int = 3  # analysed frames with a detection needed to confirm an event...
    activate_window_s: float = 4.0  # ...within this many seconds
    resolve_gap_s: float = 10.0  # an active event ends after this long without a detection
    alert_cooldown_s: float = 30.0  # no new alert for a class within this time after an alert


@dataclass
class StreamConfig:
    enabled: bool = True
    allowed_schemes: list[str] = field(default_factory=lambda: ["rtsp", "rtsps", "http", "https"])
    max_concurrent: int = 1
    reconnect_attempts: int = 5
    reconnect_delay_s: float = 2.0
    default_max_duration_s: float = 0.0  # 0 = run until cancelled
    sample_fps: float = 2.0


@dataclass
class StorageConfig:
    data_dir: str = f"{DEFAULT_HOME}/data"  # SQLite database and per-job outputs


@dataclass
class JobConfig:
    max_queued: int = 8  # submissions beyond this are refused (HTTP 429)
    shutdown_timeout_s: float = 30.0
    progress_interval_s: float = 1.0


@dataclass
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    token: str | None = None  # required for non-loopback hosts
    max_json_bytes: int = 64 * 1024


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    events: EventConfig = field(default_factory=EventConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    jobs: JobConfig = field(default_factory=JobConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    source_file: str | None = None

    # ---- resolved paths
    @property
    def weights_path(self) -> Path:
        return expand_path(self.model.weights)

    @property
    def input_dir(self) -> Path:
        return expand_path(self.video.input_dir)

    @property
    def data_dir(self) -> Path:
        return expand_path(self.storage.data_dir)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "firewatch.sqlite3"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    def public_dict(self) -> dict:
        """Configuration safe to show in the API/dashboard (no token)."""
        d = asdict(self)
        d["api"] = {k: v for k, v in d["api"].items() if k != "token"}
        d["api"]["token_required"] = bool(self.api.token)
        return d


_SECTIONS = {f.name: f.type for f in fields(AppConfig) if f.name != "source_file"}
_SECTION_CLASSES = {
    "model": ModelConfig,
    "video": VideoConfig,
    "events": EventConfig,
    "stream": StreamConfig,
    "storage": StorageConfig,
    "jobs": JobConfig,
    "api": ApiConfig,
}


def _build_section(name: str, values: dict, where: str):
    cls = _SECTION_CLASSES[name]
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) in '{name}': {', '.join(unknown)}")
    default = cls()
    kwargs = {}
    for key, value in values.items():
        ref = getattr(default, key)
        if value is None and key not in ("token",):
            raise ConfigError(f"{where}: {name}.{key} must not be null")
        if isinstance(ref, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{where}: {name}.{key} must be true or false")
        elif isinstance(ref, int) and not isinstance(ref, bool):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{where}: {name}.{key} must be an integer")
        elif isinstance(ref, float):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigError(f"{where}: {name}.{key} must be a number")
            value = float(value)
        elif isinstance(ref, list):
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ConfigError(f"{where}: {name}.{key} must be a list of strings")
        elif isinstance(ref, str) or key == "token":
            if value is not None and not isinstance(value, str | int):
                raise ConfigError(f"{where}: {name}.{key} must be a string")
            value = None if value is None else str(value)
        kwargs[key] = value
    return cls(**kwargs)


def validate(cfg: AppConfig) -> AppConfig:
    """Range and consistency checks; raises ConfigError with a precise message."""
    m, v, e, s, j, a = cfg.model, cfg.video, cfg.events, cfg.stream, cfg.jobs, cfg.api

    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ConfigError(msg)

    need(m.imgsz >= 32 and m.imgsz % 32 == 0, "model.imgsz must be a positive multiple of 32")
    need(len(m.class_names) > 0, "model.class_names must not be empty")
    need(0.0 <= m.conf_threshold < 1.0, "model.conf_threshold must be in [0, 1)")
    need(0.0 < m.iou_threshold <= 1.0, "model.iou_threshold must be in (0, 1]")
    need(m.max_det >= 1, "model.max_det must be >= 1")
    need(m.cpu_threads >= 0, "model.cpu_threads must be >= 0")
    need(m.device == "cpu" or m.device.isdigit(), "model.device must be 'cpu' or a GPU index")
    need(v.sample_fps >= 0.0, "video.sample_fps must be >= 0 (0 = every frame)")
    need(v.max_upload_mb >= 1, "video.max_upload_mb must be >= 1")
    need(v.max_duration_s > 0, "video.max_duration_s must be > 0")
    need(
        all(x.startswith(".") for x in v.allowed_extensions) and v.allowed_extensions,
        "video.allowed_extensions must be a non-empty list like [.mp4]",
    )
    need(0.0 <= e.min_confidence < 1.0, "events.min_confidence must be in [0, 1)")
    need(e.activate_hits >= 1, "events.activate_hits must be >= 1")
    need(e.activate_window_s > 0, "events.activate_window_s must be > 0")
    need(e.resolve_gap_s > 0, "events.resolve_gap_s must be > 0")
    need(e.alert_cooldown_s >= 0, "events.alert_cooldown_s must be >= 0")
    need(s.max_concurrent >= 0, "stream.max_concurrent must be >= 0")
    need(s.reconnect_attempts >= 0, "stream.reconnect_attempts must be >= 0")
    need(s.reconnect_delay_s >= 0, "stream.reconnect_delay_s must be >= 0")
    need(s.default_max_duration_s >= 0, "stream.default_max_duration_s must be >= 0")
    need(s.sample_fps > 0, "stream.sample_fps must be > 0")
    need(j.max_queued >= 1, "jobs.max_queued must be >= 1")
    need(j.shutdown_timeout_s >= 0, "jobs.shutdown_timeout_s must be >= 0")
    need(j.progress_interval_s > 0, "jobs.progress_interval_s must be > 0")
    need(1 <= a.port <= 65535 or a.port == 0, "api.port must be 0-65535")
    need(a.max_json_bytes >= 1024, "api.max_json_bytes must be >= 1024")
    if not a.token:
        try:
            loopback = a.host == "localhost" or ipaddress.ip_address(a.host).is_loopback
        except ValueError:
            loopback = False
        need(
            loopback, "api.token (or FIREWATCH_API_TOKEN) is required when api.host is not loopback"
        )
    for what, path in (("storage.data_dir", cfg.data_dir), ("video.input_dir", cfg.input_dir)):
        try:
            ensure_outside_repo(path, what)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    return cfg


def load_app_config(
    path: str | os.PathLike | None = None, overrides: dict | None = None
) -> AppConfig:
    """Load defaults <- YAML file <- ``overrides`` <- environment variables, then validate."""
    raw: dict = {}
    where = "defaults"
    if path is None:
        path = os.environ.get(CONFIG_ENV)
    if path:
        p = expand_path(path)
        if not p.is_file():
            raise ConfigError(f"app config not found: {p}")
        with p.open(encoding="utf-8") as fh:
            try:
                raw = yaml.safe_load(fh) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"{p}: invalid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{p}: top level must be a mapping")
        where = str(p)
    if overrides:
        raw = deep_merge(raw, overrides)
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ConfigError(f"{where}: unknown section(s): {', '.join(unknown)}")
    sections = {}
    for name in _SECTION_CLASSES:
        values = raw.get(name) or {}
        if not isinstance(values, dict):
            raise ConfigError(f"{where}: '{name}' must be a mapping")
        sections[name] = _build_section(name, values, where)
    cfg = AppConfig(**sections, source_file=None if where == "defaults" else where)
    home = os.environ.get("FIREWATCH_APP_HOME")
    if home:
        if "input_dir" not in (raw.get("video") or {}):
            cfg.video.input_dir = str(Path(home) / "input")
        if "data_dir" not in (raw.get("storage") or {}):
            cfg.storage.data_dir = str(Path(home) / "data")
    if os.environ.get("FIREWATCH_MODEL_PATH"):
        cfg.model.weights = os.environ["FIREWATCH_MODEL_PATH"]
    if os.environ.get("FIREWATCH_API_TOKEN"):
        cfg.api.token = os.environ["FIREWATCH_API_TOKEN"]
    return validate(cfg)
