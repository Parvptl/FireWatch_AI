"""Test doubles for the FireWatch application (no OpenCV, NumPy or Ultralytics needed)."""

from __future__ import annotations

import threading
import time

from firewatch.app.config import load_app_config
from firewatch.app.detector import Detection, DetectorUnavailable
from firewatch.app.sources import Frame, SourceError


def make_config(tmp_path, **sections):
    """Config with all directories under tmp_path (outside the repository)."""
    over = {
        "video": {"input_dir": str(tmp_path / "input"), "sample_fps": 0.0},
        "storage": {"data_dir": str(tmp_path / "data")},
        "model": {"weights": str(tmp_path / "model.pt")},
        "api": {"port": 0},
        "jobs": {"progress_interval_s": 0.01, "max_queued": 2, "shutdown_timeout_s": 5},
        "events": {
            "min_confidence": 0.3,
            "activate_hits": 2,
            "activate_window_s": 2.0,
            "resolve_gap_s": 3.0,
            "alert_cooldown_s": 0.0,
        },
    }
    for name, values in sections.items():
        over.setdefault(name, {}).update(values)
    return load_app_config(None, over)


class FakeDetector:
    """Returns detections from ``schedule`` keyed by the frame image (the frame index)."""

    def __init__(self, schedule=None, state="loaded", delay=0.0, fail_at=None):
        self.schedule = schedule or {}
        self.state = state
        self.delay = delay
        self.fail_at = fail_at
        self.calls = 0

    def status(self):
        return {
            "state": self.state,
            "weights": "fake.pt",
            "imgsz": 960,
            "device": "cpu",
            "sha256": "0" * 64,
            "error": None if self.state != "missing" else "checkpoint not found",
        }

    def info(self):
        return {
            "weights": "fake.pt",
            "sha256": "0" * 64,
            "imgsz": 960,
            "device": "cpu",
            "conf_threshold": 0.25,
            "iou_threshold": 0.7,
            "class_names": ["smoke", "fire"],
        }

    def load(self):
        if self.state == "missing":
            raise DetectorUnavailable("checkpoint not found: fake.pt")

    def predict(self, image):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail_at is not None and image == self.fail_at:
            raise RuntimeError("simulated inference failure")
        return [Detection(c, p, (10.0, 10.0, 50.0, 50.0)) for c, p in self.schedule.get(image, [])]


class FakeSource:
    kind = "video"

    def __init__(self, n_frames=20, fps=1.0, delay=0.0, raise_after=None, warning=None):
        self.n = n_frames
        self.fps = fps
        self.delay = delay
        self.raise_after = raise_after
        self.meta = {
            "fps": fps,
            "frame_count": n_frames,
            "width": 64,
            "height": 48,
            "duration_s": n_frames / fps,
        }
        self.stride = 1
        self.expected_frames = n_frames
        self.warning = warning
        self.started = threading.Event()

    def frames(self, stop=None):
        self.started.set()
        for i in range(self.n):
            if stop is not None and stop.is_set():
                return
            if self.raise_after is not None and i >= self.raise_after:
                raise SourceError("simulated decode failure")
            if self.delay:
                time.sleep(self.delay)
            yield Frame(i, i / self.fps, i)


def fire_schedule(frames, conf=0.8, cls="fire"):
    return {i: [(cls, conf)] for i in frames}


def wait_for(predicate, timeout=10.0, interval=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return False
