"""Object detector wrapper around the trained Ultralytics checkpoint.

``Detection`` is a plain dataclass so that event aggregation, storage and the API stay free of
NumPy/torch. ``YoloDetector`` imports Ultralytics only when the model is first loaded.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the analysed frame

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 4)
        d["bbox"] = [round(x, 1) for x in self.bbox]
        return d


class DetectorUnavailable(RuntimeError):
    """The checkpoint is missing or cannot be loaded; analysis cannot start."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class YoloDetector:
    """Thread-safe (serialised) detector for BGR uint8 frames."""

    def __init__(
        self,
        weights: Path,
        imgsz: int,
        device: str,
        class_names: list[str],
        conf: float,
        iou: float,
        max_det: int,
        cpu_threads: int = 0,
    ):
        self.weights = Path(weights)
        self.imgsz = imgsz
        self.device = device
        self.class_names = list(class_names)
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.cpu_threads = cpu_threads
        self._model = None
        self._names: dict[int, str] = {}
        self._lock = threading.Lock()
        self._sha256: str | None = None
        self.load_error: str | None = None

    # ---- status
    def status(self) -> dict:
        if self._model is not None:
            state = "loaded"
        elif not self.weights.is_file():
            state = "missing"
        elif self.load_error:
            state = "error"
        else:
            state = "not_loaded"
        return {
            "state": state,
            "weights": str(self.weights),
            "imgsz": self.imgsz,
            "device": self.device,
            "sha256": self._sha256,
            "error": self.load_error,
        }

    def info(self) -> dict:
        return {
            "weights": str(self.weights),
            "sha256": self._sha256,
            "imgsz": self.imgsz,
            "device": self.device,
            "conf_threshold": self.conf,
            "iou_threshold": self.iou,
            "class_names": self.class_names,
        }

    # ---- loading
    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not self.weights.is_file():
                self.load_error = f"checkpoint not found: {self.weights}"
                raise DetectorUnavailable(self.load_error)
            try:
                if "ultralytics" not in sys.modules:
                    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
                    os.environ.setdefault("YOLO_OFFLINE", "1")
                from ultralytics import YOLO

                if self.device == "cpu" and self.cpu_threads > 0:
                    import torch

                    torch.set_num_threads(self.cpu_threads)
                model = YOLO(str(self.weights))
                names = model.names
                names = names if isinstance(names, dict) else dict(enumerate(names))
                got = [names[i] for i in sorted(names)]
                if got != self.class_names:
                    raise DetectorUnavailable(
                        f"checkpoint classes {got} do not match "
                        f"model.class_names {self.class_names}"
                    )
                self._sha256 = sha256_file(self.weights)
                self._names = {int(k): v for k, v in names.items()}
                self._model = model
                self.load_error = None
            except DetectorUnavailable as exc:
                self.load_error = str(exc)
                raise
            except Exception as exc:  # corrupt checkpoint, missing ultralytics, ...
                self.load_error = (
                    f"cannot load checkpoint {self.weights}: {type(exc).__name__}: {exc}"
                )
                raise DetectorUnavailable(self.load_error) from exc

    # ---- inference
    def predict(self, frame) -> list[Detection]:
        if self._model is None:
            self.load()
        with self._lock:
            res = self._model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                max_det=self.max_det,
                verbose=False,
            )[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().tolist()
        conf = boxes.conf.cpu().tolist()
        cls = boxes.cls.cpu().tolist()
        return [
            Detection(self._names[int(c)], float(p), tuple(float(v) for v in b))
            for b, p, c in zip(xyxy, conf, cls, strict=True)
        ]
