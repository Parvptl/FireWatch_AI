"""Detector wrapper: missing/corrupt checkpoints always; real inference only when opted in.

The suite must never import Ultralytics in-process (see conftest.py), so checks that load a
checkpoint run in a child Python process.

Set FIREWATCH_TEST_WEIGHTS=/path/to/best.pt to run the real-checkpoint test (CPU, a few seconds).
This checks that the software runs the model end to end; it is not a model evaluation.
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

from firewatch.app.detector import DetectorUnavailable, YoloDetector

HAVE_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None


def make(weights, **kw):
    args = dict(
        imgsz=320,
        device="cpu",
        class_names=["smoke", "fire"],
        conf=0.25,
        iou=0.7,
        max_det=50,
        cpu_threads=2,
    )
    args.update(kw)
    return YoloDetector(weights, **args)


def run_child(code: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "YOLO_OFFLINE": "1"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_missing_checkpoint(tmp_path):
    d = make(tmp_path / "nope.pt")
    assert d.status()["state"] == "missing"
    with pytest.raises(DetectorUnavailable, match="not found"):
        d.load()


@pytest.mark.skipif(not HAVE_ULTRALYTICS, reason="ultralytics not installed")
def test_corrupt_checkpoint(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"this is not a checkpoint")
    out = run_child(f"""
        import json
        from firewatch.app.detector import DetectorUnavailable, YoloDetector
        d = YoloDetector({str(bad)!r}, 320, "cpu", ["smoke", "fire"], 0.25, 0.7, 50)
        try:
            d.load(); msg = None
        except DetectorUnavailable as e:
            msg = str(e)
        print(json.dumps({{"msg": msg, "state": d.status()["state"]}}))
    """)
    assert out["state"] == "error" and "cannot load checkpoint" in out["msg"]


WEIGHTS = os.environ.get("FIREWATCH_TEST_WEIGHTS")


@pytest.mark.skipif(
    not WEIGHTS, reason="set FIREWATCH_TEST_WEIGHTS to run real-checkpoint inference"
)
def test_real_checkpoint_inference():
    out = run_child(f"""
        import json
        import numpy as np
        from firewatch.app.detector import DetectorUnavailable, YoloDetector
        d = YoloDetector({WEIGHTS!r}, 960, "cpu", ["smoke", "fire"], 0.25, 0.7, 50, 2)
        d.load()
        dets = d.predict(np.zeros((360, 640, 3), np.uint8))
        try:
            YoloDetector({WEIGHTS!r}, 960, "cpu", ["fire", "smoke"], 0.25, 0.7, 50).load()
            mismatch = None
        except DetectorUnavailable as e:
            mismatch = str(e)
        print(json.dumps({{"status": d.status(), "dets": [x.to_dict() for x in dets],
                           "mismatch": mismatch}}))
    """)
    assert out["status"]["state"] == "loaded" and len(out["status"]["sha256"]) == 64
    for x in out["dets"]:
        assert x["class_name"] in ("smoke", "fire") and 0.25 <= x["confidence"] <= 1.0
        x1, y1, x2, y2 = x["bbox"]
        assert 0 <= x1 <= x2 <= 640 and 0 <= y1 <= y2 <= 360
    assert "do not match" in out["mismatch"]
