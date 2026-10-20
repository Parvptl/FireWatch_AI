#!/usr/bin/env python
"""Analyse one recorded video from the command line (no server), using the same pipeline as the API.

The result is stored in the application database like an API job, so it also appears in the
dashboard. Prints the job summary, events and alerts as JSON.

Example:
    python scripts/analyze_video.py --config configs/app.yaml --video site_a/cam1_2026-10-19.mp4

``--video`` is relative to video.input_dir (absolute paths inside it are accepted too).
Exit codes: 0 = completed, 1 = analysis failed, 2 = invalid input/configuration, 130 = interrupted.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

from firewatch.utils.config import ConfigError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", help="app config YAML (default: $FIREWATCH_APP_CONFIG or defaults)")
    p.add_argument("--video", required=True)
    p.add_argument("--device", help="'cpu' or a GPU index; overrides model.device")
    p.add_argument("--weights", help="checkpoint path; overrides model.weights")
    a = p.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from firewatch.app.config import load_app_config
    from firewatch.app.detector import YoloDetector
    from firewatch.app.jobs import JobError, JobManager
    from firewatch.app.store import Store

    overrides: dict = {}
    if a.device:
        overrides.setdefault("model", {})["device"] = a.device
    if a.weights:
        overrides.setdefault("model", {})["weights"] = a.weights
    try:
        cfg = load_app_config(a.config, overrides)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    video = Path(a.video)
    if video.is_absolute():
        try:
            video = video.resolve().relative_to(cfg.input_dir.resolve())
        except ValueError:
            print(
                f"error: {a.video} is not inside the input directory {cfg.input_dir}",
                file=sys.stderr,
            )
            return 2
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(cfg.db_path)
    m = cfg.model
    det = YoloDetector(
        cfg.weights_path,
        m.imgsz,
        m.device,
        m.class_names,
        m.conf_threshold,
        m.iou_threshold,
        m.max_det,
        m.cpu_threads,
    )
    mgr = JobManager(cfg, store, det)
    mgr.start(warm_model=False, recover=False)  # never touch a live server's jobs
    try:
        job = mgr.submit_video(video.as_posix())
    except JobError as exc:
        print(f"error ({exc.code}): {exc.message}", file=sys.stderr)
        mgr.shutdown(0)
        return 2

    interrupted = False

    def _stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        mgr.cancel(job["id"])

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    import time

    while store.get_job(job["id"])["status"] in ("queued", "running"):
        time.sleep(0.5)
    mgr.shutdown()
    final = store.get_job(job["id"])
    out = {
        "job": final,
        "events": store.list_events(job["id"]),
        "alerts": store.list_alerts(1000, job["id"]),
        "outputs": str(cfg.jobs_dir / job["id"]),
    }
    print(json.dumps(out, indent=1, default=str))
    store.close()
    if interrupted:
        return 130
    return 0 if final["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
