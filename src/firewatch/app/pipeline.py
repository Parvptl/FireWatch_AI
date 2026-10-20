"""Analysis pipeline: source frames -> detector -> event aggregation -> storage and job outputs.

Outputs in the job directory:
    detections.jsonl   one line per analysed frame that has at least one detection
    <event>_alert.jpg  annotated frame at which each event was confirmed (alert time)
    <event>_peak.jpg   annotated frame with the event's highest confidence
    annotated.mp4      analysed frames with boxes (recorded videos only, if enabled)
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path

from firewatch.app.config import AppConfig
from firewatch.app.events import EventAggregator
from firewatch.app.store import Store


class _Annotator:
    """Draws lazily: OpenCV is only touched when an image actually has to be produced."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def draw(self, image, dets, t):
        if not self.enabled or image is None:
            return None
        from firewatch.app.annotate import draw

        return draw(image, dets, t)

    def save(self, path: Path, image) -> str | None:
        if image is None:
            return None
        from firewatch.app.annotate import write_jpeg

        write_jpeg(path, image)
        return path.name


def run_analysis(
    job_id: str,
    source,
    detector,
    cfg: AppConfig,
    store: Store,
    job_dir: Path,
    stop: threading.Event,
    annotate: bool = True,
) -> dict:
    """Process the source until it ends or ``stop`` is set. Returns the job summary.

    Exceptions (decode failures, detector errors) propagate to the caller, which records the job
    as failed; events/alerts already stored stay in the database.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    agg = EventAggregator(cfg.events, cfg.model.class_names, id_prefix=job_id[:8])
    ann = _Annotator(annotate)
    per_class = {
        c: {"frames_with_detections": 0, "boxes": 0, "max_confidence": 0.0}
        for c in cfg.model.class_names
    }
    writer = None
    codec = None
    want_video = annotate and cfg.video.write_annotated_video and source.kind == "video"
    peak_images: dict[str, tuple[int, object]] = {}  # event -> (frame index, annotated image)
    alert_frames: dict[str, str] = {}
    frames_done = 0
    last_progress = 0.0
    last_t = 0.0
    t0 = time.monotonic()
    expected = source.expected_frames

    def persist_event(ev, keyframe=None):
        d = ev.to_dict()
        d["alert_keyframe"] = alert_frames.get(ev.event_id)
        store.upsert_event(job_id, d, keyframe)

    def resolve(events):
        for ev in events:
            frame_idx, img = peak_images.pop(ev.event_id, (None, None))
            name = ann.save(job_dir / f"{ev.event_id}_peak.jpg", img)
            d = ev.to_dict()
            d["alert_keyframe"] = alert_frames.get(ev.event_id)
            d["peak_keyframe"] = name
            # highest-confidence frame seen after confirmation (the overall peak may precede it)
            d["peak_keyframe_frame"] = frame_idx
            store.upsert_event(job_id, d, alert_frames.get(ev.event_id))

    try:
        with (job_dir / "detections.jsonl").open("w", encoding="utf-8") as det_fh:
            for frame in source.frames(stop):
                dets = detector.predict(frame.image)
                frames_done += 1
                last_t = frame.time_s
                if dets:
                    det_fh.write(
                        json.dumps(
                            {
                                "frame": frame.index,
                                "time_s": round(frame.time_s, 3),
                                "detections": [d.to_dict() for d in dets],
                            }
                        )
                        + "\n"
                    )
                    seen = defaultdict(int)
                    for d in dets:
                        s = per_class.setdefault(
                            d.class_name,
                            {"frames_with_detections": 0, "boxes": 0, "max_confidence": 0.0},
                        )
                        s["boxes"] += 1
                        s["max_confidence"] = max(s["max_confidence"], round(d.confidence, 4))
                        seen[d.class_name] += 1
                    for c in seen:
                        per_class[c]["frames_with_detections"] += 1
                upd = agg.update(frame.time_s, frame.index, dets)
                need_img = bool(want_video or upd.activated or upd.peak_improved)
                annotated = ann.draw(frame.image, dets, frame.time_s) if need_img else None
                if want_video and annotated is not None:
                    if writer is None:
                        from firewatch.app.annotate import AnnotatedVideoWriter

                        h, w = annotated.shape[:2]
                        out_fps = source.meta["fps"] / source.stride
                        try:
                            writer = AnnotatedVideoWriter(job_dir / "annotated.mp4", out_fps, w, h)
                            codec = writer.codec
                        except OSError:
                            want_video = False
                    if writer is not None:
                        writer.write(annotated)
                for ev in upd.activated:
                    name = ann.save(job_dir / f"{ev.event_id}_alert.jpg", annotated)
                    if name:
                        alert_frames[ev.event_id] = name
                    if annotated is not None:
                        peak_images[ev.event_id] = (frame.index, annotated)
                    persist_event(ev, name)
                for ev in upd.peak_improved:
                    if annotated is not None:
                        peak_images[ev.event_id] = (frame.index, annotated)
                for a in upd.alerts:
                    store.add_alert(job_id, a.to_dict(), alert_frames.get(a.event_id))
                resolve(upd.resolved)
                now = time.monotonic()
                if now - last_progress >= cfg.jobs.progress_interval_s:
                    last_progress = now
                    store.update_job(
                        job_id,
                        frames_done=frames_done,
                        video_time_s=round(last_t, 3),
                        progress=None if not expected else min(0.999, frames_done / expected),
                    )
                    for ev in agg.open_events():
                        persist_event(ev)
    finally:
        resolve(agg.finish())
        if writer is not None:
            writer.close()
        elapsed = time.monotonic() - t0
        summary = {
            "frames_analysed": frames_done,
            "analysed_until_s": round(last_t, 3),
            "per_class": per_class,
            "events": len(agg.events),
            "alerts": len(agg.alerts),
            "suppressed_alerts": sum(1 for e in agg.events if e.alert_suppressed),
            "discarded_candidates": agg.discarded_candidates,
            "annotated_video": "annotated.mp4" if writer is not None else None,
            "annotated_video_codec": codec,
            "processing_s": round(elapsed, 2),
            "analysed_frames_per_s": round(frames_done / elapsed, 2) if elapsed > 0 else None,
            "stride": getattr(source, "stride", None),
            "reconnects": getattr(source, "reconnects", None),
        }
        store.update_job(
            job_id, frames_done=frames_done, video_time_s=round(last_t, 3), summary=summary
        )
    return summary
