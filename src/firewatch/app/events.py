"""Temporal aggregation of per-frame detections into fire/smoke events and alerts.

Each class is tracked independently with a small state machine:

    idle -> candidate        first frame with a qualifying detection (confidence >= min_confidence)
    candidate -> active      ``activate_hits`` qualifying frames within ``activate_window_s``;
                             an alert is raised once, at activation (unless within the cooldown)
    candidate -> idle        no qualifying frame for ``activate_window_s`` (discarded, no alert)
    active -> resolved       no qualifying frame for ``resolve_gap_s`` (end = last detection)
    end of input             active -> resolved (ended_by="end_of_input"); candidates discarded

Times are seconds on the input's own clock (video position for files, elapsed time for streams).
The aggregator never emits more than one alert per event, and no alert for a class within
``alert_cooldown_s`` of that class's previous alert (the event is still recorded, with
``alert_suppressed=True``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from firewatch.app.config import EventConfig
from firewatch.app.detector import Detection


@dataclass
class Event:
    event_id: str
    class_name: str
    status: str  # candidate | active | resolved | discarded
    start_s: float
    start_frame: int
    last_seen_s: float
    last_seen_frame: int
    alert_s: float | None = None
    alert_frame: int | None = None
    end_s: float | None = None
    end_frame: int | None = None
    ended_by: str | None = None
    hit_frames: int = 0
    conf_sum: float = 0.0
    peak_confidence: float = 0.0
    peak_frame: int | None = None
    peak_bbox: list[float] | None = None
    alert_suppressed: bool = False
    recent_hits: list[float] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float:
        return self.conf_sum / self.hit_frames if self.hit_frames else 0.0

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "class_name": self.class_name,
            "status": self.status,
            "start_s": round(self.start_s, 3),
            "start_frame": self.start_frame,
            "alert_s": None if self.alert_s is None else round(self.alert_s, 3),
            "alert_frame": self.alert_frame,
            "end_s": None if self.end_s is None else round(self.end_s, 3),
            "end_frame": self.end_frame,
            "last_seen_s": round(self.last_seen_s, 3),
            "ended_by": self.ended_by,
            "hit_frames": self.hit_frames,
            "mean_confidence": round(self.mean_confidence, 4),
            "peak_confidence": round(self.peak_confidence, 4),
            "peak_frame": self.peak_frame,
            "peak_bbox": self.peak_bbox,
            "alert_suppressed": self.alert_suppressed,
        }


@dataclass(frozen=True)
class Alert:
    alert_id: str
    event_id: str
    class_name: str
    time_s: float
    frame_index: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "event_id": self.event_id,
            "class_name": self.class_name,
            "time_s": round(self.time_s, 3),
            "frame_index": self.frame_index,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class Update:
    """What changed after one frame: new alerts plus events whose state changed."""

    alerts: list[Alert] = field(default_factory=list)
    activated: list[Event] = field(default_factory=list)
    resolved: list[Event] = field(default_factory=list)
    peak_improved: list[Event] = field(default_factory=list)


class EventAggregator:
    def __init__(self, cfg: EventConfig, class_names: list[str], id_prefix: str = "ev"):
        self.cfg = cfg
        self.class_names = list(class_names)
        self._open: dict[str, Event] = {}
        self._last_alert_s: dict[str, float] = {}
        self._counters = {"e": itertools.count(1), "a": itertools.count(1)}
        self._prefix = id_prefix
        self.events: list[Event] = []  # every activated event (active or resolved), in order
        self.alerts: list[Alert] = []
        self.discarded_candidates = 0

    def _new_id(self, kind: str) -> str:
        return f"{self._prefix}-{kind}{next(self._counters[kind]):04d}"

    def update(self, t: float, frame_index: int, detections: list[Detection]) -> Update:
        out = Update()
        best: dict[str, Detection] = {}
        for d in detections:
            if d.confidence < self.cfg.min_confidence or d.class_name not in self.class_names:
                continue
            if d.class_name not in best or d.confidence > best[d.class_name].confidence:
                best[d.class_name] = d
        for cls in self.class_names:
            ev = self._open.get(cls)
            det = best.get(cls)
            if ev is not None:
                gap = t - ev.last_seen_s
                if ev.status == "candidate" and gap > self.cfg.activate_window_s:
                    ev.status = "discarded"
                    self.discarded_candidates += 1
                    del self._open[cls]
                    ev = None
                elif ev.status == "active" and gap > self.cfg.resolve_gap_s:
                    self._resolve(ev, "gap")
                    out.resolved.append(ev)
                    del self._open[cls]
                    ev = None
            if det is None:
                continue
            if ev is None:
                ev = Event(
                    event_id="",
                    class_name=cls,
                    status="candidate",
                    start_s=t,
                    start_frame=frame_index,
                    last_seen_s=t,
                    last_seen_frame=frame_index,
                )
                self._open[cls] = ev
            ev.last_seen_s, ev.last_seen_frame = t, frame_index
            ev.hit_frames += 1
            ev.conf_sum += det.confidence
            if det.confidence > ev.peak_confidence:
                ev.peak_confidence, ev.peak_frame = det.confidence, frame_index
                ev.peak_bbox = [round(x, 1) for x in det.bbox]
                if ev.status == "active":
                    out.peak_improved.append(ev)
            if ev.status == "candidate":
                ev.recent_hits.append(t)
                ev.recent_hits = [h for h in ev.recent_hits if t - h <= self.cfg.activate_window_s]
                if len(ev.recent_hits) >= self.cfg.activate_hits:
                    ev.status = "active"
                    ev.event_id = self._new_id("e")
                    ev.alert_s, ev.alert_frame = t, frame_index
                    ev.recent_hits = []
                    self.events.append(ev)
                    out.activated.append(ev)
                    last = self._last_alert_s.get(cls)
                    if last is not None and t - last < self.cfg.alert_cooldown_s:
                        ev.alert_suppressed = True
                    else:
                        alert = Alert(
                            self._new_id("a"), ev.event_id, cls, t, frame_index, det.confidence
                        )
                        self._last_alert_s[cls] = t
                        self.alerts.append(alert)
                        out.alerts.append(alert)
        return out

    def _resolve(self, ev: Event, reason: str) -> None:
        ev.status = "resolved"
        ev.end_s, ev.end_frame, ev.ended_by = ev.last_seen_s, ev.last_seen_frame, reason

    def finish(self) -> list[Event]:
        """End of input: resolve active events, discard candidates. Returns newly resolved."""
        resolved = []
        for cls, ev in list(self._open.items()):
            if ev.status == "active":
                self._resolve(ev, "end_of_input")
                resolved.append(ev)
            else:
                ev.status = "discarded"
                self.discarded_candidates += 1
            del self._open[cls]
        return resolved

    def open_events(self) -> list[Event]:
        return [e for e in self._open.values() if e.status == "active"]
