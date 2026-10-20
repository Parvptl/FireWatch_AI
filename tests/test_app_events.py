"""Temporal event aggregation: candidates, activation, single alert per event, resolution."""

from firewatch.app.config import EventConfig
from firewatch.app.detector import Detection
from firewatch.app.events import EventAggregator

CFG = EventConfig(
    min_confidence=0.3,
    activate_hits=3,
    activate_window_s=2.0,
    resolve_gap_s=3.0,
    alert_cooldown_s=0.0,
)


def det(cls="fire", conf=0.8):
    return Detection(cls, conf, (0.0, 0.0, 10.0, 10.0))


def feed(agg, hits, n, step=1.0, cls="fire", conf=0.8):
    """Frames 0..n-1 at `step` seconds; a detection on frames in `hits`."""
    out = []
    for i in range(n):
        out.append(agg.update(i * step, i, [det(cls, conf)] if i in hits else []))
    return out


def test_needs_enough_hits_within_window_before_alerting():
    agg = EventAggregator(CFG, ["smoke", "fire"])
    ups = feed(agg, {0, 1, 2}, 3)
    assert [len(u.alerts) for u in ups] == [0, 0, 1]
    ev = agg.events[0]
    assert ev.status == "active" and ev.start_s == 0.0 and ev.alert_s == 2.0


def test_isolated_blips_never_alert():
    agg = EventAggregator(CFG, ["smoke", "fire"])
    feed(agg, {0, 5, 10, 15}, 20)  # one hit every 5 s, window 2 s
    agg.finish()
    assert agg.alerts == [] and agg.events == []
    assert agg.discarded_candidates == 4


def test_one_alert_per_event_and_resolution_after_gap():
    agg = EventAggregator(CFG, ["smoke", "fire"])
    feed(agg, set(range(0, 10)), 20)  # continuous for 10 s, then silence
    assert len(agg.alerts) == 1
    ev = agg.events[0]
    assert ev.status == "resolved" and ev.ended_by == "gap"
    assert ev.end_s == 9.0 and ev.hit_frames == 10


def test_gap_shorter_than_resolve_keeps_one_event():
    agg = EventAggregator(CFG, ["smoke", "fire"])
    feed(agg, {0, 1, 2, 3, 6, 7, 8}, 9)  # 2-frame gap (< 3 s) inside
    agg.finish()
    assert len(agg.events) == 1 and len(agg.alerts) == 1
    assert agg.events[0].ended_by == "end_of_input" and agg.events[0].end_s == 8.0


def test_new_event_after_resolution_raises_new_alert():
    agg = EventAggregator(CFG, ["smoke", "fire"])
    feed(agg, {0, 1, 2, 10, 11, 12}, 13)
    assert len(agg.events) == 2 and len(agg.alerts) == 2


def test_cooldown_suppresses_second_alert_but_keeps_event():
    cfg = EventConfig(**{**CFG.__dict__, "alert_cooldown_s": 60.0})
    agg = EventAggregator(cfg, ["smoke", "fire"])
    feed(agg, {0, 1, 2, 10, 11, 12}, 13)
    assert len(agg.events) == 2 and len(agg.alerts) == 1
    assert agg.events[1].alert_suppressed is True


def test_classes_are_tracked_independently():
    agg = EventAggregator(CFG, ["smoke", "fire"])
    for i in range(4):
        agg.update(float(i), i, [det("fire"), det("smoke", 0.6)])
    agg.finish()
    assert sorted(e.class_name for e in agg.events) == ["fire", "smoke"]
    assert len(agg.alerts) == 2


def test_low_confidence_and_unknown_classes_ignored():
    agg = EventAggregator(CFG, ["smoke", "fire"])
    for i in range(5):
        agg.update(float(i), i, [det("fire", 0.2), det("person", 0.9)])
    agg.finish()
    assert agg.events == [] and agg.discarded_candidates == 0


def test_peak_and_mean_confidence():
    agg = EventAggregator(CFG, ["smoke", "fire"])
    for i, c in enumerate([0.4, 0.9, 0.5]):
        agg.update(float(i), i, [det("fire", c)])
    ev = agg.events[0]
    assert ev.peak_confidence == 0.9 and ev.peak_frame == 1
    assert abs(ev.mean_confidence - 0.6) < 1e-9


def test_finish_resolves_active_and_reports_dict():
    agg = EventAggregator(CFG, ["smoke", "fire"])
    feed(agg, {0, 1, 2, 3}, 4)
    resolved = agg.finish()
    assert len(resolved) == 1
    d = resolved[0].to_dict()
    assert d["status"] == "resolved" and d["end_s"] == 3.0 and d["alert_s"] == 2.0
    assert agg.open_events() == []
