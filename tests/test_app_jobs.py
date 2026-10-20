"""Job lifecycle, queue limits, cancellation, failure handling and persistence across restarts."""

import json

import pytest

from app_fakes import FakeDetector, FakeSource, fire_schedule, make_config, wait_for
from firewatch.app.jobs import (
    Conflict,
    InvalidInput,
    JobManager,
    ModelUnavailable,
    NotFound,
    QueueFull,
)
from firewatch.app.sources import SourceError
from firewatch.app.store import Store


def make_manager(tmp_path, detector=None, source=None, **cfg_sections):
    cfg = make_config(tmp_path, **cfg_sections)
    cfg.input_dir.mkdir(parents=True, exist_ok=True)
    (cfg.input_dir / "clip.mp4").write_bytes(b"x")
    (cfg.input_dir / "other.mp4").write_bytes(b"x")
    store = Store(cfg.db_path)
    factory = source if callable(source) else (lambda path: source or FakeSource())
    mgr = JobManager(
        cfg, store, detector or FakeDetector(), video_source_factory=factory, annotate=False
    )
    mgr.start(warm_model=False)
    return cfg, store, mgr


def finished(store, job_id):
    return lambda: store.get_job(job_id)["status"] not in ("queued", "running")


def test_completed_job_records_events_alerts_and_detections(tmp_path):
    det = FakeDetector(fire_schedule(range(3, 9)))
    cfg, store, mgr = make_manager(tmp_path, det, FakeSource(n_frames=20))
    job = mgr.submit_video("clip.mp4")
    assert job["status"] == "queued" and job["frames_total"] == 20
    assert wait_for(finished(store, job["id"]))
    j = store.get_job(job["id"])
    assert j["status"] == "completed" and j["progress"] == 1.0 and j["frames_done"] == 20
    assert j["summary"]["per_class"]["fire"]["frames_with_detections"] == 6
    assert j["summary"]["events"] == 1 and j["summary"]["alerts"] == 1
    events = store.list_events(job["id"])
    assert len(events) == 1 and events[0]["status"] == "resolved"
    assert events[0]["start_s"] == 3.0 and events[0]["alert_s"] == 4.0 and events[0]["end_s"] == 8.0
    alerts = store.list_alerts(job_id=job["id"])
    assert len(alerts) == 1 and alerts[0]["event_id"] == events[0]["event_id"]
    lines = (cfg.jobs_dir / job["id"] / "detections.jsonl").read_text().splitlines()
    assert len(lines) == 6 and json.loads(lines[0])["frame"] == 3
    mgr.shutdown()


@pytest.mark.parametrize(
    "rel, exc",
    [
        ("../etc/passwd.mp4", InvalidInput),
        ("/abs/clip.mp4", InvalidInput),
        ("clip.txt", InvalidInput),
        ("missing.mp4", NotFound),
        ("", InvalidInput),
        (None, InvalidInput),
    ],
)
def test_invalid_video_paths(tmp_path, rel, exc):
    _, _, mgr = make_manager(tmp_path)
    with pytest.raises(exc):
        mgr.submit_video(rel)
    mgr.shutdown()


def test_symlink_escaping_input_dir_rejected(tmp_path):
    cfg, _, mgr = make_manager(tmp_path)
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"x")
    (cfg.input_dir / "link.mp4").symlink_to(outside)
    with pytest.raises(InvalidInput, match="outside"):
        mgr.submit_video("link.mp4")
    mgr.shutdown()


def test_unreadable_video_rejected_at_submission(tmp_path):
    def bad(path):
        raise SourceError("video contains no decodable frames")

    _, store, mgr = make_manager(tmp_path, source=bad)
    with pytest.raises(InvalidInput, match="no decodable frames"):
        mgr.submit_video("clip.mp4")
    assert store.list_jobs() == []
    mgr.shutdown()


def test_missing_model_refused(tmp_path):
    _, _, mgr = make_manager(tmp_path, FakeDetector(state="missing"))
    with pytest.raises(ModelUnavailable):
        mgr.submit_video("clip.mp4")
    mgr.shutdown()


def test_queue_is_bounded(tmp_path):
    slow = lambda path: FakeSource(n_frames=200, delay=0.02)  # noqa: E731
    _, store, mgr = make_manager(tmp_path, source=slow)
    first = mgr.submit_video("clip.mp4")
    assert wait_for(lambda: store.get_job(first["id"])["status"] == "running")
    mgr.submit_video("clip.mp4")
    mgr.submit_video("other.mp4")  # max_queued = 2 waiting
    with pytest.raises(QueueFull):
        mgr.submit_video("clip.mp4")
    mgr.shutdown(timeout=5)
    assert store.count_active() == 0


def test_cancel_running_and_queued(tmp_path):
    slow = lambda path: FakeSource(n_frames=500, delay=0.01)  # noqa: E731
    _, store, mgr = make_manager(tmp_path, source=slow)
    a = mgr.submit_video("clip.mp4")
    b = mgr.submit_video("other.mp4")
    assert wait_for(lambda: store.get_job(a["id"])["status"] == "running")
    assert mgr.cancel(b["id"])["status"] == "cancelled"
    mgr.cancel(a["id"])
    assert wait_for(finished(store, a["id"]))
    ja = store.get_job(a["id"])
    assert ja["status"] == "cancelled" and 0 < ja["frames_done"] < 500
    with pytest.raises(Conflict):
        mgr.cancel(a["id"])
    with pytest.raises(NotFound):
        mgr.cancel("0123456789abcdef")
    mgr.shutdown()


def test_failures_are_recorded_and_worker_survives(tmp_path):
    det = FakeDetector(fail_at=5)
    _, store, mgr = make_manager(tmp_path, det, FakeSource(n_frames=10))
    j1 = mgr.submit_video("clip.mp4")
    assert wait_for(finished(store, j1["id"]))
    r1 = store.get_job(j1["id"])
    assert r1["status"] == "failed" and "simulated inference failure" in r1["error"]
    assert r1["frames_done"] == 5  # partial progress kept
    det.fail_at = None
    j2 = mgr.submit_video("clip.mp4")
    assert wait_for(finished(store, j2["id"]))
    assert store.get_job(j2["id"])["status"] == "completed"
    mgr.shutdown()


def test_decode_error_mid_video_fails_job(tmp_path):
    _, store, mgr = make_manager(tmp_path, source=lambda p: FakeSource(n_frames=10, raise_after=4))
    j = mgr.submit_video("clip.mp4")
    assert wait_for(finished(store, j["id"]))
    r = store.get_job(j["id"])
    assert r["status"] == "failed" and r["error"].startswith("video error")
    mgr.shutdown()


def test_shutdown_interrupts_running_and_queued(tmp_path):
    slow = lambda path: FakeSource(n_frames=1000, delay=0.01)  # noqa: E731
    _, store, mgr = make_manager(tmp_path, source=slow)
    a = mgr.submit_video("clip.mp4")
    b = mgr.submit_video("other.mp4")
    assert wait_for(lambda: store.get_job(a["id"])["status"] == "running")
    mgr.shutdown(timeout=5)
    assert store.get_job(a["id"])["status"] == "interrupted"
    assert store.get_job(b["id"])["status"] == "interrupted"


def test_results_persist_and_stale_jobs_recovered_after_restart(tmp_path):
    det = FakeDetector(fire_schedule(range(0, 5)))
    cfg, store, mgr = make_manager(tmp_path, det, FakeSource(n_frames=10))
    j = mgr.submit_video("clip.mp4")
    assert wait_for(finished(store, j["id"]))
    mgr.shutdown()
    # a job left 'running' by a crashed process
    store.create_job("ffffffffffffffff", "video", "x", "x.mp4", {}, {}, 10)
    store.update_job("ffffffffffffffff", status="running")
    store.close()

    store2 = Store(cfg.db_path)
    assert store2.recover_after_restart() == 1
    assert store2.get_job("ffffffffffffffff")["status"] == "interrupted"
    assert store2.get_job(j["id"])["status"] == "completed"
    assert len(store2.list_events(j["id"])) == 1 and len(store2.list_alerts()) == 1
    store2.close()


def test_cli_style_manager_does_not_interrupt_live_jobs(tmp_path):
    slow = lambda path: FakeSource(n_frames=300, delay=0.01)  # noqa: E731
    cfg, store, mgr = make_manager(tmp_path, source=slow)
    job = mgr.submit_video("clip.mp4")
    assert wait_for(lambda: store.get_job(job["id"])["status"] == "running")
    other = JobManager(
        cfg, Store(cfg.db_path), FakeDetector(), video_source_factory=slow, annotate=False
    )
    other.start(warm_model=False, recover=False)
    assert store.get_job(job["id"])["status"] == "running"
    other.shutdown()
    mgr.shutdown()
