"""Video decoding, sampling, corrupt inputs, annotated outputs and stream source (OpenCV)."""

import threading

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from app_fakes import FakeDetector, make_config  # noqa: E402
from firewatch.app.jobs import JobManager  # noqa: E402
from firewatch.app.pipeline import run_analysis  # noqa: E402
from firewatch.app.sources import (  # noqa: E402
    SourceError,
    StreamSource,
    VideoFileSource,
    probe_video,
)
from firewatch.app.store import Store  # noqa: E402


def write_video(path, n=50, fps=10.0, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert w.isOpened()
    for i in range(n):
        img = np.full((size[1], size[0], 3), i * 4 % 255, np.uint8)
        cv2.putText(img, str(i), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        w.write(img)
    w.release()
    return path


class BrightnessDetector(FakeDetector):
    """'Detects' fire on frames whose mean brightness is above a threshold (real pixel input)."""

    def predict(self, image):
        from firewatch.app.detector import Detection

        self.calls += 1
        return [Detection("fire", 0.9, (5.0, 5.0, 60.0, 60.0))] if image.mean() > 60 else []


def test_probe_and_sampling(tmp_path):
    p = write_video(tmp_path / "v.mp4", n=50, fps=10.0)
    meta = probe_video(p)
    assert meta["fps"] == 10.0 and meta["frame_count"] == 50 and meta["width"] == 160
    src = VideoFileSource(p, sample_fps=2.0)
    frames = list(src.frames())
    assert src.stride == 5 and [f.index for f in frames] == list(range(0, 50, 5))
    assert [f.time_s for f in frames][:3] == [0.0, 0.5, 1.0]
    assert frames[0].image.shape == (120, 160, 3) and src.warning is None
    assert len(list(VideoFileSource(p, sample_fps=0).frames())) == 50


def test_corrupt_and_empty_files_rejected(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x01garbage" * 100)
    with pytest.raises(SourceError):
        probe_video(bad)
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(SourceError):
        VideoFileSource(empty, 2.0)


def test_truncated_video_reports_warning(tmp_path):
    p = tmp_path / "v.avi"  # MJPG/AVI still opens when cut short (an MP4 loses its index)
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    for i in range(200):
        w.write(np.full((120, 160, 3), i % 255, np.uint8))
    w.release()
    data = p.read_bytes()
    t = tmp_path / "trunc.avi"
    t.write_bytes(data[: len(data) // 2])
    src = VideoFileSource(t, sample_fps=0)
    n = len(list(src.frames()))
    assert 0 < n < 200
    assert "truncated" in src.warning


def test_duration_limit(tmp_path):
    p = write_video(tmp_path / "v.mp4", n=50, fps=10.0)
    with pytest.raises(SourceError, match="limit"):
        VideoFileSource(p, 2.0, max_duration_s=3.0)


def test_pipeline_writes_keyframes_video_and_events(tmp_path):
    cfg = make_config(tmp_path, video={"sample_fps": 0.0})
    p = write_video(tmp_path / "v.mp4", n=40, fps=10.0)  # brightness > 60 from frame 16 on
    store = Store(cfg.db_path)
    store.create_job("aaaaaaaaaaaaaaaa", "video", str(p), "v.mp4", {}, {}, 40)
    job_dir = cfg.jobs_dir / "aaaaaaaaaaaaaaaa"
    summary = run_analysis(
        "aaaaaaaaaaaaaaaa",
        VideoFileSource(p, 0),
        BrightnessDetector(),
        cfg,
        store,
        job_dir,
        threading.Event(),
        annotate=True,
    )
    assert summary["frames_analysed"] == 40 and summary["events"] == 1
    assert (
        summary["annotated_video"] == "annotated.mp4"
        and (job_dir / "annotated.mp4").stat().st_size > 0
    )
    ev = store.list_events("aaaaaaaaaaaaaaaa")[0]
    assert ev["status"] == "resolved" and ev["ended_by"] == "end_of_input"
    assert (job_dir / ev["alert_keyframe"]).is_file() and (job_dir / ev["peak_keyframe"]).is_file()
    assert cv2.imread(str(job_dir / ev["alert_keyframe"])) is not None
    assert probe_video(job_dir / "annotated.mp4")["frame_count"] == 40
    store.close()


def test_job_manager_with_real_video(tmp_path):
    cfg = make_config(tmp_path, video={"sample_fps": 5.0})
    cfg.input_dir.mkdir(parents=True)
    write_video(cfg.input_dir / "v.mp4", n=40, fps=10.0)
    store = Store(cfg.db_path)
    mgr = JobManager(cfg, store, BrightnessDetector())
    mgr.start(warm_model=False)
    job = mgr.submit_video("v.mp4")
    assert job["frames_total"] == 20 and job["meta"]["fps"] == 10.0
    from app_fakes import wait_for

    assert wait_for(lambda: store.get_job(job["id"])["status"] == "completed")
    assert store.get_job(job["id"])["summary"]["frames_analysed"] == 20
    mgr.shutdown()
    store.close()


def test_stream_source_reads_and_stops(tmp_path):
    p = write_video(tmp_path / "v.mp4", n=30, fps=10.0)
    # OpenCV opens a local path through the same capture API used for rtsp:// URLs
    src = StreamSource(str(p), sample_fps=1000.0, reconnect_attempts=0, reconnect_delay_s=0.0)
    frames = list(src.frames())
    # wall-clock sampling: a local file decodes faster than real time, so frames are skipped
    idx = [f.index for f in frames]
    assert 1 <= len(frames) <= 30 and idx == sorted(set(idx)) and idx[0] == 0
    assert src.warning.startswith("stream lost")  # end of input is treated as a lost stream
    stop = threading.Event()
    src2 = StreamSource(str(p), sample_fps=1000.0, reconnect_attempts=0, reconnect_delay_s=0.0)
    got = []
    for f in src2.frames(stop):
        got.append(f)
        stop.set()
    assert len(got) == 1  # stops as soon as the stop event is set


def test_stream_source_unreachable(tmp_path):
    src = StreamSource(
        str(tmp_path / "missing.mp4"), 2.0, reconnect_attempts=1, reconnect_delay_s=0.0
    )
    with pytest.raises(SourceError, match="cannot open stream"):
        list(src.frames(threading.Event()))


def test_stream_source_samples_on_media_time(tmp_path):
    p = write_video(tmp_path / "v.mp4", n=30, fps=10.0)  # 3 s of media, decodes in milliseconds
    src = StreamSource(str(p), sample_fps=2.0, reconnect_attempts=0, reconnect_delay_s=0.0)
    frames = list(src.frames())
    assert src.meta["clock"] == "media"
    assert 5 <= len(frames) <= 7  # ~2 frames per second of media, not of wall-clock time
    gaps = [b.time_s - a.time_s for a, b in zip(frames, frames[1:], strict=False)]
    assert all(g >= 0.49 for g in gaps)
