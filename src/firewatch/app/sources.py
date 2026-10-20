"""Frame sources: recorded video files and network streams (RTSP/HTTP) through OpenCV.

Both yield ``Frame`` objects at a sampled rate and share one interface, so the analysis pipeline
does not care where frames come from. OpenCV is imported lazily.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path


class SourceError(RuntimeError):
    """The input cannot be opened or decoded."""


@dataclass
class Frame:
    index: int  # frame number in the source (0-based)
    time_s: float  # position in the video, or elapsed seconds for a stream
    image: object  # BGR uint8 ndarray (None only in tests)


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SourceError("OpenCV (cv2) is not installed; install the 'train' extra") from exc
    return cv2


def probe_video(path: Path) -> dict:
    """Open a video, read one frame and return its metadata; raise SourceError if unusable."""
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise SourceError("cannot open video (unsupported format or corrupt file)")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise SourceError("video contains no decodable frames")
        if not math.isfinite(fps) or fps <= 0 or fps > 1000:
            raise SourceError(f"video reports an invalid frame rate ({fps})")
        h, w = frame.shape[:2]
        duration = count / fps if count > 0 else None
        return {
            "fps": round(fps, 3),
            "frame_count": count if count > 0 else None,
            "width": int(w),
            "height": int(h),
            "duration_s": None if duration is None else round(duration, 3),
        }
    finally:
        cap.release()


class VideoFileSource:
    kind = "video"

    def __init__(self, path: Path, sample_fps: float, max_duration_s: float | None = None):
        self.path = Path(path)
        self.sample_fps = sample_fps
        self.max_duration_s = max_duration_s
        self.meta = probe_video(self.path)
        dur = self.meta["duration_s"]
        if max_duration_s and dur is not None and dur > max_duration_s:
            raise SourceError(f"video is {dur:.0f} s long; the limit is {max_duration_s:.0f} s")
        fps = self.meta["fps"]
        self.stride = 1 if sample_fps <= 0 else max(1, round(fps / sample_fps))
        self.warning: str | None = None

    @property
    def expected_frames(self) -> int | None:
        n = self.meta["frame_count"]
        return None if n is None else (n + self.stride - 1) // self.stride

    def frames(self, stop: threading.Event | None = None):
        cv2 = _cv2()
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise SourceError("cannot open video (unsupported format or corrupt file)")
        fps = self.meta["fps"]
        count = self.meta["frame_count"]
        idx = 0
        try:
            while stop is None or not stop.is_set():
                if idx % self.stride == 0:
                    ok, img = cap.read()
                    if not ok or img is None:
                        break
                    yield Frame(idx, idx / fps, img)
                elif not cap.grab():
                    break
                idx += 1
                if self.max_duration_s and idx / fps > self.max_duration_s:
                    self.warning = f"stopped at the {self.max_duration_s:.0f} s duration limit"
                    break
        finally:
            cap.release()
        if (
            count
            and idx < count - max(2, self.stride)
            and not (stop and stop.is_set())
            and self.warning is None
        ):
            self.warning = (
                f"decoding stopped at frame {idx} of {count} (truncated or corrupt video)"
            )


class StreamSource:
    """Live stream (rtsp://, http(s)://, or any OpenCV-readable URL).

    Frames are read continuously so the decoder buffer does not lag; one frame is emitted every
    1/sample_fps seconds of stream time. Stream time follows the container's media timestamps
    when they advance and falls back to wall-clock time when a live source provides none. On read
    failure the stream is reopened up to ``reconnect_attempts`` times. Runs until ``stop`` is set
    or ``max_duration_s`` (if > 0) of stream time has been processed.
    """

    kind = "stream"

    def __init__(
        self,
        url: str,
        sample_fps: float,
        reconnect_attempts: int,
        reconnect_delay_s: float,
        max_duration_s: float = 0.0,
    ):
        self.url = url
        self.sample_fps = sample_fps
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s
        self.max_duration_s = max_duration_s
        self.meta: dict = {"url": url}
        self.warning: str | None = None
        self.expected_frames = None
        self.reconnects = 0

    def _open(self):
        cv2 = _cv2()
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def frames(self, stop: threading.Event | None = None):
        cv2 = _cv2()
        start = time.monotonic()
        next_emit = 0.0  # in stream time
        idx = 0
        failures = 0
        emitted = 0
        seg_offset = 0.0  # stream time accumulated over previous connections
        seg_media0 = None  # first media timestamp of the current connection
        last_media = -1.0
        last_t = 0.0
        self.meta["clock"] = None
        cap = self._open()
        while cap is None and failures < self.reconnect_attempts:
            failures += 1
            if stop is not None and stop.wait(self.reconnect_delay_s):
                return
            cap = self._open()
        if cap is None:
            raise SourceError(f"cannot open stream {self.url}")
        try:
            while stop is None or not stop.is_set():
                if self.max_duration_s and last_t >= self.max_duration_s:
                    break
                ok, img = cap.read()
                if not ok or img is None:
                    cap.release()
                    cap = None
                    seg_offset, seg_media0, last_media = last_t, None, -1.0
                    while cap is None and failures < self.reconnect_attempts:
                        failures += 1
                        self.reconnects += 1
                        if stop is not None and stop.wait(self.reconnect_delay_s):
                            return
                        cap = self._open()
                    if cap is None:
                        self.warning = f"stream lost after {emitted} analysed frames"
                        if emitted == 0:
                            raise SourceError(f"stream {self.url} produced no frames")
                        return
                    continue
                failures = 0
                if not self.meta.get("width"):
                    self.meta.update(width=int(img.shape[1]), height=int(img.shape[0]))
                # Stream clock: the container's media timestamps when they advance (files or
                # recorded streams served faster than real time), else wall-clock time (live
                # sources whose timestamps are missing).
                media = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                if media > 0.0 and media >= last_media:
                    if seg_media0 is None:
                        seg_media0 = media
                    t = seg_offset + (media - seg_media0)
                    self.meta["clock"] = "media"
                else:
                    t = time.monotonic() - start
                    self.meta["clock"] = "wall"
                last_media = media
                last_t = max(last_t, t)
                if t >= next_emit:
                    next_emit = t + 1.0 / self.sample_fps
                    emitted += 1
                    yield Frame(idx, t, img)
                idx += 1
        finally:
            if cap is not None:
                cap.release()
