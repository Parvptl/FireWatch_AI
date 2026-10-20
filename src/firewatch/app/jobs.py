"""Job management: bounded queue, one video worker, optional stream workers, cancel and shutdown.

Recorded videos are processed one at a time by a single worker thread (the model is shared and
inference is serialised). Submissions beyond ``jobs.max_queued`` waiting jobs are refused.
Streams run in their own threads (at most ``stream.max_concurrent``) until cancelled or their
duration limit. On shutdown running jobs are stopped and recorded as ``interrupted`` with the
results obtained so far; jobs still queued are recorded as ``interrupted`` too.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from firewatch.app.config import AppConfig
from firewatch.app.detector import DetectorUnavailable
from firewatch.app.pipeline import run_analysis
from firewatch.app.sources import SourceError, StreamSource, VideoFileSource
from firewatch.app.store import Store

log = logging.getLogger("firewatch.app")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class JobError(Exception):
    status = 400
    code = "bad_request"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidInput(JobError):
    status, code = 400, "invalid_input"


class NotFound(JobError):
    status, code = 404, "not_found"


class Conflict(JobError):
    status, code = 409, "conflict"


class QueueFull(JobError):
    status, code = 429, "queue_full"


class ModelUnavailable(JobError):
    status, code = 503, "model_unavailable"


class ShuttingDown(JobError):
    status, code = 503, "shutting_down"


def safe_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name
    base = _SAFE.sub("_", base).strip("._") or "video"
    return base[:120]


class JobManager:
    def __init__(
        self,
        cfg: AppConfig,
        store: Store,
        detector,
        video_source_factory: Callable | None = None,
        stream_source_factory: Callable | None = None,
        annotate: bool = True,
    ):
        self.cfg = cfg
        self.store = store
        self.detector = detector
        self.video_source_factory = video_source_factory or (
            lambda path: VideoFileSource(path, cfg.video.sample_fps, cfg.video.max_duration_s)
        )
        self.stream_source_factory = stream_source_factory or (
            lambda url, max_s: StreamSource(
                url,
                cfg.stream.sample_fps,
                cfg.stream.reconnect_attempts,
                cfg.stream.reconnect_delay_s,
                max_s,
            )
        )
        self.annotate = annotate
        self._queue: queue.Queue = queue.Queue(maxsize=cfg.jobs.max_queued)
        self._sources: dict[str, object] = {}
        self._stops: dict[str, threading.Event] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()
        self._closing = False
        self._worker: threading.Thread | None = None
        self._stream_threads: dict[str, threading.Thread] = {}

    # ---- lifecycle
    def start(self, warm_model: bool = True, recover: bool = True) -> None:
        """Start the worker. ``recover`` marks jobs left queued/running by a previous server
        process as interrupted; only the server (the single owner of the data directory) should
        do this, never a CLI run that may share the database with a live server."""
        if recover:
            n = self.store.recover_after_restart()
            if n:
                log.warning("marked %d job(s) from a previous run as interrupted", n)
        self.cfg.input_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._worker = threading.Thread(target=self._work_loop, name="video-worker", daemon=True)
        self._worker.start()
        if warm_model:
            threading.Thread(target=self._warm, name="model-warmup", daemon=True).start()

    def _warm(self) -> None:
        try:
            self.detector.load()
            log.info("model loaded: %s", self.detector.status())
        except DetectorUnavailable as exc:
            log.error("model unavailable: %s", exc)

    def shutdown(self, timeout: float | None = None) -> None:
        timeout = self.cfg.jobs.shutdown_timeout_s if timeout is None else timeout
        with self._lock:
            self._closing = True
            for stop in self._stops.values():
                stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker is not None:
            self._worker.join(timeout)
        for t in list(self._stream_threads.values()):
            t.join(timeout)
        while True:  # jobs that never started
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                self.store.finish_job(
                    item[0], "interrupted", error="server stopped before the job ran"
                )

    # ---- queries
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def running_jobs(self) -> list[str]:
        with self._lock:
            return list(self._stops)

    def list_videos(self, limit: int = 500) -> list[dict]:
        root = self.cfg.input_dir
        exts = {e.lower() for e in self.cfg.video.allowed_extensions}
        out = []
        if not root.is_dir():
            return out
        for p in sorted(root.rglob("*")):
            if len(out) >= limit:
                break
            if p.is_file() and p.suffix.lower() in exts and self._within_input(p):
                st = p.stat()
                out.append(
                    {
                        "path": p.relative_to(root).as_posix(),
                        "size_bytes": st.st_size,
                        "modified": st.st_mtime,
                    }
                )
        return out

    # ---- submission
    def _within_input(self, p: Path) -> bool:
        try:
            p.resolve().relative_to(self.cfg.input_dir.resolve())
            return True
        except ValueError:
            return False

    def resolve_video(self, rel: str) -> Path:
        if not isinstance(rel, str) or not rel.strip():
            raise InvalidInput("'video' must be a path relative to the input directory")
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise InvalidInput("'video' must be a relative path inside the input directory")
        p = self.cfg.input_dir / rel_path
        if not self._within_input(p):
            raise InvalidInput("'video' resolves outside the input directory")
        if p.suffix.lower() not in {e.lower() for e in self.cfg.video.allowed_extensions}:
            raise InvalidInput(
                f"unsupported file type {p.suffix!r}; allowed: "
                f"{', '.join(self.cfg.video.allowed_extensions)}"
            )
        if not p.is_file():
            raise NotFound(f"video not found in input directory: {rel}")
        return p

    def _check_accepting(self) -> None:
        if self._closing:
            raise ShuttingDown("server is shutting down")
        st = self.detector.status()
        if st["state"] in ("missing", "error"):
            raise ModelUnavailable(st["error"] or f"checkpoint not found: {st['weights']}")

    def submit_video(self, rel: str) -> dict:
        self._check_accepting()
        path = self.resolve_video(rel)
        if self._queue.full():
            raise QueueFull(f"{self.cfg.jobs.max_queued} jobs already waiting; try again later")
        try:
            source = self.video_source_factory(path)
        except SourceError as exc:
            raise InvalidInput(f"{rel}: {exc}") from exc
        job_id = uuid.uuid4().hex[:16]
        params = {
            "sample_fps": self.cfg.video.sample_fps,
            "events": vars(self.cfg.events).copy(),
            "conf_threshold": self.cfg.model.conf_threshold,
            "imgsz": self.cfg.model.imgsz,
        }
        job = self.store.create_job(
            job_id,
            "video",
            str(path),
            path.relative_to(self.cfg.input_dir).as_posix(),
            params,
            dict(source.meta),
            source.expected_frames,
        )
        try:
            self._queue.put_nowait((job_id, source))
        except queue.Full:
            self.store.finish_job(job_id, "failed", error="queue full")
            raise QueueFull(
                f"{self.cfg.jobs.max_queued} jobs already waiting; try again later"
            ) from None
        return job

    def submit_stream(
        self, url: str, max_duration_s: float | None = None, name: str | None = None
    ) -> dict:
        if not self.cfg.stream.enabled or self.cfg.stream.max_concurrent == 0:
            raise InvalidInput("stream analysis is disabled in the configuration")
        self._check_accepting()
        if not isinstance(url, str) or not url:
            raise InvalidInput("'url' is required")
        scheme = urlparse(url).scheme.lower()
        if scheme not in self.cfg.stream.allowed_schemes:
            raise InvalidInput(
                f"URL scheme {scheme!r} not allowed; allowed: "
                f"{', '.join(self.cfg.stream.allowed_schemes)}"
            )
        if max_duration_s is None:
            max_duration_s = self.cfg.stream.default_max_duration_s
        if not isinstance(max_duration_s, int | float) or max_duration_s < 0:
            raise InvalidInput("'max_duration_s' must be a number >= 0 (0 = until cancelled)")
        with self._lock:
            live = [j for j, t in self._stream_threads.items() if t.is_alive()]
            if len(live) >= self.cfg.stream.max_concurrent:
                raise QueueFull(
                    f"{len(live)} stream(s) already running (limit "
                    f"{self.cfg.stream.max_concurrent})"
                )
        source = self.stream_source_factory(url, float(max_duration_s))
        job_id = uuid.uuid4().hex[:16]
        parsed = urlparse(url)
        display = (
            name or f"{parsed.scheme}://{parsed.hostname or ''}{parsed.path}"
        )  # no credentials
        params = {
            "sample_fps": self.cfg.stream.sample_fps,
            "max_duration_s": max_duration_s,
            "events": vars(self.cfg.events).copy(),
            "conf_threshold": self.cfg.model.conf_threshold,
        }
        job = self.store.create_job(
            job_id, "stream", display, display, params, {"source": display}, None
        )
        t = threading.Thread(
            target=self._run, args=(job_id, source), name=f"stream-{job_id}", daemon=True
        )
        with self._lock:
            self._stream_threads[job_id] = t
        t.start()
        return job

    def cancel(self, job_id: str) -> dict:
        job = self.store.get_job(job_id)
        if job is None:
            raise NotFound(f"job {job_id} not found")
        with self._lock:
            stop = self._stops.get(job_id)
            if stop is not None:
                self._cancelled.add(job_id)
                stop.set()
                return self.store.get_job(job_id)
            if job["status"] == "queued":
                self._cancelled.add(job_id)
                self.store.finish_job(job_id, "cancelled", error="cancelled before start")
                return self.store.get_job(job_id)
        raise Conflict(f"job {job_id} is already {job['status']}")

    # ---- execution
    def _work_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            job_id, source = item
            with self._lock:
                skip = job_id in self._cancelled
            if skip:
                continue
            if self._closing:
                self.store.finish_job(
                    job_id, "interrupted", error="server stopped before the job ran"
                )
                continue
            self._run(job_id, source)

    def _run(self, job_id: str, source) -> None:
        stop = threading.Event()
        with self._lock:
            if self._closing:
                self.store.finish_job(
                    job_id, "interrupted", error="server stopped before the job ran"
                )
                return
            self._stops[job_id] = stop
        job_dir = self.cfg.jobs_dir / job_id
        status, error, summary = "failed", None, None
        try:
            self.detector.load()
            self.store.mark_running(job_id, self.detector.info())
            summary = run_analysis(
                job_id,
                source,
                self.detector,
                self.cfg,
                self.store,
                job_dir,
                stop,
                annotate=self.annotate,
            )
            if stop.is_set():
                status = "cancelled" if job_id in self._cancelled else "interrupted"
                error = (
                    "cancelled by user"
                    if status == "cancelled"
                    else "server shutdown during analysis"
                )
            else:
                status = "completed"
        except DetectorUnavailable as exc:
            error = f"model unavailable: {exc}"
        except SourceError as exc:
            error = f"video error: {exc}"
        except Exception as exc:  # keep the worker alive; record the failure
            error = f"{type(exc).__name__}: {exc}"
            log.error("job %s failed:\n%s", job_id, traceback.format_exc())
        finally:
            with self._lock:
                self._stops.pop(job_id, None)
            self.store.finish_job(
                job_id,
                status,
                error=error,
                warning=getattr(source, "warning", None),
                summary=summary,
            )
