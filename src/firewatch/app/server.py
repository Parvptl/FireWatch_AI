"""HTTP API and dashboard server (Python standard library; no web framework is installed here).

Endpoints (JSON unless noted; errors are ``{"error": {"code", "message"}}``):

    GET  /api/health                      service, model, queue and storage status (no auth)
    GET  /api/config                      effective configuration (token omitted)
    GET  /api/videos                      videos available in the input directory
    POST /api/uploads?filename=NAME       raw video bytes in the body -> saved under input/uploads/
    POST /api/jobs        {"video": REL}  analyse a video from the input directory
    POST /api/streams     {"url", "max_duration_s"?, "name"?}   analyse a live stream
    GET  /api/jobs?limit=&status=         job list (newest first)
    GET  /api/jobs/{id}                   job status, progress, summary, error
    POST /api/jobs/{id}/cancel            cancel a queued or running job
    GET  /api/jobs/{id}/events            aggregated fire/smoke events
    GET  /api/jobs/{id}/detections?offset=&limit=   per-frame detections (frames with boxes)
    GET  /api/jobs/{id}/files/{name}      key-frame JPEGs and annotated.mp4
    GET  /api/alerts?limit=&job_id=       alert history
    GET  /  and /static/*                 dashboard

If ``api.token`` is set, every /api route except /api/health requires
``Authorization: Bearer <token>`` (or the ``X-API-Token`` header).
"""

from __future__ import annotations

import hmac
import json
import logging
import mimetypes
import re
import shutil
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from firewatch.app.config import AppConfig
from firewatch.app.jobs import InvalidInput, JobError, JobManager, NotFound, safe_filename
from firewatch.app.sources import SourceError, probe_video
from firewatch.app.store import Store

log = logging.getLogger("firewatch.app")
STATIC_DIR = Path(__file__).resolve().parent / "static"
FILE_RE = re.compile(r"^[A-Za-z0-9._-]+\.(jpg|mp4)$")
STARTED = time.time()
DRAIN_LIMIT = 16 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _int_param(q: dict, name: str, default: int, lo: int, hi: int) -> int:
    raw = q.get(name, [None])[0]
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ApiError(400, "invalid_input", f"'{name}' must be an integer") from None
    if not lo <= v <= hi:
        raise ApiError(400, "invalid_input", f"'{name}' must be between {lo} and {hi}")
    return v


class App:
    """Holds the shared components; the request handler class reads it from the server."""

    def __init__(self, cfg: AppConfig, store: Store, manager: JobManager, detector):
        self.cfg, self.store, self.manager, self.detector = cfg, store, manager, detector

    def health(self) -> dict:
        model = self.detector.status()
        data_ok = self.cfg.data_dir.is_dir()
        status = "ok" if model["state"] in ("loaded", "not_loaded") and data_ok else "degraded"
        return {
            "status": status,
            "uptime_s": round(time.time() - STARTED, 1),
            "model": model,
            "queue": {
                "waiting": self.manager.queue_depth(),
                "max_queued": self.cfg.jobs.max_queued,
                "running": self.manager.running_jobs(),
            },
            "storage": {
                "data_dir": str(self.cfg.data_dir),
                "writable": data_ok,
                "input_dir": str(self.cfg.input_dir),
            },
            "stream_enabled": self.cfg.stream.enabled and self.cfg.stream.max_concurrent > 0,
            "auth_required": bool(self.cfg.api.token),
        }

    def job_or_404(self, job_id: str) -> dict:
        job = self.store.get_job(job_id)
        if job is None:
            raise ApiError(404, "not_found", f"job {job_id} not found")
        return job


class Handler(BaseHTTPRequestHandler):
    server_version = "FireWatch/0.1"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # route to logging instead of stderr
        log.info("%s - %s", self.address_string(), fmt % args)

    # ---- helpers
    def _send_json(self, status: int, obj) -> None:
        body = json.dumps(obj, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str) -> None:
        self._send_json(status, {"error": {"code": code, "message": message}})

    def _send_file(self, path: Path, ctype: str | None = None) -> None:
        if not path.is_file():
            raise ApiError(404, "not_found", "file not found")
        ctype = ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with path.open("rb") as fh:
            shutil.copyfileobj(fh, self.wfile)

    def _read_json(self) -> dict:
        length = self._content_length(self.app.cfg.api.max_json_bytes)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            obj = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            raise ApiError(400, "invalid_json", "request body is not valid JSON") from None
        if not isinstance(obj, dict):
            raise ApiError(400, "invalid_json", "request body must be a JSON object")
        return obj

    def _content_length(self, limit: int) -> int:
        raw = self.headers.get("Content-Length")
        if raw is None:
            if self.headers.get("Transfer-Encoding"):
                raise ApiError(411, "length_required", "Content-Length is required")
            return 0
        try:
            n = int(raw)
        except ValueError:
            raise ApiError(400, "bad_request", "invalid Content-Length") from None
        if n < 0:
            raise ApiError(400, "bad_request", "invalid Content-Length")
        if n > limit:
            # Drain a bounded amount so the client can read the 413 instead of a reset
            # connection; larger bodies are cut off by closing the connection.
            remaining = min(n, DRAIN_LIMIT)
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 16, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.close_connection = True
            raise ApiError(413, "too_large", f"request body larger than {limit} bytes")
        return n

    def _authorised(self) -> bool:
        token = self.app.cfg.api.token
        if not token:
            return True
        auth = self.headers.get("Authorization", "")
        given = auth[7:] if auth.startswith("Bearer ") else self.headers.get("X-API-Token", "")
        if not given:  # browsers cannot set headers on <img>/<a>; allow ?token= for file routes
            given = parse_qs(urlparse(self.path).query).get("token", [""])[0]
        return hmac.compare_digest(given.encode(), token.encode())

    # ---- dispatch
    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        url = urlparse(self.path)
        path, q = url.path, parse_qs(url.query)
        try:
            if path.startswith("/api/"):
                if path != "/api/health" and not self._authorised():
                    raise ApiError(401, "unauthorised", "missing or invalid API token")
                self._api(method, path, q)
            elif method == "GET":
                self._static(path)
            else:
                raise ApiError(405, "method_not_allowed", f"{method} not allowed here")
        except (ApiError, JobError) as exc:
            if method == "POST":  # the request body may be unread: do not reuse the connection
                self.close_connection = True
            self._error(exc.status, exc.code, exc.message)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # pragma: no cover - last resort
            log.exception("unhandled error for %s %s", method, path)
            self._error(500, "internal_error", f"{type(exc).__name__}: {exc}")

    def _static(self, path: str) -> None:
        if path in ("/", "/index.html"):
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path.startswith("/static/"):
            name = path[len("/static/") :]
            if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
                raise ApiError(404, "not_found", "not found")
            return self._send_file(STATIC_DIR / name)
        raise ApiError(404, "not_found", "not found")

    def _api(self, method: str, path: str, q: dict) -> None:
        app = self.app
        parts = [p for p in path.split("/") if p][1:]  # drop "api"
        route = (method, *parts[:1])
        if route == ("GET", "health") and len(parts) == 1:
            return self._send_json(200, app.health())
        if route == ("GET", "config") and len(parts) == 1:
            return self._send_json(200, app.cfg.public_dict())
        if route == ("GET", "videos") and len(parts) == 1:
            return self._send_json(
                200, {"input_dir": str(app.cfg.input_dir), "videos": app.manager.list_videos()}
            )
        if route == ("POST", "uploads") and len(parts) == 1:
            return self._upload(q)
        if route == ("POST", "streams") and len(parts) == 1:
            body = self._read_json()
            job = app.manager.submit_stream(
                body.get("url"), body.get("max_duration_s"), body.get("name")
            )
            return self._send_json(201, job)
        if route == ("GET", "alerts") and len(parts) == 1:
            limit = _int_param(q, "limit", 100, 1, 1000)
            return self._send_json(
                200, {"alerts": app.store.list_alerts(limit, q.get("job_id", [None])[0])}
            )
        if parts[:1] == ["jobs"]:
            return self._jobs(method, parts[1:], q)
        raise ApiError(404, "not_found", f"no route for {method} {path}")

    def _jobs(self, method: str, rest: list[str], q: dict) -> None:
        app = self.app
        if not rest:
            if method == "POST":
                body = self._read_json()
                return self._send_json(201, app.manager.submit_video(body.get("video")))
            if method == "GET":
                limit = _int_param(q, "limit", 50, 1, 500)
                status = q.get("status", [None])[0]
                return self._send_json(200, {"jobs": app.store.list_jobs(limit, status)})
        job_id = rest[0]
        if not re.fullmatch(r"[0-9a-f]{16}", job_id):
            raise ApiError(404, "not_found", f"job {job_id} not found")
        if len(rest) == 1 and method == "GET":
            return self._send_json(200, app.job_or_404(job_id))
        if len(rest) == 2 and rest[1] == "cancel" and method == "POST":
            return self._send_json(200, app.manager.cancel(job_id))
        if len(rest) == 2 and rest[1] == "events" and method == "GET":
            app.job_or_404(job_id)
            return self._send_json(200, {"job_id": job_id, "events": app.store.list_events(job_id)})
        if len(rest) == 2 and rest[1] == "detections" and method == "GET":
            app.job_or_404(job_id)
            offset = _int_param(q, "offset", 0, 0, 10**9)
            limit = _int_param(q, "limit", 200, 1, 2000)
            return self._send_json(200, self._detections(job_id, offset, limit))
        if len(rest) == 3 and rest[1] == "files" and method == "GET":
            app.job_or_404(job_id)
            if not FILE_RE.fullmatch(rest[2]):
                raise ApiError(404, "not_found", "file not found")
            return self._send_file(app.cfg.jobs_dir / job_id / rest[2])
        raise ApiError(404, "not_found", "no such job route")

    def _detections(self, job_id: str, offset: int, limit: int) -> dict:
        path = self.app.cfg.jobs_dir / job_id / "detections.jsonl"
        items, total = [], 0
        if path.is_file():
            with path.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    total += 1
                    if offset <= i < offset + limit:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:  # line being written right now
                            total -= 1
        return {
            "job_id": job_id,
            "offset": offset,
            "limit": limit,
            "frames_with_detections": total,
            "items": items,
        }

    def _upload(self, q: dict) -> None:
        cfg = self.app.cfg
        name = q.get("filename", [""])[0]
        if not name:
            raise ApiError(400, "invalid_input", "query parameter 'filename' is required")
        fname = safe_filename(name)
        if Path(fname).suffix.lower() not in {e.lower() for e in cfg.video.allowed_extensions}:
            raise ApiError(
                400,
                "invalid_input",
                f"unsupported file type; allowed: {', '.join(cfg.video.allowed_extensions)}",
            )
        limit = cfg.video.max_upload_mb * 1024 * 1024
        length = self._content_length(limit)
        if length == 0:
            raise ApiError(400, "invalid_input", "empty upload")
        dest_dir = cfg.input_dir / "uploads"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{uuid.uuid4().hex[:8]}_{fname}"
        tmp = dest.with_name(dest.name + ".part")
        remaining = length
        try:
            with tmp.open("wb") as fh:
                while remaining:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        raise ApiError(400, "invalid_input", "upload ended early")
                    fh.write(chunk)
                    remaining -= len(chunk)
            try:
                meta = probe_video(tmp)
            except SourceError as exc:
                raise InvalidInput(f"uploaded file is not a usable video: {exc}") from exc
            tmp.replace(dest)
        finally:
            if tmp.exists():
                tmp.unlink()
        rel = dest.relative_to(cfg.input_dir).as_posix()
        if q.get("submit", ["0"])[0] in ("1", "true"):
            try:
                job = self.app.manager.submit_video(rel)
            except NotFound:  # pragma: no cover - file was just written
                raise
            return self._send_json(201, {"video": rel, "meta": meta, "job": job})
        return self._send_json(201, {"video": rel, "meta": meta})


class FireWatchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app: App):
        super().__init__(address, Handler)
        self.app = app


def build_server(
    cfg: AppConfig,
    detector=None,
    video_source_factory=None,
    stream_source_factory=None,
    annotate: bool = True,
    warm_model: bool = True,
):
    """Create store, detector, job manager and HTTP server (not yet serving)."""
    from firewatch.app.detector import YoloDetector

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(cfg.db_path)
    if detector is None:
        m = cfg.model
        detector = YoloDetector(
            cfg.weights_path,
            m.imgsz,
            m.device,
            m.class_names,
            m.conf_threshold,
            m.iou_threshold,
            m.max_det,
            m.cpu_threads,
        )
    manager = JobManager(
        cfg, store, detector, video_source_factory, stream_source_factory, annotate
    )
    manager.start(warm_model=warm_model)
    app = App(cfg, store, manager, detector)
    server = FireWatchServer((cfg.api.host, cfg.api.port), app)
    return server, app


def shutdown(server: FireWatchServer) -> None:
    server.shutdown()
    server.app.manager.shutdown()
    server.app.store.close()
    server.server_close()
