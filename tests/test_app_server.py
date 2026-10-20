"""HTTP API: health, submission, results, errors, auth, upload limits, restart persistence."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from app_fakes import FakeDetector, FakeSource, fire_schedule, make_config, wait_for
from firewatch.app.server import build_server, shutdown


class Client:
    def __init__(self, server, token=None):
        host, port = server.server_address[:2]
        self.base = f"http://{host}:{port}"
        self.token = token

    def req(self, method, path, body=None, headers=None, raw=None):
        h = dict(headers or {})
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        data = raw
        if body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        r = urllib.request.Request(self.base + path, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                payload = resp.read()
                ctype = resp.headers.get("Content-Type", "")
                return resp.status, json.loads(payload) if "json" in ctype else payload
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")


@pytest.fixture
def running(tmp_path):
    started = []

    def _start(detector=None, source_factory=None, tmp=tmp_path, **cfg):
        c = make_config(tmp, **cfg)
        c.input_dir.mkdir(parents=True, exist_ok=True)
        (c.input_dir / "clip.mp4").write_bytes(b"x")
        server, app = build_server(
            c,
            detector=detector or FakeDetector(fire_schedule(range(2, 8))),
            video_source_factory=source_factory or (lambda p: FakeSource(n_frames=12)),
            annotate=False,
            warm_model=False,
        )
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        started.append(server)
        return server, app, c

    yield _start
    for s in started:
        shutdown(s)


def test_health_and_static_dashboard(running):
    server, _, _ = running()
    c = Client(server)
    st, h = c.req("GET", "/api/health")
    assert st == 200 and h["status"] == "ok" and h["model"]["state"] == "loaded"
    st, page = c.req("GET", "/")
    assert st == 200 and b"FireWatch" in page
    st, js = c.req("GET", "/static/app.js")
    assert st == 200 and b"/api/jobs" in js


def test_submit_and_retrieve_results(running):
    server, app, _ = running()
    c = Client(server)
    st, vids = c.req("GET", "/api/videos")
    assert st == 200 and [v["path"] for v in vids["videos"]] == ["clip.mp4"]
    st, job = c.req("POST", "/api/jobs", {"video": "clip.mp4"})
    assert st == 201 and job["status"] == "queued"
    assert wait_for(lambda: c.req("GET", f"/api/jobs/{job['id']}")[1]["status"] == "completed")
    st, j = c.req("GET", f"/api/jobs/{job['id']}")
    assert j["summary"]["events"] == 1 and j["summary"]["alerts"] == 1
    st, ev = c.req("GET", f"/api/jobs/{job['id']}/events")
    assert (
        st == 200 and ev["events"][0]["class_name"] == "fire" and ev["events"][0]["start_s"] == 2.0
    )
    st, dets = c.req("GET", f"/api/jobs/{job['id']}/detections?limit=3")
    assert st == 200 and dets["frames_with_detections"] == 6 and len(dets["items"]) == 3
    st, al = c.req("GET", "/api/alerts")
    assert st == 200 and len(al["alerts"]) == 1 and al["alerts"][0]["source_name"] == "clip.mp4"
    st, jobs = c.req("GET", "/api/jobs")
    assert st == 200 and jobs["jobs"][0]["id"] == job["id"]


@pytest.mark.parametrize(
    "method, path, body, status, code",
    [
        ("POST", "/api/jobs", {"video": "../x.mp4"}, 400, "invalid_input"),
        ("POST", "/api/jobs", {"video": "nope.mp4"}, 404, "not_found"),
        ("POST", "/api/jobs", {}, 400, "invalid_input"),
        ("GET", "/api/jobs/0123456789abcdef", None, 404, "not_found"),
        ("GET", "/api/jobs/not-a-job", None, 404, "not_found"),
        ("GET", "/api/nothing", None, 404, "not_found"),
        ("GET", "/api/jobs?limit=0", None, 400, "invalid_input"),
        ("POST", "/api/streams", {"url": "file:///etc/passwd"}, 400, "invalid_input"),
    ],
)
def test_api_errors(running, method, path, body, status, code):
    server, _, _ = running()
    st, r = Client(server).req(method, path, body)
    assert st == status and r["error"]["code"] == code


def test_invalid_json_and_oversized_body(running):
    server, _, _ = running()
    c = Client(server)
    st, r = c.req(
        "POST", "/api/jobs", raw=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert st == 400 and r["error"]["code"] == "invalid_json"
    st, r = c.req(
        "POST", "/api/jobs", raw=b"x" * 70000, headers={"Content-Type": "application/json"}
    )
    assert st == 413


def test_missing_checkpoint_returns_503(running):
    server, _, _ = running(detector=FakeDetector(state="missing"))
    c = Client(server)
    st, h = c.req("GET", "/api/health")
    assert h["status"] == "degraded" and h["model"]["state"] == "missing"
    st, r = c.req("POST", "/api/jobs", {"video": "clip.mp4"})
    assert st == 503 and r["error"]["code"] == "model_unavailable"


def test_token_required_when_configured(running):
    server, _, _ = running(api={"token": "abc123"})
    st, _ = Client(server).req("GET", "/api/health")
    assert st == 200
    st, r = Client(server).req("GET", "/api/jobs")
    assert st == 401 and r["error"]["code"] == "unauthorised"
    st, _ = Client(server, token="wrong").req("GET", "/api/jobs")
    assert st == 401
    st, _ = Client(server, token="abc123").req("GET", "/api/jobs")
    assert st == 200


def test_upload_validation(running):
    server, _, cfg = running(video={"max_upload_mb": 1})
    c = Client(server)
    st, r = c.req("POST", "/api/uploads?filename=a.exe", raw=b"abc")
    assert st == 400
    st, r = c.req("POST", "/api/uploads", raw=b"abc")
    assert st == 400 and "filename" in r["error"]["message"]
    st, r = c.req("POST", "/api/uploads?filename=big.mp4", raw=b"0" * (1024 * 1024 + 1))
    assert st == 413
    st, r = c.req("POST", "/api/uploads?filename=bad.mp4", raw=b"not a video at all")
    assert st == 400 and "not a usable video" in r["error"]["message"]
    assert not any((cfg.input_dir / "uploads").glob("*"))  # nothing left behind


def test_cancel_via_api(running):
    server, _, _ = running(source_factory=lambda p: FakeSource(n_frames=500, delay=0.01))
    c = Client(server)
    _, job = c.req("POST", "/api/jobs", {"video": "clip.mp4"})
    assert wait_for(lambda: c.req("GET", f"/api/jobs/{job['id']}")[1]["status"] == "running")
    st, r = c.req("POST", f"/api/jobs/{job['id']}/cancel")
    assert st == 200
    assert wait_for(lambda: c.req("GET", f"/api/jobs/{job['id']}")[1]["status"] == "cancelled")
    st, r = c.req("POST", f"/api/jobs/{job['id']}/cancel")
    assert st == 409


def test_results_survive_server_restart(running, tmp_path):
    server, _, _ = running()
    c = Client(server)
    _, job = c.req("POST", "/api/jobs", {"video": "clip.mp4"})
    assert wait_for(lambda: c.req("GET", f"/api/jobs/{job['id']}")[1]["status"] == "completed")
    shutdown(server)
    server2, _, _ = running()  # same data directory
    c2 = Client(server2)
    st, j = c2.req("GET", f"/api/jobs/{job['id']}")
    assert st == 200 and j["status"] == "completed"
    assert len(c2.req("GET", f"/api/jobs/{job['id']}/events")[1]["events"]) == 1
    assert len(c2.req("GET", "/api/alerts")[1]["alerts"]) == 1


def test_dashboard_only_calls_existing_routes(running):
    """Every API path used by the dashboard script is answered by the server (no 404 route)."""
    import re as _re

    from firewatch.app.server import STATIC_DIR

    server, _, _ = running()
    c = Client(server)
    _, job = c.req("POST", "/api/jobs", {"video": "clip.mp4"})
    assert wait_for(lambda: c.req("GET", f"/api/jobs/{job['id']}")[1]["status"] == "completed")
    js = (STATIC_DIR / "app.js").read_text()
    paths = set(_re.findall(r"[`\"](/api/[^`\"?]*)", js))
    assert paths, "no API paths found in app.js"
    for p in sorted(paths):
        concrete = p.replace("${id}", job["id"]).replace("${jobId}", job["id"])
        concrete = concrete.replace("${encodeURIComponent(name)}${q}", "x.jpg")
        method = "POST" if concrete.endswith(("/cancel", "/uploads", "/streams")) else "GET"
        if concrete.endswith("/files/"):
            continue
        st, body = c.req(method, concrete, {} if method == "POST" else None)
        err = (body.get("error") or {}) if isinstance(body, dict) else {}
        code = err.get("code") if isinstance(err, dict) else None
        assert not (st == 404 and code == "not_found" and "no route" in err.get("message", "")), p
