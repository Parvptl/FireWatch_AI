"""SQLite persistence for jobs, events and alerts (survives page reloads and server restarts).

Per-frame detections are written by the pipeline to ``<job_dir>/detections.jsonl``; the database
keeps job metadata, summaries, events and alerts. All access goes through one connection guarded
by a lock (the server is threaded; SQLite runs in WAL mode).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

ACTIVE_STATUSES = ("queued", "running")
FINAL_STATUSES = ("completed", "failed", "cancelled", "interrupted")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    source_name TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    progress REAL,
    frames_done INTEGER NOT NULL DEFAULT 0,
    frames_total INTEGER,
    video_time_s REAL NOT NULL DEFAULT 0,
    error TEXT,
    warning TEXT,
    meta TEXT NOT NULL DEFAULT '{}',
    params TEXT NOT NULL DEFAULT '{}',
    model TEXT NOT NULL DEFAULT '{}',
    summary TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    class_name TEXT NOT NULL,
    status TEXT NOT NULL,
    start_s REAL NOT NULL,
    alert_s REAL,
    end_s REAL,
    data TEXT NOT NULL,
    keyframe TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    event_id TEXT NOT NULL,
    class_name TEXT NOT NULL,
    time_s REAL NOT NULL,
    frame_index INTEGER NOT NULL,
    confidence REAL NOT NULL,
    keyframe TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id, start_s);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
"""

_JSON_COLS = ("meta", "params", "model", "summary")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, args)

    def _rows(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    # ---- jobs
    @staticmethod
    def _job_out(row: dict) -> dict:
        for c in _JSON_COLS:
            row[c] = json.loads(row[c] or "{}")
        return row

    def create_job(
        self,
        job_id: str,
        kind: str,
        source: str,
        source_name: str,
        params: dict,
        meta: dict,
        frames_total: int | None,
    ) -> dict:
        self._exec(
            "INSERT INTO jobs (id, kind, source, source_name, status, created_at, params, meta, "
            "frames_total, progress) VALUES (?,?,?,?,?,?,?,?,?,0)",
            (
                job_id,
                kind,
                source,
                source_name,
                "queued",
                utc_now(),
                json.dumps(params),
                json.dumps(meta),
                frames_total,
            ),
        )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM jobs WHERE id=?", (job_id,))
        return self._job_out(rows[0]) if rows else None

    def list_jobs(self, limit: int = 50, status: str | None = None) -> list[dict]:
        if status:
            rows = self._rows(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            rows = self._rows("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._job_out(r) for r in rows]

    def update_job(self, job_id: str, **fields) -> None:
        if not fields:
            return
        cols, vals = [], []
        for k, v in fields.items():
            cols.append(f"{k}=?")
            vals.append(json.dumps(v) if k in _JSON_COLS else v)
        self._exec(f"UPDATE jobs SET {', '.join(cols)} WHERE id=?", (*vals, job_id))

    def mark_running(self, job_id: str, model: dict) -> None:
        self.update_job(job_id, status="running", started_at=utc_now(), model=model)

    def finish_job(
        self,
        job_id: str,
        status: str,
        error: str | None = None,
        warning: str | None = None,
        summary: dict | None = None,
    ) -> None:
        assert status in FINAL_STATUSES
        fields = {"status": status, "finished_at": utc_now(), "error": error, "warning": warning}
        if summary is not None:
            fields["summary"] = summary
        if status == "completed":
            fields["progress"] = 1.0
        self.update_job(job_id, **fields)

    def recover_after_restart(self) -> int:
        """Jobs left queued/running by a previous process cannot resume: mark them interrupted."""
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET status='interrupted', finished_at=?, "
                "error=COALESCE(error, 'server stopped before the job finished') "
                "WHERE status IN ('queued','running')",
                (utc_now(),),
            )
            return cur.rowcount

    def count_active(self) -> int:
        return self._rows("SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued','running')")[0][
            "n"
        ]

    # ---- events and alerts
    def upsert_event(self, job_id: str, event: dict, keyframe: str | None = None) -> None:
        self._exec(
            "INSERT INTO events (id, job_id, class_name, status, start_s, alert_s, end_s, data, "
            "keyframe, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "status=excluded.status, end_s=excluded.end_s, data=excluded.data, "
            "keyframe=COALESCE(excluded.keyframe, events.keyframe), updated_at=excluded.updated_at",
            (
                event["event_id"],
                job_id,
                event["class_name"],
                event["status"],
                event["start_s"],
                event["alert_s"],
                event["end_s"],
                json.dumps(event),
                keyframe,
                utc_now(),
            ),
        )

    def list_events(self, job_id: str) -> list[dict]:
        out = []
        for r in self._rows("SELECT * FROM events WHERE job_id=? ORDER BY start_s, id", (job_id,)):
            d = json.loads(r["data"])
            d["keyframe"] = r["keyframe"]
            d["job_id"] = job_id
            out.append(d)
        return out

    def add_alert(self, job_id: str, alert: dict, keyframe: str | None) -> None:
        self._exec(
            "INSERT OR IGNORE INTO alerts (id, job_id, event_id, class_name, time_s, frame_index, "
            "confidence, keyframe, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                alert["alert_id"],
                job_id,
                alert["event_id"],
                alert["class_name"],
                alert["time_s"],
                alert["frame_index"],
                alert["confidence"],
                keyframe,
                utc_now(),
            ),
        )

    def list_alerts(self, limit: int = 100, job_id: str | None = None) -> list[dict]:
        sql = (
            "SELECT a.*, j.source_name FROM alerts a JOIN jobs j ON j.id=a.job_id {} "
            "ORDER BY a.created_at DESC LIMIT ?"
        )
        if job_id:
            return self._rows(sql.format("WHERE a.job_id=?"), (job_id, limit))
        return self._rows(sql.format(""), (limit,))
