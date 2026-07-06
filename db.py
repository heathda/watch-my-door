#!/usr/bin/env python3
"""
SQLite event log for the camera watcher.

Every classification is recorded here from day one -- not just the ones that
trigger an alert -- so we can tune prompts/cooldowns later and have a dataset
ready for the (shelved) threat-scoring work. Cooldown state is derived from
this same table (the most recent alerted event per camera), so there is a
single source of truth rather than a separate JSON file.
"""

import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

# Override with CAM_WATCHER_DB to relocate the event log (also used by tests).
DB_FILE = Path(os.getenv("CAM_WATCHER_DB", Path(__file__).parent / "events.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY,
    ts_iso         TEXT    NOT NULL,   -- human-readable timestamp
    ts_epoch       REAL    NOT NULL,   -- for cooldown math
    camera         TEXT    NOT NULL,
    classification TEXT,               -- the UPPERCASED label (line 1 of output)
    description    TEXT,               -- the model's plain-English description
    raw_response   TEXT,               -- exact model output, before parsing
    latency_ms     INTEGER,            -- how long Ollama took
    image_path     TEXT,
    alerted        INTEGER NOT NULL DEFAULT 0,  -- 1 if we sent a notification
    note           TEXT                -- e.g. "cooldown", "no-match", "error: ..."
);
CREATE INDEX IF NOT EXISTS idx_events_cam_alerted
    ON events (camera, alerted, ts_epoch);
"""


def connect(db_file: Path | None = None) -> sqlite3.Connection:
    """Open the DB, creating the schema on first use and migrating old ones.
    Defaults to DB_FILE, read at call time so it can be overridden in tests."""
    conn = sqlite3.connect(db_file or DB_FILE)
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a DB was first created."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    if "description" not in existing:
        conn.execute("ALTER TABLE events ADD COLUMN description TEXT")
        conn.commit()


def log_event(
    conn: sqlite3.Connection,
    camera: str,
    classification: str | None,
    description: str | None,
    raw_response: str | None,
    latency_ms: int | None,
    image_path: str | None,
    alerted: bool,
    note: str | None = None,
) -> None:
    """Record one classification (or failure) and commit it."""
    now = time.time()
    conn.execute(
        """
        INSERT INTO events
            (ts_iso, ts_epoch, camera, classification, description,
             raw_response, latency_ms, image_path, alerted, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            now,
            camera,
            classification,
            description,
            raw_response,
            latency_ms,
            image_path,
            1 if alerted else 0,
            note,
        ),
    )
    conn.commit()


def last_alert_epoch(conn: sqlite3.Connection, camera: str) -> float:
    """Epoch time of the most recent *alerting* event for a camera, or 0.0."""
    row = conn.execute(
        "SELECT MAX(ts_epoch) FROM events WHERE camera = ? AND alerted = 1",
        (camera,),
    ).fetchone()
    return row[0] or 0.0
