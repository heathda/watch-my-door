"""db: logging, cooldown derivation, per-camera isolation, schema migration."""
import sqlite3
import time

import db


def test_schema_and_log(temp_db):
    conn = db.connect()
    db.log_event(conn, "cam", "OPEN", "door is open", "OPEN\ndoor is open",
                 100, "x.jpg", alerted=True)
    rows = conn.execute(
        "SELECT camera, classification, description, alerted FROM events"
    ).fetchall()
    assert rows == [("cam", "OPEN", "door is open", 1)]


def test_cooldown_only_counts_alerted_events(temp_db):
    conn = db.connect()
    assert db.last_alert_epoch(conn, "cam") == 0.0  # no events yet

    db.log_event(conn, "cam", "CLOSED", "d", "r", 10, "x", alerted=False, note="no-match")
    assert db.last_alert_epoch(conn, "cam") == 0.0  # a no-match must not set cooldown

    db.log_event(conn, "cam", "OPEN", "d", "r", 10, "x", alerted=True)
    assert abs(db.last_alert_epoch(conn, "cam") - time.time()) < 5


def test_camera_isolation(temp_db):
    conn = db.connect()
    db.log_event(conn, "a", "OPEN", "d", "r", 10, "x", alerted=True)
    assert db.last_alert_epoch(conn, "b") == 0.0


def test_migration_adds_description_column(tmp_path, monkeypatch):
    # Build a pre-`description` events table, then let connect() migrate it.
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, ts_iso TEXT, ts_epoch REAL, "
        "camera TEXT, classification TEXT, raw_response TEXT, latency_ms INTEGER, "
        "image_path TEXT, alerted INTEGER, note TEXT)"
    )
    old.commit()
    old.close()

    monkeypatch.setattr(db, "DB_FILE", path)
    conn = db.connect()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
    assert "description" in cols
    # And logging through the new column works post-migration.
    db.log_event(conn, "cam", "OPEN", "desc", "r", 1, "x", alerted=False)
    assert conn.execute("SELECT description FROM events").fetchone() == ("desc",)
