"""Replay's --since window: parsing, and the warm-up that makes it correct."""
import time
from datetime import datetime

import pytest

import db
import replay

CONFIG = {
    "_risk": {
        "half_life_sec": 600,
        "window_sec": 7200,
        "tiers": {"QUIET": 0, "NOTICE": 20, "ELEVATED": 50},
    },
    "cam": {"risk": {"multiplier": 1.0, "label_weights": {"PERSON": 30}}},
}


def _epoch(iso: str) -> float:
    return datetime.strptime(iso, "%Y-%m-%d %H:%M:%S").timestamp()


def test_parse_since_accepts_date_and_datetime():
    assert replay.parse_since("2026-07-26") == _epoch("2026-07-26 00:00:00")
    assert replay.parse_since("2026-07-26 14:30") == _epoch("2026-07-26 14:30:00")
    assert replay.parse_since("2026-07-26 14:30:05") == _epoch("2026-07-26 14:30:05")


def test_parse_since_rejects_garbage():
    with pytest.raises(SystemExit):
        replay.parse_since("last tuesday")


def _seed(conn, rows):
    for ts, camera, label in rows:
        conn.execute(
            "INSERT INTO events (ts_iso, ts_epoch, camera, classification, alerted)"
            " VALUES (?, ?, ?, ?, 0)",
            (ts, _epoch(ts), camera, label),
        )
    conn.commit()


def test_load_events_from_epoch_filters(temp_db):
    conn = db.connect(temp_db)
    _seed(conn, [
        ("2026-07-25 23:00:00", "cam", "PERSON"),
        ("2026-07-26 01:00:00", "cam", "PERSON"),
    ])
    assert len(replay.load_events(conn)) == 2
    assert len(replay.load_events(conn, _epoch("2026-07-26 00:00:00"))) == 1


def test_warmup_preserves_score_across_the_cutoff(temp_db):
    """An event just after --since must be scored against the events before it.
    Loading from the cutoff itself would understate it; loading one window
    earlier reproduces the score a full replay gives."""
    conn = db.connect(temp_db)
    _seed(conn, [
        ("2026-07-25 23:55:00", "cam", "PERSON"),  # 10 min before the cutoff
        ("2026-07-26 00:05:00", "cam", "PERSON"),
    ])
    cutoff = _epoch("2026-07-26 00:00:00")

    full = list(replay.rescore(replay.load_events(conn), CONFIG))
    after = [s for s in full if s[1] >= cutoff]

    warm = replay.load_events(conn, cutoff - 7200)
    warmed = [s for s in replay.rescore(warm, CONFIG) if s[1] >= cutoff]
    assert [s[6] for s in warmed] == pytest.approx([s[6] for s in after])

    # Without warm-up the carried-over history is lost, so the score is lower.
    cold = replay.load_events(conn, cutoff)
    cold_scored = list(replay.rescore(cold, CONFIG))
    assert cold_scored[0][6] < warmed[0][6]
