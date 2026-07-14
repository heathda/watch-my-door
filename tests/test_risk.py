"""risk: decay math, night window, tier boundaries, config-off behavior, and
the shadow-mode audit columns landing in the database."""
from datetime import datetime

import db
import risk

CONFIG = {
    "_risk": {
        "half_life_sec": 600,
        "window_sec": 7200,
        "night": {"start": "22:00", "end": "06:00", "multiplier": 3.0},
        "tiers": {"QUIET": 0, "NOTICE": 20, "ELEVATED": 50, "URGENT": 80},
    },
    "cam": {
        "prompt": "p",
        "alert_on": ["PERSON"],
        "risk": {
            "multiplier": 2.0,
            "label_weights": {"PERSON": 10, "IGNORED": 0},
        },
    },
}

NOON = datetime(2026, 1, 1, 12, 0).timestamp()      # daytime, local
MIDNIGHT = datetime(2026, 1, 1, 0, 30).timestamp()  # inside the wrapped night window


def test_delta_is_weight_times_multiplier():
    assert risk.event_delta("cam", "PERSON", NOON, CONFIG) == 20.0


def test_delta_night_multiplier_wraps_midnight():
    assert risk.event_delta("cam", "PERSON", MIDNIGHT, CONFIG) == 60.0
    late = datetime(2026, 1, 1, 22, 0).timestamp()   # boundary: night starts
    assert risk.event_delta("cam", "PERSON", late, CONFIG) == 60.0
    morning = datetime(2026, 1, 1, 6, 0).timestamp()  # boundary: night over
    assert risk.event_delta("cam", "PERSON", morning, CONFIG) == 20.0


def test_unknown_label_and_zero_weight_score_nothing():
    assert risk.event_delta("cam", "MARTIAN", NOON, CONFIG) == 0.0
    assert risk.event_delta("cam", "IGNORED", NOON, CONFIG) == 0.0
    assert risk.event_delta("othercam", "PERSON", NOON, CONFIG) == 0.0
    assert risk.event_delta("cam", None, NOON, CONFIG) == 0.0


def test_score_decays_by_half_life():
    events = [("cam", "PERSON", NOON)]
    assert risk.score_events(events, NOON, CONFIG) == 20.0
    assert risk.score_events(events, NOON + 600, CONFIG) == 10.0   # one half-life
    assert risk.score_events(events, NOON + 1200, CONFIG) == 5.0   # two


def test_score_sums_and_ignores_outside_window():
    events = [
        ("cam", "PERSON", NOON - 8000),      # older than window: ignored
        ("cam", "PERSON", NOON + 60),        # from the future: ignored
        ("other", "PERSON", NOON - 600),     # distinct signal, decayed to half
        ("cam", "PERSON", NOON),             # full weight
    ]
    config = dict(CONFIG, other=CONFIG["cam"])
    assert risk.score_events(events, NOON, config) == 30.0


def test_repeat_dampening_same_signal_does_not_stack():
    config = {**CONFIG, "_risk": {**CONFIG["_risk"], "repeat_dampening": 0.5}}
    # Three frames of the same ongoing condition, close together.
    events = [("cam", "PERSON", NOON - 2), ("cam", "PERSON", NOON - 1), ("cam", "PERSON", NOON)]
    score = risk.score_events(events, NOON, config)
    # ~20 (newest, full) + ~10 (x0.5) + ~5 (x0.25), bar negligible time decay.
    assert 34.9 < score < 35.0

    # Distinct signals are NOT dampened -- three cameras stack in full.
    spread = {**config, "b": config["cam"], "c": config["cam"]}
    events = [("cam", "PERSON", NOON), ("b", "PERSON", NOON), ("c", "PERSON", NOON)]
    assert risk.score_events(events, NOON, spread) == 60.0


def test_tier_boundaries():
    for score, tier in ((0, "QUIET"), (19.9, "QUIET"), (20, "NOTICE"),
                        (50, "ELEVATED"), (80, "URGENT"), (500, "URGENT")):
        assert risk.tier_for(score, CONFIG) == tier, score


def test_no_risk_config_is_a_no_op():
    bare = {"cam": {"prompt": "p", "alert_on": ["PERSON"]}}
    assert risk.risk_config(bare) is None
    assert risk.event_delta("cam", "PERSON", NOON, bare) == 0.0
    assert risk.score_events([("cam", "PERSON", NOON)], NOON, bare) == 0.0
    assert risk.tier_for(100.0, bare) is None


def test_audit_columns_round_trip(temp_db):
    conn = db.connect()
    db.log_event(conn, "cam", "PERSON", "d", "r", 10, "x", alerted=False,
                 note="no-match", risk_delta=20.0, risk_score=27.5, risk_tier="NOTICE")
    row = conn.execute("SELECT risk_delta, risk_score, risk_tier FROM events").fetchone()
    assert row == (20.0, 27.5, "NOTICE")


def test_recent_events_window(temp_db):
    conn = db.connect()
    db.log_event(conn, "a", "PERSON", "d", "r", 10, "x", alerted=False)
    db.log_event(conn, "b", "NONE", "d", "r", 10, "x", alerted=False)
    rows = db.recent_events(conn, 0)
    assert [(r[0], r[1]) for r in rows] == [("a", "PERSON"), ("b", "NONE")]
    assert db.recent_events(conn, rows[-1][2] + 1) == []


def test_migration_adds_risk_columns(tmp_path, monkeypatch):
    import sqlite3
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, ts_iso TEXT, ts_epoch REAL, "
        "camera TEXT, classification TEXT, description TEXT, raw_response TEXT, "
        "latency_ms INTEGER, image_path TEXT, alerted INTEGER, note TEXT)"
    )
    old.commit()
    old.close()

    monkeypatch.setattr(db, "DB_FILE", path)
    conn = db.connect()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
    assert {"risk_delta", "risk_score", "risk_tier"} <= cols
