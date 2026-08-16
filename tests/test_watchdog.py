"""watchdog: each health check, the learned staleness threshold, and the
notify-on-transition state machine. No network, no email, no real clock."""
import json
import time
from datetime import datetime

import pytest

import db
import watchdog

CONFIG = {
    "_watchdog": {"enabled": True},
    "_risk": {
        "half_life_sec": 600, "window_sec": 7200,
        "night": {"start": "22:00", "end": "06:00", "multiplier": 3.0},
        "tiers": {"QUIET": 0, "NOTICE": 20},
    },
    "cam": {"prompt": "p", "alert_on": ["PERSON"],
            "risk": {"multiplier": 1.0, "label_weights": {"PERSON": 25}}},
}

NOON = datetime(2026, 7, 20, 12, 0).timestamp()
NIGHT = datetime(2026, 7, 20, 3, 0).timestamp()


def seed(conn, when, count, *, label="NONE", spacing=60):
    """`count` events ending at `when`, one every `spacing` seconds."""
    for i in range(count):
        conn.execute(
            "INSERT INTO events (ts_iso, ts_epoch, camera, classification, alerted)"
            " VALUES (?, ?, 'cam', ?, 0)",
            (datetime.fromtimestamp(when - i * spacing).strftime("%Y-%m-%d %H:%M:%S"),
             when - i * spacing, label),
        )
    conn.commit()


@pytest.fixture
def live_db(temp_db):
    conn = db.connect()
    yield conn
    conn.close()


def ro(temp_db):
    return watchdog.open_ro(temp_db)


# --- stale -----------------------------------------------------------------

def test_stale_uses_day_floor_and_fires_on_a_midday_gap(live_db, temp_db):
    seed(live_db, NOON - 90 * 60, 200)          # steady traffic, then silence
    check = watchdog.check_stale(ro(temp_db), NOON, CONFIG)
    assert check.failing and "90m ago" in check.detail


def test_stale_tolerates_the_same_gap_at_night(live_db, temp_db):
    """3am silence is normal; the night floor (180m) must not fire at 90m."""
    seed(live_db, NIGHT - 90 * 60, 200)
    assert watchdog.check_stale(ro(temp_db), NIGHT, CONFIG).ok is True


def test_stale_threshold_is_learned_from_this_hour_of_day(live_db, temp_db):
    # 13 prior days of steady 2-minute traffic during the noon hour only.
    for day in range(1, 14):
        seed(live_db, NOON - day * 86400 + 58 * 60, 30, spacing=120)
    threshold, why = watchdog.stale_threshold_min(ro(temp_db), NOON, CONFIG)
    # p99 gap for hour 12 is ~2 min, so the learned value (x2.5) is far below
    # the floor -- the floor wins, but the history is what proved it.
    assert threshold == 45 and "p99 gap 2m" in why


def test_stale_threshold_rises_where_history_is_sparse(live_db, temp_db):
    """A quiet hour whose normal cadence is ~30 min must NOT be judged by the
    45-minute day floor -- the learned value takes over and the check relaxes."""
    cfg = {**CONFIG, "_watchdog": {"enabled": True, "stale": {"min_samples": 10}}}
    for day in range(1, 14):
        seed(live_db, NOON - day * 86400 + 30 * 60, 2, spacing=30 * 60)  # 12:00, 12:30
    threshold, why = watchdog.stale_threshold_min(ro(temp_db), NOON, cfg)
    assert threshold == pytest.approx(75.0) and "x2.5" in why  # 2.5 x 30 min


def test_learner_ignores_gaps_left_by_past_outages(live_db, temp_db):
    """A previous dark period must not raise tomorrow's threshold. Without the
    ceiling filter, one 20-hour hole teaches the watchdog to sleep through the
    next one."""
    for day in range(1, 14):
        seed(live_db, NOON - day * 86400 + 58 * 60, 30, spacing=120)
    threshold, _ = watchdog.stale_threshold_min(ro(temp_db), NOON, CONFIG)
    assert threshold == 45  # not dragged up by the 23h between daily batches


def test_stale_with_no_events_at_all_is_a_failure(live_db, temp_db):
    assert watchdog.check_stale(ro(temp_db), NOON, CONFIG).failing


# --- errors ----------------------------------------------------------------

def test_errors_catches_the_flowing_but_failing_outage(live_db, temp_db):
    """The 2026-07-25 shape: events arrive fine, every one fails to classify."""
    seed(live_db, NOON, 20, label=None)
    check = watchdog.check_errors(ro(temp_db), NOON, CONFIG)
    assert check.failing and "100%" in check.detail


def test_errors_ignores_a_healthy_trickle_of_failures(live_db, temp_db):
    seed(live_db, NOON, 19)
    seed(live_db, NOON - 5, 1, label=None)
    assert watchdog.check_errors(ro(temp_db), NOON, CONFIG).ok is True


def test_errors_declines_to_judge_a_tiny_sample(live_db, temp_db):
    seed(live_db, NOON, 2, label=None)
    check = watchdog.check_errors(ro(temp_db), NOON, CONFIG)
    assert check.ok is None and not check.failing  # unknown != alarm


# --- ollama / disk ---------------------------------------------------------

def test_ollama_unreachable_fails(monkeypatch):
    import requests
    monkeypatch.setattr(watchdog.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("refused")))
    check = watchdog.check_ollama(CONFIG)
    assert check.failing and "unreachable" in check.detail


def test_ollama_reachable_but_model_missing_fails(monkeypatch):
    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"models": [{"name": "llama3.2:3b"}]}
    monkeypatch.setattr(watchdog.requests, "get", lambda *a, **k: Resp())
    monkeypatch.setenv("OLLAMA_MODEL", "cam-watcher")
    check = watchdog.check_ollama(CONFIG)
    assert check.failing and "not loaded" in check.detail


def test_ollama_healthy(monkeypatch):
    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"models": [{"name": "cam-watcher:latest"}]}
    monkeypatch.setattr(watchdog.requests, "get", lambda *a, **k: Resp())
    monkeypatch.setenv("OLLAMA_MODEL", "cam-watcher")
    assert watchdog.check_ollama(CONFIG).ok is True


def test_disk_reports_both_targets_when_they_share_a_volume(monkeypatch, tmp_path):
    """On the real box events.db and the alert folder are both on C:. Keying by
    volume dropped one from the report, which reads as "not being checked"."""
    monkeypatch.setattr(watchdog.shutil, "disk_usage",
                        lambda p: type("U", (), {"free": 60 * 1024**3})())
    monkeypatch.setenv("ALERT_IMAGE_DIR", str(tmp_path))
    check = watchdog.check_disk(CONFIG, tmp_path / "events.db")
    assert check.ok is True
    assert "events.db + alert images" in check.detail
    assert check.detail.count("GB free") == 1  # one volume, reported once


def test_disk_flags_a_full_volume(monkeypatch, tmp_path):
    monkeypatch.setattr(watchdog.shutil, "disk_usage",
                        lambda p: type("U", (), {"free": 1 * 1024**3})())
    monkeypatch.delenv("ALERT_IMAGE_DIR", raising=False)
    assert watchdog.check_disk(CONFIG, tmp_path / "events.db").failing


# --- notification state machine -------------------------------------------

def run(state, passes, config=CONFIG, start=NOON, step=300):
    """Drive the state machine over consecutive passes, as Task Scheduler does.
    Returns (final_state, [(kind, offset_seconds), ...] of what it emailed)."""
    sent = []
    for i, failing in enumerate(passes):
        state, should, kind = watchdog.advance(state, failing, start + i * step, config)
        if should:
            sent.append((kind, i * step))
    return state, sent


def test_a_problem_must_persist_before_it_is_announced():
    fresh = {"failing": [], "since": None, "last_notified": None}
    # One bad pass, then fine: the blip that produced 4 junk emails in 28 days.
    _, sent = run(fresh, [[], ["stale"], [], []])
    assert sent == []
    # Two consecutive bad passes: real, and announced on the second.
    _, sent = run(fresh, [[], ["stale"], ["stale"], ["stale"]])
    assert sent == [("problem", 600)]


def test_ongoing_problem_stays_quiet_then_reminds():
    state, sent = run({"failing": [], "since": None, "last_notified": None},
                      [["errors"]] * 40)  # 40 passes x 5 min = 200 min
    kinds = [k for k, _ in sent]
    assert kinds == ["problem", "reminder"]      # not 40 emails
    assert sent[1][1] - sent[0][1] >= 180 * 60   # renotify_min respected


def test_a_second_failing_check_is_news():
    state = {"failing": ["errors"], "since": NOON, "last_notified": NOON}
    _, sent = run(state, [["errors", "ollama"]] * 3)
    assert [k for k, _ in sent] == ["worse"]


def test_recovery_is_announced_once_and_must_also_persist():
    state = {"failing": ["ollama"], "since": NOON, "last_notified": NOON}
    # True flapping -- never two clean passes in a row, so never confirmed.
    _, sent = run(state, [[], ["ollama"], [], ["ollama"], []])
    assert sent == []
    final, sent = run(state, [[], [], []])           # steady recovery
    assert [k for k, _ in sent] == ["recovery"]
    assert final["failing"] == [] and final["since"] is None


def test_recovery_can_be_switched_off():
    cfg = {**CONFIG, "_watchdog": {"enabled": True, "notify": {"recovery": False}}}
    state = {"failing": ["ollama"], "since": NOON, "last_notified": NOON}
    final, sent = run(state, [[], [], []], config=cfg)
    assert sent == []
    assert final["failing"] == []  # state still clears, we just don't email


def test_since_records_when_the_problem_started_not_when_confirmed():
    fresh = {"failing": [], "since": None, "last_notified": None}
    final, _ = run(fresh, [["stale"], ["stale"], ["stale"]])
    assert final["since"] == NOON  # first bad pass, not the confirming one


def test_settings_overlay_keeps_unmentioned_defaults():
    cfg = {"_watchdog": {"errors": {"window_min": 5}}}
    s = watchdog.settings(cfg, "errors")
    assert s["window_min"] == 5
    assert s["min_events"] == watchdog.DEFAULTS["errors"]["min_events"]


def test_state_round_trips(tmp_path):
    path = tmp_path / "state.json"
    assert watchdog.load_state(path) == {"failing": [], "since": None, "last_notified": None}
    watchdog.save_state(path, {"failing": ["stale"], "since": 1.0, "last_notified": 2.0})
    assert watchdog.load_state(path)["failing"] == ["stale"]
    path.write_text("{not json")
    assert watchdog.load_state(path)["failing"] == []  # corrupt state must not crash


# --- report ----------------------------------------------------------------

def test_report_shows_every_check_and_the_risk_line():
    checks = [watchdog.Check("stale", True, "last event 1m ago"),
              watchdog.Check("errors", False, "20/20 failed"),
              watchdog.Check("disk", None, "no volumes to check")]
    out = watchdog.format_report(checks, "risk 0.0 (QUIET), 3 events in window", NOON)
    assert "OK   stale" in out and "FAIL errors" in out and "??   disk" in out
    assert "QUIET" in out


def test_gather_survives_a_broken_check(live_db, temp_db, monkeypatch):
    monkeypatch.setattr(watchdog, "check_ollama",
                        lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")))
    checks = watchdog.gather(ro(temp_db), NOON, CONFIG, temp_db)
    broken = [c for c in checks if c.name == "ollama"][0]
    assert broken.ok is None and "check itself failed" in broken.detail
    assert len(checks) == 4  # the other three still ran
