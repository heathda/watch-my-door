#!/usr/bin/env python3
"""
Pipeline self-watchdog -- notices when the alert pipeline goes dark.

The whole system is silent by design: no news is supposed to mean nothing is
happening. That makes total failure indistinguishable from a quiet afternoon,
which is exactly what happened twice in July 2026 -- 2026-07-05 lost 40% of a
day and 2026-07-25 lost 7 straight hours (925 frames) to an unreachable Ollama,
and nothing said a word. This script is the thing that says the word.

RUN IT ON A SCHEDULE (Task Scheduler, every 5 min), never from the alert path.
It reads events.db mode=ro and never touches cam_watcher's flow -- a broken
watchdog must not be able to break alerting, which is the whole point of it.

THE FOUR CHECKS, and why they are not redundant:

  stale   -- no events at all: BlueIris stopped, the box rebooted, the cameras
             dropped. Threshold is time-of-day aware, learned from the event
             log itself (see stale_threshold_min) because "20 minutes of
             silence" means nothing at 3am and means everything at noon.
  errors  -- events ARE arriving but failing to classify. THIS is the check
             that catches the real outages: on 2026-07-25 BlueIris fired all
             day and cam_watcher dutifully logged 925 rows, every one of them
             "error: ollama call failed". A silence check would have seen a
             perfectly healthy event rate and said nothing.
  ollama  -- reach the model host directly, and confirm the configured model is
             still loaded. Catches the failure before frames are lost, and
             catches "someone renamed/deleted the cam-watcher model", which
             looks identical to a network outage from the log's point of view.
  disk    -- events.db and the BlueIris alert folder share a volume with the
             video store. A full disk kills recording and alerting together.

Notifications fire on STATE CHANGE, not on every run: going bad, getting worse,
and recovering. A problem that persists re-notifies at most every
`renotify_min`. Without that, a 7-hour outage at a 5-minute cadence is 84
identical emails and you learn to ignore them.

Usage:
    python watchdog.py                 # one pass; emails only on a state change
    python watchdog.py --dry-run       # print the health report, never email
    python watchdog.py --force         # email the report regardless of state
    python watchdog.py --check errors  # run one check (repeatable)

Config: the `_watchdog` block in cameras.yaml. No block = disabled.
"""

import argparse
import json
import logging
import os
import shutil
import sqlite3
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

import risk
from notify import send_email

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

CONFIG_FILE = SCRIPT_DIR / "cameras.yaml"
DEFAULT_DB = Path(os.getenv("CAM_WATCHER_DB", SCRIPT_DIR / "events.db"))
# Watchdog state is a side file on purpose. Everything else in this project
# derives its state from events.db (see db.last_alert_epoch), but "did I
# already email about this outage" is not a fact about camera events, and the
# alert path must never block on a write from a sidecar process.
DEFAULT_STATE = SCRIPT_DIR / "watchdog_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(SCRIPT_DIR / "watchdog.log")],
)
logger = logging.getLogger(__name__)

DEFAULTS = {
    "stale": {"floor_min": 45, "night_floor_min": 180, "multiplier": 2.5,
              "ceiling_min": 240, "history_days": 14, "min_samples": 30},
    "errors": {"window_min": 30, "min_events": 5, "max_fail_ratio": 0.5},
    "ollama": {"timeout_sec": 5, "check_model": True},
    "disk": {"min_free_gb": 5},
    "notify": {"renotify_min": 180, "recovery": True, "confirm_passes": 2},
}


class Check:
    """One health signal. `ok=None` means "couldn't determine" -- treated as
    healthy for alerting (a watchdog that cries wolf gets ignored) but shown."""

    def __init__(self, name, ok, detail):
        self.name, self.ok, self.detail = name, ok, detail

    @property
    def failing(self) -> bool:
        return self.ok is False

    def __repr__(self):
        mark = {True: "OK  ", False: "FAIL", None: "??  "}[self.ok]
        return f"{mark} {self.name:8} {self.detail}"


def load_config(path: Path = CONFIG_FILE) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.error(f"Could not read {path}: {e}")
        return {}


def settings(config: dict, section: str) -> dict:
    """Defaults overlaid with the `_watchdog` block, so a partial config block
    only overrides what it mentions."""
    cfg = (config.get("_watchdog") or {}).get(section) or {}
    return {**DEFAULTS[section], **cfg}


def open_ro(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def stale_threshold_min(conn, now: float, config: dict) -> tuple[float, str]:
    """How long silence must last, at THIS hour, before it means something.

    Learned from history rather than guessed: event rate swings ~40x between
    3am and noon (p99 gap 59 min vs 2.7 min in the real log), so one fixed
    threshold either screams all night or sleeps through a midday outage. We
    take the 99th-percentile gap for this hour of day, scale it, and clamp.

    The floor is raised inside the existing `_risk.night` window because at
    3am a dead pipeline and a quiet house look identical, and slow detection
    is the correct trade there. Calibrated against 28 days: one alarm, and it
    was the genuine 14-hour dark period on 2026-06-28.
    """
    s = settings(config, "stale")
    risk_cfg = risk.risk_config(config) or {}
    is_night = risk.night_multiplier(now, risk_cfg) > 1.0
    floor = s["night_floor_min"] if is_night else s["floor_min"]

    hour = datetime.fromtimestamp(now).hour
    since = now - s["history_days"] * 86400
    rows = [r[0] for r in conn.execute(
        "SELECT ts_epoch FROM events WHERE ts_epoch >= ? ORDER BY ts_epoch", (since,))]
    # Gaps longer than the ceiling are dropped, not averaged in. Such a gap is
    # a past outage (or the overnight lull straddling this hour), and letting
    # one into the history would raise tomorrow's threshold -- a watchdog that
    # goes deafer every time it fails is worse than none.
    gaps = sorted(
        gap for gap in (
            (b - a) / 60 for a, b in zip(rows, rows[1:])
            if datetime.fromtimestamp(a).hour == hour
        ) if gap <= s["ceiling_min"]
    )
    if len(gaps) < s["min_samples"]:
        return floor, f"floor {floor:.0f}m (not enough history for hour {hour:02d})"
    p99 = gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))]
    learned = min(s["ceiling_min"], s["multiplier"] * p99)
    basis = "night floor" if is_night else "day floor"
    if learned > floor:
        return learned, f"{learned:.0f}m (hour {hour:02d} p99 gap {p99:.0f}m x{s['multiplier']})"
    return floor, f"{floor:.0f}m ({basis}; hour {hour:02d} p99 gap {p99:.0f}m)"


def check_stale(conn, now: float, config: dict) -> Check:
    # Bounded above by `now` so the check is replayable: pointing it at a past
    # moment must see only what was known then. In production now == the
    # present, so this costs nothing; it also ignores future-dated rows from a
    # skewed clock, the same stance risk.score_events takes.
    row = conn.execute("SELECT MAX(ts_epoch) FROM events WHERE ts_epoch <= ?", (now,)).fetchone()
    last = row[0] if row else None
    if not last:
        return Check("stale", False, "no events in the database at all")
    quiet_min = (now - last) / 60
    threshold, why = stale_threshold_min(conn, now, config)
    seen = datetime.fromtimestamp(last).strftime("%Y-%m-%d %H:%M")
    detail = f"last event {seen} ({quiet_min:.0f}m ago), threshold {why}"
    return Check("stale", quiet_min <= threshold, detail)


def check_errors(conn, now: float, config: dict) -> Check:
    """Events arriving but not classifying -- the 2026-07-25 failure mode."""
    s = settings(config, "errors")
    window = s["window_min"] * 60
    total, failed = conn.execute(
        "SELECT COUNT(*), SUM(classification IS NULL) FROM events"
        " WHERE ts_epoch >= ? AND ts_epoch <= ?",
        (now - window, now),
    ).fetchone()
    failed = failed or 0
    if total < s["min_events"]:
        return Check("errors", None,
                     f"{total} events in {s['window_min']}m -- too few to judge")
    ratio = failed / total
    detail = f"{failed}/{total} failed to classify in {s['window_min']}m ({ratio:.0%})"
    if ratio > s["max_fail_ratio"]:
        note = conn.execute(
            "SELECT note FROM events WHERE classification IS NULL"
            " AND ts_epoch >= ? AND ts_epoch <= ?"
            " ORDER BY ts_epoch DESC LIMIT 1", (now - window, now)).fetchone()
        detail += f" -- last: {note[0] if note else 'unknown'}"
    return Check("errors", ratio <= s["max_fail_ratio"], detail)


def check_ollama(config: dict) -> Check:
    """Probe the model host directly, so we hear about it before frames die."""
    s = settings(config, "ollama")
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    model = os.getenv("OLLAMA_MODEL", "cam-watcher")
    base = url.split("/api/")[0]
    try:
        resp = requests.get(f"{base}/api/tags", timeout=s["timeout_sec"])
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
    except requests.RequestException as e:
        return Check("ollama", False, f"unreachable at {base}: {e}")
    except ValueError as e:
        return Check("ollama", False, f"bad response from {base}: {e}")
    if s["check_model"] and not any(n.split(":")[0] == model.split(":")[0] for n in names):
        return Check("ollama", False,
                     f"reachable, but model '{model}' is not loaded (have: {', '.join(names) or 'none'})")
    return Check("ollama", True, f"reachable at {base}, model '{model}' present")


def check_disk(config: dict, db_path: Path) -> Check:
    s = settings(config, "disk")
    # Group by volume, don't key by it: events.db and the alert folder usually
    # share C:, and keying by anchor silently dropped one from the report --
    # making it look unchecked when it was merely the same disk.
    targets: dict[str, list[str]] = {}
    volume = db_path.resolve().anchor or str(db_path.parent)
    targets.setdefault(volume, []).append("events.db")
    alert_dir = os.getenv("ALERT_IMAGE_DIR", "")
    if alert_dir and Path(alert_dir).is_dir():
        volume = Path(alert_dir).resolve().anchor or alert_dir
        targets.setdefault(volume, []).append("alert images")
    parts, ok = [], True
    for path, what in targets.items():
        label = " + ".join(what)
        try:
            free_gb = shutil.disk_usage(path).free / 1024**3
        except OSError as e:
            parts.append(f"{label}: unreadable ({e})")
            continue
        parts.append(f"{label} on {path}: {free_gb:.1f} GB free")
        if free_gb < s["min_free_gb"]:
            ok = False
    if not parts:
        return Check("disk", None, "no volumes to check")
    return Check("disk", ok, "; ".join(parts))


def current_risk(conn, now: float, config: dict) -> str:
    """The score-poller half of this job: what the risk level is right now.
    Free to compute (the score is a pure function of the log) and it makes the
    heartbeat email tell you something even when everything is healthy."""
    if not risk.risk_config(config):
        return "risk scoring off"
    window = float((risk.risk_config(config) or {}).get("window_sec", risk.DEFAULT_WINDOW_SEC))
    events = conn.execute(
        "SELECT camera, classification, ts_epoch FROM events WHERE ts_epoch >= ?",
        (now - window,)).fetchall()
    score = risk.score_events(events, now, config)
    return f"risk {score:.1f} ({risk.tier_for(score, config)}), {len(events)} events in window"


def gather(conn, now: float, config: dict, db_path: Path, only=None) -> list[Check]:
    runners = {
        "stale": lambda: check_stale(conn, now, config),
        "errors": lambda: check_errors(conn, now, config),
        "ollama": lambda: check_ollama(config),
        "disk": lambda: check_disk(config, db_path),
    }
    checks = []
    for name, run in runners.items():
        if only and name not in only:
            continue
        try:
            checks.append(run())
        except Exception as e:  # a broken check must not take the watchdog down
            logger.exception(f"check {name} raised")
            checks.append(Check(name, None, f"check itself failed: {e}"))
    return checks


# --------------------------------------------------------------------------
# state + notification
# --------------------------------------------------------------------------

def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {"failing": [], "since": None, "last_notified": None}


def save_state(path: Path, state: dict) -> None:
    try:
        path.write_text(json.dumps(state, indent=2))
    except OSError as e:
        logger.error(f"could not write state {path}: {e}")


def advance(state: dict, failing: list[str], now: float, config: dict):
    """Fold one pass into the notification state machine.

    Returns (new_state, should_notify, kind) with kind in
    {"problem", "worse", "reminder", "recovery"} or None.

    A change must hold for `confirm_passes` consecutive runs before it is
    announced. Measured over 28 days of real history: without this, a single
    55-minute morning lull produces a "problem" email and a "recovery" email
    five minutes apart, and an intermittent outage flaps -- 4 of 24 emails were
    pure noise of that shape. Ten minutes of confirmation costs nothing at
    these timescales and removes all of it.
    """
    s = settings(config, "notify")
    state = dict(state)
    announced = set(state.get("failing") or [])
    current = set(failing)

    if current == announced:
        state["pending"], state["pending_count"] = None, 0
        if current:
            last = state.get("last_notified") or 0
            if now - last >= s["renotify_min"] * 60:
                state["last_notified"] = now
                return state, True, "reminder"
        return state, False, None

    # A change: it has to survive a few passes to count as real.
    if state.get("pending") == failing:
        state["pending_count"] = state.get("pending_count", 0) + 1
    else:
        state["pending"], state["pending_count"] = failing, 1
        state["pending_since"] = now
    if state["pending_count"] < s["confirm_passes"]:
        return state, False, None

    kind = "recovery" if not current else ("problem" if not announced else "worse")
    if not announced:
        state["since"] = state.get("pending_since", now)
    if not current:
        state["since"] = None
    state["failing"] = failing
    state["pending"], state["pending_count"] = None, 0
    if kind == "recovery" and not s["recovery"]:
        return state, False, kind
    state["last_notified"] = now
    return state, True, kind


def format_report(checks: list[Check], risk_line: str, now: float) -> str:
    stamp = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"watch-my-door pipeline health at {stamp}", ""]
    lines += [f"  {c!r}" for c in checks]
    lines += ["", f"  {risk_line}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--db", default=None, help="event log (default: events.db)")
    ap.add_argument("--config", default=str(CONFIG_FILE), help="cameras.yaml to read _watchdog from")
    ap.add_argument("--state", default=str(DEFAULT_STATE), help="notification state file")
    ap.add_argument("--check", action="append", dest="only",
                    choices=["stale", "errors", "ollama", "disk"],
                    help="run only this check (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="print the report, never email")
    ap.add_argument("--force", action="store_true", help="email the report regardless of state")
    args = ap.parse_args()

    config = load_config(Path(args.config))
    if not (config.get("_watchdog") or {}).get("enabled", False) and not args.dry_run:
        logger.info("_watchdog not enabled in config -- nothing to do")
        return

    db_path = Path(args.db) if args.db else DEFAULT_DB
    now = time.time()
    try:
        conn = open_ro(db_path)
    except sqlite3.Error as e:
        # The event log itself being unopenable is the loudest possible symptom.
        checks, risk_line = [Check("stale", False, f"cannot open {db_path}: {e}")], "unknown"
    else:
        checks = gather(conn, now, config, db_path, args.only)
        risk_line = current_risk(conn, now, config)

    report = format_report(checks, risk_line, now)
    print(report)

    failing = sorted(c.name for c in checks if c.failing)
    state, should, kind = advance(load_state(Path(args.state)), failing, now, config)
    if args.force:
        should, kind = True, kind or "heartbeat"

    if should and not args.dry_run:
        subject = {
            "problem": f"PIPELINE PROBLEM: {', '.join(failing)}",
            "worse": f"PIPELINE PROBLEM (changed): {', '.join(failing)}",
            "reminder": f"PIPELINE STILL DOWN: {', '.join(failing)}",
            "recovery": "pipeline recovered",
            "heartbeat": "pipeline health report",
        }[kind]
        body = report
        if kind == "recovery" and state.get("since"):
            down = (now - state["since"]) / 60
            body += f"\n\nWas failing for {down:.0f} minutes ({', '.join(state['failing'])})."
        sent = send_email(subject=f"[watch-my-door] {subject}", body=body)
        logger.info(f"{kind}: {'emailed' if sent else 'EMAIL FAILED'} -- {failing or 'all clear'}")
        if not sent:
            # Don't let a failed send count as "already told them" -- clearing
            # the stamp makes the next pass try again instead of waiting out
            # renotify_min. Email may well be down for the same reason the
            # pipeline is.
            state["last_notified"] = None
    elif failing:
        logger.info(f"failing ({', '.join(failing)}) -- no notification this run "
                    f"({kind or 'unconfirmed, pass %s' % state.get('pending_count', 0)})")

    if not args.dry_run:
        save_state(Path(args.state), state)

    sys.exit(1 if failing else 0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        # Same contract as cam_watcher: never crash noisily on a scheduled run.
        logger.exception("watchdog failed")
