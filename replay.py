#!/usr/bin/env python3
"""
Replay the event log through the risk scorer -- the tuning tool.

Because every classification is stored in events.db and the score is a pure
function of the event log + config (see risk.py), history can be re-scored
under any weights WITHOUT re-running the vision model. Edit the `_risk` /
per-camera `risk` blocks (or point --config at an experimental copy) and see
what the whole week would have looked like.

Usage:
    python replay.py                     # score events.db with cameras.yaml
    python replay.py --config alt.yaml   # A/B an experimental config
    python replay.py --db other.db       # score a different event log
    python replay.py --top 10            # show the N highest-scoring moments
    python replay.py --tier NOTICE       # list every event at/above a tier
    python replay.py --since 2026-07-26  # only score from a date onward

A prompt change alters which labels get produced at all, so history either side
of one is two different label distributions and averaging them hides both.
--since restricts the *report* to events at/after the cutoff while still
loading one window's worth of earlier events, so the first events after the
cutoff are scored against the same history cam_watcher saw.

Reads only; never writes to the database.
"""

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

import db
import risk


def parse_since(text: str) -> float:
    """'YYYY-MM-DD', 'YYYY-MM-DD HH:MM' or '...:SS' -> local epoch (ts_epoch
    is local-time epoch, so strptime().timestamp() matches it)."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    raise SystemExit(f"--since: cannot parse {text!r} (use 'YYYY-MM-DD[ HH:MM[:SS]]')")


def load_events(conn, from_epoch: float | None = None):
    """Every event, oldest first: (ts_iso, ts_epoch, camera, label, alerted).
    from_epoch loads only events at/after it -- pass the cutoff minus one
    window so the first reported event still has its history to decay."""
    if from_epoch is None:
        return conn.execute(
            "SELECT ts_iso, ts_epoch, camera, classification, alerted"
            " FROM events ORDER BY ts_epoch"
        ).fetchall()
    return conn.execute(
        "SELECT ts_iso, ts_epoch, camera, classification, alerted"
        " FROM events WHERE ts_epoch >= ? ORDER BY ts_epoch",
        (from_epoch,),
    ).fetchall()


def rescore(events, config):
    """Score each event moment exactly as cam_watcher would have: run
    risk.score_events over the window of events up to and including this one.
    Yields (ts_iso, ts_epoch, camera, label, alerted, delta, score, tier)."""
    risk_cfg = risk.risk_config(config) or {}
    window = float(risk_cfg.get("window_sec", risk.DEFAULT_WINDOW_SEC))
    recent: list[tuple[str, str | None, float]] = []
    start = 0
    for ts_iso, ts_epoch, camera, label, alerted in events:
        recent.append((camera, label, ts_epoch))
        while recent[start][2] < ts_epoch - window:
            start += 1
        delta = risk.event_delta(camera, label, ts_epoch, config)
        score = risk.score_events(recent[start:], ts_epoch, config)
        tier = risk.tier_for(score, config)
        yield ts_iso, ts_epoch, camera, label, alerted, delta, score, tier


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", default="cameras.yaml", help="config to score with")
    ap.add_argument("--db", default=None, help="events database (default: events.db)")
    ap.add_argument("--top", type=int, default=5, help="highest-scoring moments to show")
    ap.add_argument("--tier", default=None, help="list every event at/above this tier")
    ap.add_argument(
        "--since", default=None, help="only report events at/after 'YYYY-MM-DD[ HH:MM]'"
    )
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    if not risk.risk_config(config):
        print(f"{args.config} has no _risk block -- nothing to score.")
        return

    since = parse_since(args.since) if args.since else None
    window = float((risk.risk_config(config) or {}).get("window_sec", risk.DEFAULT_WINDOW_SEC))

    conn = db.connect(Path(args.db) if args.db else None)
    # Load one window before the cutoff as warm-up; report only at/after it.
    events = load_events(conn, since - window if since is not None else None)
    scored = [s for s in rescore(events, config) if since is None or s[1] >= since]
    if not scored:
        print("No events in the database." if since is None else f"No events since {args.since}.")
        return

    tiers = (risk.risk_config(config) or {}).get("tiers", {})
    order = [name for name, _ in sorted(tiers.items(), key=lambda kv: kv[1])]

    print(f"{len(scored)} events, {scored[0][0]} .. {scored[-1][0]}\n")

    print("Tier at event time (all events):")
    dist = Counter(s[7] for s in scored)
    for name in order:
        print(f"  {name:10} {dist.get(name, 0):6}")

    print("\nTier at event time (events that actually emailed):")
    dist_alerted = Counter(s[7] for s in scored if s[4])
    for name in order:
        print(f"  {name:10} {dist_alerted.get(name, 0):6}")

    # The mismatch report: what tiering would change about what emailed.
    quiet = order[0] if order else None
    noisy = sum(1 for s in scored if s[4] and s[7] == quiet)
    missed = sum(1 for s in scored if not s[4] and s[7] != quiet)
    print(f"\nEmailed but scored {quiet}: {noisy}  (candidate spam under tiering)")
    print(f"Scored above {quiet} but not emailed: {missed}  (tiering would surface)")

    print(f"\nTop {args.top} highest-scoring moments:")
    for ts_iso, _, camera, label, alerted, delta, score, tier in sorted(
        scored, key=lambda s: -s[6]
    )[: args.top]:
        mark = "*" if alerted else " "
        print(f" {mark} {ts_iso}  {camera:14} {label or '-':18} +{delta:6.1f} -> {score:6.1f}  {tier}")

    if args.tier:
        want = args.tier.upper()
        if want not in tiers:
            print(f"\nUnknown tier {want!r}; configured: {', '.join(order)}")
            return
        floor = tiers[want]
        hits = [s for s in scored if s[6] >= floor]
        print(f"\nEvents at/above {want} ({len(hits)}):")
        for ts_iso, _, camera, label, alerted, delta, score, tier in hits:
            mark = "*" if alerted else " "
            print(f" {mark} {ts_iso}  {camera:14} {label or '-':18} +{delta:6.1f} -> {score:6.1f}  {tier}")

    print("\n(* = actually emailed at the time)")


if __name__ == "__main__":
    main()
