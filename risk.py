#!/usr/bin/env python3
"""
Deterministic risk scoring over the event log.

The score at any moment is a pure function of recent events -- the sum of each
event's points ("delta"), decayed exponentially by its age:

    score(now) = sum( delta(e) * 0.5 ** ((now - e.ts_epoch) / half_life) )

There is no stored running level: this decayed sum is mathematically identical
to the recursive "decay the old level, add the new delta" form, but with no
mutable state to drift or corrupt, and it makes replaying history through a
different config trivial. The risk_* columns in events.db are an audit log of
what the scorer decided at the time -- they are written, never read back.

The LLM never produces the score; it only supplies the per-frame label.

Config lives in cameras.yaml: a global `_risk` block (half-life, window, night
hours, tier thresholds) plus an optional per-camera `risk` block (multiplier +
label_weights). No `_risk` block = feature off (deltas 0, tier None). A label
missing from label_weights scores 0 -- unknown is not scary, and it keeps the
config the single source of truth.
"""

from datetime import datetime

DEFAULT_HALF_LIFE_SEC = 600
DEFAULT_WINDOW_SEC = 7200


def risk_config(config: dict) -> dict | None:
    """The global `_risk` block, or None when the feature is off."""
    cfg = config.get("_risk")
    return cfg if isinstance(cfg, dict) else None


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def night_multiplier(ts_epoch: float, risk_cfg: dict) -> float:
    """Multiplier for events inside the configured night window (local time).
    The window may wrap midnight (e.g. 22:00-06:00)."""
    night = risk_cfg.get("night")
    if not night:
        return 1.0
    t = datetime.fromtimestamp(ts_epoch)
    now_m = t.hour * 60 + t.minute
    start, end = _minutes(str(night["start"])), _minutes(str(night["end"]))
    if start <= end:
        is_night = start <= now_m < end
    else:  # wraps midnight
        is_night = now_m >= start or now_m < end
    return float(night.get("multiplier", 1.0)) if is_night else 1.0


def event_delta(camera: str, label: str | None, ts_epoch: float, config: dict) -> float:
    """Points one event contributes at the moment it happens:
    label weight x camera multiplier x time-of-day multiplier."""
    risk_cfg = risk_config(config)
    if not risk_cfg or not label:
        return 0.0
    cam_risk = (config.get(camera) or {}).get("risk") or {}
    weight = cam_risk.get("label_weights", {}).get(label, 0)
    if not weight:
        return 0.0
    multiplier = float(cam_risk.get("multiplier", 1.0))
    return float(weight) * multiplier * night_multiplier(ts_epoch, risk_cfg)


def score_events(events, now: float, config: dict) -> float:
    """Decayed sum over (camera, label, ts_epoch) tuples. Events outside the
    window (or from the future) contribute nothing.

    Repeat dampening: sustained activity re-triggers motion every ~30s, so a
    raw sum scales with *frame count*, not with distinct activity (an open
    garage door becomes 20 near-identical events). The newest event of each
    (camera, label) counts in full; each older repeat of the same signal is
    additionally scaled by `repeat_dampening` per rank. Distinct signals stack
    -- the same frame re-observed does not."""
    risk_cfg = risk_config(config)
    if not risk_cfg:
        return 0.0
    half_life = float(risk_cfg.get("half_life_sec", DEFAULT_HALF_LIFE_SEC))
    window = float(risk_cfg.get("window_sec", DEFAULT_WINDOW_SEC))
    dampening = float(risk_cfg.get("repeat_dampening", 1.0))
    score = 0.0
    repeats: dict[tuple[str, str], int] = {}
    # Newest first, so the undampened copy of each signal is the most recent.
    for camera, label, ts_epoch in sorted(events, key=lambda e: -e[2]):
        age = now - ts_epoch
        if age < 0 or age > window:
            continue
        delta = event_delta(camera, label, ts_epoch, config)
        if not delta:
            continue
        rank = repeats.get((camera, label), 0)
        repeats[(camera, label)] = rank + 1
        score += delta * (dampening**rank) * 0.5 ** (age / half_life)
    return score


def tier_for(score: float, config: dict) -> str | None:
    """Highest tier whose lower bound the score meets, or None if the feature
    is off / no tiers are configured."""
    risk_cfg = risk_config(config)
    tiers = (risk_cfg or {}).get("tiers")
    if not tiers:
        return None
    best = None
    for name, floor in sorted(tiers.items(), key=lambda kv: kv[1]):
        if score >= floor:
            best = name
    return best
