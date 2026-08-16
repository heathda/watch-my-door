"""
Validate the camera config(s). cameras.yaml.example is always present and
checked; the real cameras.yaml is checked too when it exists locally (it's
gitignored, so a fresh clone only has the example).

Candidate configs at the repo root (cameras-<name>-<date>.yaml, the A/B
candidates) are checked as well. One of them is one `copy` away from being
production, and attempt 4 shipped only after test_config caught a weighted
label it had deleted from the prompt -- a candidate that has not been validated
is a candidate that can take the alerting down when it wins its A/B. Historical
snapshots under docs/configs/ are deliberately NOT checked: they are a record of
what was measured, including the rejects, and rewriting history to satisfy a
later rule would defeat the point of keeping them.

These checks catch the typos that hurt most: a malformed entry, or an alert_on
label the model is never told about (so the alert could never fire).
"""
from pathlib import Path

import pytest
import yaml

import watchdog

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILES = sorted(
    p for p in ROOT.glob("cameras*.yaml*")
    if p.is_file() and p.suffix in (".yaml", ".example")
)


def camera_entries(config: dict) -> dict:
    """Just the cameras: keys starting with '_' (e.g. _risk) are global blocks,
    not cameras -- BlueIris camera names can't start with an underscore."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_config_well_formed(path):
    config = yaml.safe_load(path.read_text())
    assert isinstance(config, dict) and config, f"{path.name}: must be a non-empty mapping"
    cameras = camera_entries(config)
    assert cameras, f"{path.name}: no camera entries"
    for name, cfg in cameras.items():
        assert isinstance(cfg.get("prompt"), str) and cfg["prompt"].strip(), f"{name}: missing prompt"
        assert isinstance(cfg.get("alert_on"), list), f"{name}: alert_on must be a list"
        assert all(isinstance(x, str) for x in cfg["alert_on"]), f"{name}: alert_on labels must be strings"
        cooldown = cfg.get("notify_cooldown_sec", 0)
        assert isinstance(cooldown, int) and cooldown >= 0, f"{name}: notify_cooldown_sec must be int >= 0"


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_alert_labels_are_mentioned_in_prompt(path):
    cameras = camera_entries(yaml.safe_load(path.read_text()))
    for name, cfg in cameras.items():
        prompt = cfg["prompt"].upper()
        for label in cfg["alert_on"]:
            assert label.upper() in prompt, (
                f"{path.name}/{name}: alert_on label {label!r} does not appear in the prompt"
            )


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_watchdog_config_well_formed(path):
    """A misconfigured watchdog fails silently in the worst way: it looks
    installed and never fires. Absent block = feature off, which is fine."""
    cfg = yaml.safe_load(path.read_text()).get("_watchdog")
    if cfg is None:
        return
    assert isinstance(cfg.get("enabled"), bool), f"{path.name}: _watchdog.enabled must be a bool"
    known = set(watchdog.DEFAULTS)
    unknown = set(cfg) - known - {"enabled"}
    assert not unknown, f"{path.name}: unknown _watchdog section(s) {sorted(unknown)} (typo?)"
    for section, values in cfg.items():
        if section == "enabled":
            continue
        stray = set(values) - set(watchdog.DEFAULTS[section])
        assert not stray, f"{path.name}: unknown key(s) {sorted(stray)} in _watchdog.{section}"
        for key, val in values.items():
            default = watchdog.DEFAULTS[section][key]
            assert isinstance(val, type(default)), (
                f"{path.name}: _watchdog.{section}.{key} should be {type(default).__name__}"
            )
            if isinstance(val, bool):
                continue
            assert val > 0, f"{path.name}: _watchdog.{section}.{key} must be > 0"

    stale = {**watchdog.DEFAULTS["stale"], **(cfg.get("stale") or {})}
    assert stale["night_floor_min"] >= stale["floor_min"], (
        f"{path.name}: night_floor_min below floor_min would make the watchdog "
        f"twitchier at 3am than at noon, which is backwards"
    )
    assert stale["ceiling_min"] >= stale["floor_min"], (
        f"{path.name}: ceiling_min below floor_min clamps the threshold under its own floor"
    )
    errors = {**watchdog.DEFAULTS["errors"], **(cfg.get("errors") or {})}
    assert 0 < errors["max_fail_ratio"] <= 1, f"{path.name}: max_fail_ratio must be in (0, 1]"


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_no_time_of_day_labels(path):
    """Time of day belongs to risk.py, not to the label space.

    risk.night_multiplier already scales EVERY event by its timestamp, so a
    night-flavored label (AT_NIGHT_PERSON, GATE_OPEN_NIGHT, ...) double-counts
    the same signal -- and in 28 days of production the model never once
    emitted AT_NIGHT_PERSON on any camera, so the escalation it existed to
    trigger never fired. Retired 2026-07-25; this keeps them from creeping back.
    """
    for name, cfg in camera_entries(yaml.safe_load(path.read_text())).items():
        labels = set(cfg["alert_on"]) | set((cfg.get("risk") or {}).get("label_weights", {}))
        offenders = [l for l in labels if "NIGHT" in l.upper()]
        assert not offenders, (
            f"{path.name}/{name}: time-of-day label(s) {offenders} -- let the "
            f"_risk night multiplier handle the clock instead"
        )


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_vehicle_check_matches_the_prompt(path):
    """The household vehicles must be described the same way in both places.

    `vehicle_check.known` is what Python compares against; the prompt is what
    the model is told. If they drift, the model and the comparison are working
    from different ideas of "ours" -- and the failure is silent, because both
    halves keep working, just on different definitions. Same reasoning as
    test_alert_labels_are_mentioned_in_prompt.
    """
    import vehicle_id

    for name, cfg in camera_entries(yaml.safe_load(path.read_text())).items():
        vc = vehicle_id.check_config(cfg)
        if vc is None:
            continue
        main_prompt = cfg["prompt"].lower()
        second_prompt = vc.get("prompt", "").lower()
        known = vc.get("known") or []
        assert known, f"{path.name}/{name}: vehicle_check.known is empty"
        # The household vehicles are described in two places -- the camera prompt
        # and the second-call prompt -- and `known` is the anchor asserting they
        # agree. Drift is silent: both halves keep working, on different ideas of
        # "ours", and the symptom is an email about your own car.
        for entry in known:
            for word in str(entry).lower().split():
                assert word in main_prompt, (
                    f"{path.name}/{name}: vehicle_check.known says {entry!r} but the "
                    f"camera prompt never says {word!r}"
                )
                assert word in second_prompt, (
                    f"{path.name}/{name}: vehicle_check.known says {entry!r} but the "
                    f"second-call prompt never says {word!r}"
                )
        # The second call is asked for a verdict, and vehicle_id only understands
        # OURS / NOT-OURS. A prompt that forgets to ask for them yields 'unsure'
        # on every frame -- a feature that is on, costs a call, and does nothing.
        assert "not-ours" in second_prompt and "ours" in second_prompt, (
            f"{path.name}/{name}: vehicle_check.prompt must ask for OURS / NOT-OURS"
        )
        for label_key in ("known_label", "unknown_label"):
            label = vc.get(label_key)
            assert label and label.upper() in cfg["prompt"].upper(), (
                f"{path.name}/{name}: vehicle_check.{label_key}={label!r} is not a "
                f"label this prompt defines, so the check could rewrite an event "
                f"into a label that can never match alert_on"
            )
        assert vc.get("triggers"), f"{path.name}/{name}: vehicle_check has no triggers"
        assert vc.get("prompt", "").strip(), f"{path.name}/{name}: vehicle_check has no prompt"


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_risk_config_well_formed(path):
    """When risk scoring is configured, the _risk block and every per-camera
    risk block must be usable: sane globals, numeric weights, and every
    weighted label actually offered to the model in that camera's prompt
    (a label the model can never emit would silently never score)."""
    config = yaml.safe_load(path.read_text())
    risk_cfg = config.get("_risk")
    cameras = camera_entries(config)
    if risk_cfg is None:
        assert not any("risk" in cfg for cfg in cameras.values()), (
            f"{path.name}: per-camera risk blocks present but no global _risk block"
        )
        return

    for key in ("half_life_sec", "window_sec"):
        val = risk_cfg.get(key)
        assert isinstance(val, (int, float)) and val > 0, f"{path.name}: _risk.{key} must be > 0"
    tiers = risk_cfg.get("tiers")
    assert isinstance(tiers, dict) and tiers, f"{path.name}: _risk.tiers must be a non-empty mapping"
    floors = list(tiers.values())
    assert all(isinstance(v, (int, float)) and v >= 0 for v in floors), (
        f"{path.name}: tier thresholds must be numbers >= 0"
    )
    assert min(floors) == 0, f"{path.name}: one tier must start at 0 (the resting tier)"
    assert len(set(floors)) == len(floors), f"{path.name}: tier thresholds must be distinct"
    night = risk_cfg.get("night")
    if night:
        for key in ("start", "end"):
            h, m = str(night[key]).split(":")
            assert 0 <= int(h) <= 23 and 0 <= int(m) <= 59, f"{path.name}: bad night.{key}"
        assert float(night.get("multiplier", 1.0)) > 0

    for name, cfg in cameras.items():
        rcfg = cfg.get("risk")
        if not rcfg:
            continue
        assert float(rcfg.get("multiplier", 1.0)) > 0, f"{name}: risk.multiplier must be > 0"
        weights = rcfg.get("label_weights")
        assert isinstance(weights, dict) and weights, f"{name}: risk.label_weights must be a non-empty mapping"
        prompt = cfg["prompt"].upper()
        for label, weight in weights.items():
            assert isinstance(weight, (int, float)) and weight >= 0, (
                f"{path.name}/{name}: weight for {label!r} must be a number >= 0"
            )
            assert label.upper() in prompt, (
                f"{path.name}/{name}: weighted label {label!r} does not appear in the prompt"
            )
