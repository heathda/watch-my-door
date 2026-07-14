"""
Validate the camera config(s). cameras.yaml.example is always present and
checked; the real cameras.yaml is checked too when it exists locally (it's
gitignored, so a fresh clone only has the example).

These checks catch the typos that hurt most: a malformed entry, or an alert_on
label the model is never told about (so the alert could never fire).
"""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILES = [p for p in (ROOT / "cameras.yaml.example", ROOT / "cameras.yaml") if p.is_file()]


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
