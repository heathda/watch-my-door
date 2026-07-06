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


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_config_well_formed(path):
    cameras = yaml.safe_load(path.read_text())
    assert isinstance(cameras, dict) and cameras, f"{path.name}: must be a non-empty mapping"
    for name, cfg in cameras.items():
        assert isinstance(cfg.get("prompt"), str) and cfg["prompt"].strip(), f"{name}: missing prompt"
        assert isinstance(cfg.get("alert_on"), list), f"{name}: alert_on must be a list"
        assert all(isinstance(x, str) for x in cfg["alert_on"]), f"{name}: alert_on labels must be strings"
        cooldown = cfg.get("notify_cooldown_sec", 0)
        assert isinstance(cooldown, int) and cooldown >= 0, f"{name}: notify_cooldown_sec must be int >= 0"


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_alert_labels_are_mentioned_in_prompt(path):
    cameras = yaml.safe_load(path.read_text())
    for name, cfg in cameras.items():
        prompt = cfg["prompt"].upper()
        for label in cfg["alert_on"]:
            assert label.upper() in prompt, (
                f"{path.name}/{name}: alert_on label {label!r} does not appear in the prompt"
            )
