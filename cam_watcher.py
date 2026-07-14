#!/usr/bin/env python3
"""
BlueIris -> Ollama camera alert agent (the glue script).

BlueIris calls this identically for every camera:

    python cam_watcher.py "&CAM" "&ALERT_PATH"

Behavior is driven entirely by cameras.yaml, so adding a camera is a config
edit, not a code change. The flow:

    argv -> look up camera config -> base64 the image -> ask Ollama ->
    normalise answer -> log to SQLite (always) -> if it matches alert_on and
    we're past the cooldown, email it (with the image attached).

Design rule: BlueIris does not see or care about our exit code, so nothing here
is allowed to crash. Every failure path is logged and we exit cleanly.
"""

import base64
import logging
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

import db
import risk
from notify import send_email

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")  # must run before the os.getenv calls below

CONFIG_FILE = SCRIPT_DIR / "cameras.yaml"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "cam-watcher")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))


def _keep_alive(raw: str):
    """Ollama keep_alive: a number (seconds, -1 = forever) or a duration string
    like '30m'/'8h'. Send numerics as numbers, everything else as a string."""
    try:
        return int(raw)
    except ValueError:
        return raw


# How long Ollama keeps the model loaded after a request. The default keeps it
# warm so sparse motion alerts don't each pay a ~7s model reload. Use "-1" to
# keep it resident forever (best latency; holds the VRAM permanently).
OLLAMA_KEEP_ALIVE = _keep_alive(os.getenv("OLLAMA_KEEP_ALIVE", "30m"))
# BlueIris's &ALERT_PATH can arrive as a bare filename (no folder). If so, we
# resolve it against this directory -- BlueIris's alert-image storage folder.
ALERT_IMAGE_DIR = os.getenv("ALERT_IMAGE_DIR", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(SCRIPT_DIR / "cam_watcher.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load cameras.yaml; returns {} (and logs) if it can't be read."""
    try:
        return yaml.safe_load(CONFIG_FILE.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.error(f"Could not read {CONFIG_FILE}: {e}")
        return {}


def parse_answer(raw: str) -> tuple[str, str]:
    """
    Split the model output into (LABEL, description).

    The model is told to put a single UPPER_SNAKE_CASE label on the first line
    and a description after it. We stay tolerant of it collapsing both onto one
    line ("LABEL: description" / "LABEL - description") so a slightly off-format
    reply still yields a usable label.
    """
    text = raw.strip()
    if not text:
        return "", ""
    first, _, rest = text.partition("\n")
    # First whitespace/colon/dash-delimited token of line 1 is the label.
    parts = re.split(r"[\s:,\-]+", first.strip(), maxsplit=1)
    label = parts[0].upper()
    inline = parts[1].strip() if len(parts) > 1 else ""
    description = " ".join(p for p in (inline, rest.strip()) if p).strip()
    return label, description


def classify(prompt: str, img_b64: str) -> tuple[str | None, int]:
    """
    Ask Ollama about the image. Returns (raw_answer, latency_ms).
    raw_answer is None on any error (already logged).
    """
    start = time.monotonic()
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "user", "content": prompt, "images": [img_b64]}
                ],
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        latency_ms = int((time.monotonic() - start) * 1000)
        answer = resp.json()["message"]["content"].strip()
        return answer, latency_ms
    except requests.RequestException as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.error(f"Ollama request failed: {e}")
        return None, latency_ms
    except (KeyError, ValueError) as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.error(f"Unexpected Ollama response shape: {e}")
        return None, latency_ms


def resolve_image(image_path: str) -> Path | None:
    """
    Find the alert image on disk. BlueIris's &ALERT_PATH may be a full path or,
    depending on version/settings, just a bare filename. Try it as given first,
    then fall back to ALERT_IMAGE_DIR / filename. Returns the resolved Path, or
    None if the file can't be found.
    """
    p = Path(image_path)
    if p.is_file():
        return p
    if ALERT_IMAGE_DIR:
        candidate = Path(ALERT_IMAGE_DIR) / p.name
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    if len(sys.argv) < 3:
        logger.error('Usage: cam_watcher.py "<CAM>" "<ALERT_PATH>"')
        return

    cam = sys.argv[1]
    image_path = sys.argv[2]

    config = load_config()
    cfg = config.get(cam)
    if not cfg:
        logger.info(f"No config for camera '{cam}', skipping.")
        return

    image_file = resolve_image(image_path)
    if image_file is None:
        logger.error(f"[{cam}] alert image not found: {image_path}")
        return
    image_path = str(image_file)  # use the resolved absolute path downstream

    try:
        img_b64 = base64.b64encode(image_file.read_bytes()).decode()
    except OSError as e:
        logger.error(f"[{cam}] could not read image {image_path}: {e}")
        return

    raw, latency_ms = classify(cfg["prompt"], img_b64)

    conn = db.connect()

    if raw is None:
        db.log_event(
            conn, cam, None, None, None, latency_ms, image_path,
            alerted=False, note="error: ollama call failed",
        )
        return

    label, description = parse_answer(raw)
    logger.info(f"[{cam}] {label} -- {description} ({latency_ms} ms)")

    # Risk scoring (shadow mode): compute and log the rolling score, but take
    # no action on it yet. The score is a pure function of the recent event
    # log, so it needs no stored state -- see risk.py.
    risk_kw = {}
    risk_cfg = risk.risk_config(config)
    if risk_cfg:
        now = time.time()
        window = float(risk_cfg.get("window_sec", risk.DEFAULT_WINDOW_SEC))
        past = db.recent_events(conn, now - window)
        delta = risk.event_delta(cam, label, now, config)
        # Score the current event WITH the history (not added separately) so
        # repeat dampening ranks it as the newest of its (camera, label).
        score = risk.score_events(past + [(cam, label, now)], now, config)
        tier = risk.tier_for(score, config)
        risk_kw = {"risk_delta": delta, "risk_score": score, "risk_tier": tier}
        logger.info(f"[{cam}] risk score {score:.1f} ({tier}), event +{delta:.1f}")

    alert_tags = cfg.get("alert_on", [])
    is_match = any(tag.upper() in label for tag in alert_tags)

    if not is_match:
        db.log_event(
            conn, cam, label, description, raw, latency_ms, image_path,
            alerted=False, note="no-match", **risk_kw,
        )
        return

    # Matched an alert label -- check the per-camera cooldown.
    cooldown = cfg.get("notify_cooldown_sec", 0)
    last = db.last_alert_epoch(conn, cam)
    if time.time() - last < cooldown:
        logger.info(f"[{cam}] '{label}' suppressed (cooldown {cooldown}s)")
        db.log_event(
            conn, cam, label, description, raw, latency_ms, image_path,
            alerted=False, note="cooldown", **risk_kw,
        )
        return

    body = f"{cam}: {label}\n\n{description}\n\nImage: {image_path}"
    sent = send_email(
        subject=f"Camera Alert: {cam} - {label}",
        body=body,
        image_path=image_path,
    )
    db.log_event(
        conn, cam, label, description, raw, latency_ms, image_path,
        alerted=sent, note=None if sent else "email failed", **risk_kw,
    )


if __name__ == "__main__":
    main()
