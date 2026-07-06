"""
End-to-end orchestration of cam_watcher.main(), with classify() and send_email()
mocked. Verifies the alert decision, logging, cooldown, and failure handling.
"""
import sys

import cam_watcher
import db

CONFIG = {"testcam": {"prompt": "p", "alert_on": ["OPEN"], "notify_cooldown_sec": 1000}}


def run_main(monkeypatch, cam, image, *, classify_ret, email_ret=True):
    """Invoke main() with mocked classify/send_email; return recorded emails."""
    monkeypatch.setattr(cam_watcher, "ALERT_IMAGE_DIR", "")  # determinism
    monkeypatch.setattr(cam_watcher, "load_config", lambda: CONFIG)
    monkeypatch.setattr(cam_watcher, "classify", lambda prompt, b64: classify_ret)

    emails = []
    monkeypatch.setattr(
        cam_watcher, "send_email",
        lambda subject, body, image_path=None: emails.append((subject, body, image_path)) or email_ret,
    )
    monkeypatch.setattr(sys, "argv", ["cam_watcher.py", cam, str(image)])
    cam_watcher.main()
    return emails


def rows():
    return db.connect().execute(
        "SELECT camera, classification, alerted, note FROM events ORDER BY id"
    ).fetchall()


def test_alert_match_emails_and_logs(monkeypatch, temp_db, sample_image):
    emails = run_main(monkeypatch, "testcam", sample_image, classify_ret=("OPEN\nthe door is open", 50))
    assert len(emails) == 1
    assert rows() == [("testcam", "OPEN", 1, None)]


def test_no_match_does_not_email(monkeypatch, temp_db, sample_image):
    emails = run_main(monkeypatch, "testcam", sample_image, classify_ret=("CLOSED\nshut", 50))
    assert emails == []
    assert rows() == [("testcam", "CLOSED", 0, "no-match")]


def test_unknown_camera_skips(monkeypatch, temp_db, sample_image):
    emails = run_main(monkeypatch, "ghostcam", sample_image, classify_ret=("OPEN\nx", 1))
    assert emails == []
    assert rows() == []


def test_missing_image_skips(monkeypatch, temp_db, tmp_path):
    emails = run_main(monkeypatch, "testcam", tmp_path / "nope.jpg", classify_ret=("OPEN\nx", 1))
    assert emails == []
    assert rows() == []


def test_ollama_failure_logged(monkeypatch, temp_db, sample_image):
    emails = run_main(monkeypatch, "testcam", sample_image, classify_ret=(None, 120))
    assert emails == []
    r = rows()
    assert r[0][:3] == ("testcam", None, 0)
    assert "error" in r[0][3]


def test_cooldown_suppresses_second_alert(monkeypatch, temp_db, sample_image):
    run_main(monkeypatch, "testcam", sample_image, classify_ret=("OPEN\nx", 10))
    emails = run_main(monkeypatch, "testcam", sample_image, classify_ret=("OPEN\nx", 10))
    assert emails == []  # within the 1000s cooldown
    r = rows()
    assert r[0][2] == 1               # first alerted
    assert r[1][2] == 0 and r[1][3] == "cooldown"


def test_failed_email_does_not_start_cooldown(monkeypatch, temp_db, sample_image):
    # First alert: email fails -> alerted=0, so no cooldown is established.
    run_main(monkeypatch, "testcam", sample_image, classify_ret=("OPEN\nx", 10), email_ret=False)
    # Second alert immediately after should STILL try to email.
    emails = run_main(monkeypatch, "testcam", sample_image, classify_ret=("OPEN\nx", 10), email_ret=True)
    assert len(emails) == 1
    r = rows()
    assert r[0][2] == 0 and "email failed" in (r[0][3] or "")
    assert r[1][2] == 1
