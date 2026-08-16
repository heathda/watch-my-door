"""
End-to-end orchestration of cam_watcher.main(), with classify() and send_email()
mocked. Verifies the alert decision, logging, cooldown, and failure handling.
"""
import sys

import cam_watcher
import db

CONFIG = {"testcam": {"prompt": "p", "alert_on": ["OPEN"], "notify_cooldown_sec": 1000}}


def run_main(monkeypatch, cam, image, *, classify_ret, email_ret=True, config=CONFIG):
    """Invoke main() with mocked classify/send_email; return recorded emails."""
    monkeypatch.setattr(cam_watcher, "ALERT_IMAGE_DIR", "")  # determinism
    monkeypatch.setattr(cam_watcher, "load_config", lambda: config)
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


# A prompt that actually defines its labels, so the unknown-label guard is live
# (with CONFIG's placeholder "p" prompt it correctly does nothing).
LABELLED = {
    "testcam": {
        "prompt": "Choose one: OPEN (the door is open), NONE (nothing).",
        "alert_on": ["OPEN"],
        "notify_cooldown_sec": 1000,
    }
}


def test_invented_label_is_flagged_and_does_not_email(monkeypatch, temp_db, sample_image):
    """The silent failure this guard exists for: a label the prompt never
    offered matches no alert_on entry, so it looks exactly like a correct
    no-match in the log. The note is what tells them apart."""
    emails = run_main(monkeypatch, "testcam", sample_image,
                      classify_ret=("DOOR_AJAR\nhalf open", 10), config=LABELLED)
    assert emails == []
    camera, label, alerted, note = rows()[0]
    assert (label, alerted) == ("DOOR_AJAR", 0)
    assert "unknown-label" in note and "no-match" in note


def test_known_label_keeps_its_plain_note(monkeypatch, temp_db, sample_image):
    """No false positives: a defined label must not pick up the tag."""
    run_main(monkeypatch, "testcam", sample_image,
             classify_ret=("NONE\nnothing there", 10), config=LABELLED)
    assert rows()[0][3] == "no-match"


def test_invented_label_still_alerts_if_it_matches(monkeypatch, temp_db, sample_image):
    """The guard annotates, it never suppresses. An invented OPEN_WIDE still
    contains 'OPEN', so it must email -- downgrading that to a miss would make
    the cure worse than the disease."""
    emails = run_main(monkeypatch, "testcam", sample_image,
                      classify_ret=("OPEN_WIDE\nvery open", 10), config=LABELLED)
    assert len(emails) == 1
    assert rows()[0][3] == "unknown-label"


def test_guard_is_inert_when_prompt_has_no_label_definitions(monkeypatch, temp_db, sample_image):
    """CONFIG's prompt is just "p" -- unparseable, so nothing may be flagged."""
    run_main(monkeypatch, "testcam", sample_image, classify_ret=("WHATEVER\nx", 10))
    assert rows()[0][3] == "no-match"


def test_risk_shadow_mode_logs_score_without_changing_behavior(monkeypatch, temp_db, sample_image):
    config = {
        "_risk": {
            "half_life_sec": 600,
            "window_sec": 7200,
            "repeat_dampening": 0.5,
            "tiers": {"QUIET": 0, "NOTICE": 20, "ELEVATED": 50},
        },
        "testcam": {
            "prompt": "p", "alert_on": ["OPEN"], "notify_cooldown_sec": 1000,
            "risk": {"multiplier": 1.0, "label_weights": {"OPEN": 25}},
        },
    }
    emails = run_main(monkeypatch, "testcam", sample_image,
                      classify_ret=("OPEN\nx", 10), config=config)
    assert len(emails) == 1  # alerting behavior unchanged by risk config
    row = db.connect().execute(
        "SELECT risk_delta, risk_score, risk_tier, alerted FROM events"
    ).fetchone()
    assert row == (25.0, 25.0, "NOTICE", 1)

    # Second event moments later: dampened repeat (~+12.5), score rises, and
    # the cooldown still suppresses -- shadow mode takes no action.
    emails = run_main(monkeypatch, "testcam", sample_image,
                      classify_ret=("OPEN\nx", 10), config=config)
    assert emails == []
    _, score, tier, note = db.connect().execute(
        "SELECT risk_delta, risk_score, risk_tier, note FROM events ORDER BY id DESC"
    ).fetchone()
    assert note == "cooldown" and tier == "NOTICE" and 25.0 < score < 37.5


def test_failed_email_does_not_start_cooldown(monkeypatch, temp_db, sample_image):
    # First alert: email fails -> alerted=0, so no cooldown is established.
    run_main(monkeypatch, "testcam", sample_image, classify_ret=("OPEN\nx", 10), email_ret=False)
    # Second alert immediately after should STILL try to email.
    emails = run_main(monkeypatch, "testcam", sample_image, classify_ret=("OPEN\nx", 10), email_ret=True)
    assert len(emails) == 1
    r = rows()
    assert r[0][2] == 0 and "email failed" in (r[0][3] or "")
    assert r[1][2] == 1


# --- the two-call vehicle identity check, end to end ------------------------
# vehicle_id has its own unit tests; these check the one thing they cannot --
# that the rewritten label is what reaches alert_on, the cooldown and the email,
# and not merely what gets logged.

VEHICLE_CONFIG = {
    "drivecam": {
        "prompt": "a WHITE pickup and a BLACK SUV. KNOWN_VEHICLE UNFAMILIAR_VEHICLE",
        "alert_on": ["UNFAMILIAR_VEHICLE"],
        "notify_cooldown_sec": 0,
        "vehicle_check": {
            "enabled": True,
            "triggers": ["KNOWN_VEHICLE", "UNFAMILIAR_VEHICLE"],
            "known": ["white pickup", "black suv"],
            "known_label": "KNOWN_VEHICLE",
            "unknown_label": "UNFAMILIAR_VEHICLE",
            "prompt": "Is it OURS or NOT-OURS?",
        },
    }
}


def run_two_call(monkeypatch, first, second, image, email_ret=True):
    """main() with a classify() that answers call 1 and call 2 differently."""
    replies = iter([first, second])
    monkeypatch.setattr(cam_watcher, "ALERT_IMAGE_DIR", "")
    monkeypatch.setattr(cam_watcher, "load_config", lambda: VEHICLE_CONFIG)
    monkeypatch.setattr(cam_watcher, "classify", lambda prompt, b64: next(replies))
    emails = []
    monkeypatch.setattr(
        cam_watcher, "send_email",
        lambda subject, body, image_path=None: emails.append(subject) or email_ret,
    )
    monkeypatch.setattr(sys, "argv", ["cam_watcher.py", "drivecam", str(image)])
    cam_watcher.main()
    return emails


def test_identity_check_turns_a_silent_frame_into_an_email(monkeypatch, temp_db, sample_image):
    """The defect the whole feature exists for. Call 1 says KNOWN_VEHICLE, which
    is weight 0 and not in alert_on, so today this frame dies silently. Call 2
    identifies a silver sedan and the owner gets mail."""
    emails = run_two_call(
        monkeypatch, ("KNOWN_VEHICLE\na vehicle is parked", 50), ("NOT-OURS silver sedan", 40), sample_image)
    assert emails == ["Camera Alert: drivecam - UNFAMILIAR_VEHICLE"]
    cam, label, alerted, note = rows()[0]
    assert (label, alerted) == ("UNFAMILIAR_VEHICLE", 1)
    assert note == "vehicle-id=unknown->UNFAMILIAR_VEHICLE"


def test_identity_check_suppresses_a_false_alarm_on_our_own_truck(monkeypatch, temp_db, sample_image):
    """The other direction, and the one that decides whether this is safe to
    leave on: call 1 cries UNFAMILIAR_VEHICLE at the household pickup, and the
    check must stop the email, not just annotate it."""
    emails = run_two_call(
        monkeypatch, ("UNFAMILIAR_VEHICLE\nsome vehicle", 50), ("OURS white pickup truck", 40), sample_image)
    assert emails == []
    cam, label, alerted, note = rows()[0]
    assert (label, alerted) == ("KNOWN_VEHICLE", 0)
    assert note == "vehicle-id=known->KNOWN_VEHICLE; no-match"


def test_second_call_failure_leaves_the_alert_decision_untouched(monkeypatch, temp_db, sample_image):
    """Ollama dying on call 2 must not silence a real alert."""
    emails = run_two_call(
        monkeypatch, ("UNFAMILIAR_VEHICLE\nsome vehicle", 50), (None, 40), sample_image)
    assert emails == ["Camera Alert: drivecam - UNFAMILIAR_VEHICLE"]
    assert rows()[0][1:] == ("UNFAMILIAR_VEHICLE", 1, "vehicle-id=call-failed")


def test_unsure_keeps_call_one_and_still_emails(monkeypatch, temp_db, sample_image):
    emails = run_two_call(
        monkeypatch, ("UNFAMILIAR_VEHICLE\nsome vehicle", 50), ("I cannot tell", 40), sample_image)
    assert emails == ["Camera Alert: drivecam - UNFAMILIAR_VEHICLE"]
    assert rows()[0][3].startswith("vehicle-id=unsure")


def test_non_vehicle_frames_make_only_one_call(monkeypatch, temp_db, sample_image):
    """~85% of frames. A second call here would double the pipeline's latency
    for nothing, so the second reply is a tripwire that must never be reached."""
    def one_only(prompt, b64):
        return ("NONE\nempty driveway", 50)
    monkeypatch.setattr(cam_watcher, "ALERT_IMAGE_DIR", "")
    monkeypatch.setattr(cam_watcher, "load_config", lambda: VEHICLE_CONFIG)
    calls = []
    monkeypatch.setattr(cam_watcher, "classify",
                        lambda prompt, b64: calls.append(prompt) or one_only(prompt, b64))
    monkeypatch.setattr(cam_watcher, "send_email", lambda **kw: True)
    monkeypatch.setattr(sys, "argv", ["cam_watcher.py", "drivecam", str(sample_image)])
    cam_watcher.main()
    assert len(calls) == 1
    assert rows()[0][3] == "no-match"


def test_latency_recorded_is_the_sum_of_both_calls(monkeypatch, temp_db, sample_image):
    """Otherwise the cost of this feature is invisible in the very log that
    exists to make costs visible."""
    run_two_call(monkeypatch, ("KNOWN_VEHICLE\nx", 50), ("NOT-OURS silver sedan", 40), sample_image)
    latency = db.connect().execute("SELECT latency_ms FROM events").fetchone()[0]
    assert latency == 90
