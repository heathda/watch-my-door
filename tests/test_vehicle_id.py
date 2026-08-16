"""The two-call vehicle identity check.

This module can change which label reaches alert_on, so its failure modes are
the expensive kind. The tests below are weighted accordingly: most of them are
about what it must NOT do -- never fire on a non-vehicle frame, never invent a
label, never let an Ollama failure or an unparseable answer flip a label, and
never raise. Offline: `classify` is a stub, no Ollama needed.
"""
import pytest

import vehicle_id

CFG = {
    "prompt": "... a WHITE pickup truck and a BLACK SUV ... KNOWN_VEHICLE ... UNFAMILIAR_VEHICLE ...",
    "vehicle_check": {
        "enabled": True,
        "triggers": ["KNOWN_VEHICLE", "UNFAMILIAR_VEHICLE"],
        "known": ["white pickup", "black suv"],
        "known_label": "KNOWN_VEHICLE",
        "unknown_label": "UNFAMILIAR_VEHICLE",
        "prompt": "Is it OURS or NOT-OURS?",
    },
}


def answering(text, latency=1200):
    """A stub classify() that always replies with `text`."""
    def _classify(prompt, img_b64):
        return text, latency
    return _classify


def boom(prompt, img_b64):
    raise AssertionError("call 2 must not have been made")


# --- when it runs at all ----------------------------------------------------

@pytest.mark.parametrize("label", ["NONE", "PERSON", "MULTIPLE_PEOPLE", "DELIVERY", ""])
def test_does_not_fire_on_non_vehicle_labels(label):
    """~85% of this camera's frames are an empty driveway. If the check fired on
    those it would double the latency of the whole pipeline for nothing."""
    out = vehicle_id.identify(label, CFG, "img", boom)
    assert out == {"label": label, "note": None, "latency_ms": 0, "ran": False}


def test_absent_block_means_off():
    out = vehicle_id.identify("KNOWN_VEHICLE", {"prompt": "p"}, "img", boom)
    assert out["ran"] is False and out["label"] == "KNOWN_VEHICLE"


def test_enabled_false_means_off():
    cfg = {"prompt": "p", "vehicle_check": {**CFG["vehicle_check"], "enabled": False}}
    out = vehicle_id.identify("KNOWN_VEHICLE", cfg, "img", boom)
    assert out["ran"] is False


# --- the point of the whole thing -------------------------------------------

def test_promotes_a_stranger_to_unfamiliar():
    """The defect this exists for: the model calls every vehicle KNOWN_VEHICLE.
    A silver sedan is not one of ours, and Python cannot decline to notice."""
    out = vehicle_id.identify("KNOWN_VEHICLE", CFG, "img", answering("NOT-OURS silver sedan"))
    assert out["label"] == "UNFAMILIAR_VEHICLE"
    assert out["note"] == "vehicle-id=unknown->UNFAMILIAR_VEHICLE"
    assert out["ran"] is True and out["latency_ms"] == 1200


def test_demotes_our_own_truck_back_to_known():
    """It has to work in both directions, or turning it on trades a missed
    alert for a nightly false one on the household truck."""
    out = vehicle_id.identify("UNFAMILIAR_VEHICLE", CFG, "img", answering("OURS white pickup truck"))
    assert out["label"] == "KNOWN_VEHICLE"
    assert "->KNOWN_VEHICLE" in out["note"]


def test_agreeing_with_call_one_leaves_the_label_alone():
    out = vehicle_id.identify("KNOWN_VEHICLE", CFG, "img", answering("OURS black suv"))
    assert out["label"] == "KNOWN_VEHICLE"
    assert out["note"] == "vehicle-id=known"  # no arrow: nothing changed


# --- tolerance to how the model actually answers ----------------------------

@pytest.mark.parametrize("reply", [
    "OURS", "ours", "OURS White pickup truck parked in the driveway",
    "It is ours.", "OURS\nwhite pickup",
])
def test_parses_the_ways_it_says_ours(reply):
    assert vehicle_id.identify("UNFAMILIAR_VEHICLE", CFG, "img",
                               answering(reply))["label"] == "KNOWN_VEHICLE"


@pytest.mark.parametrize("reply", [
    "NOT-OURS", "NOT OURS", "notours", "not_ours",
    "NOT-OURS White sedan parked", "This one is not ours.",
])
def test_parses_the_ways_it_says_not_ours(reply):
    assert vehicle_id.identify("KNOWN_VEHICLE", CFG, "img",
                               answering(reply))["label"] == "UNFAMILIAR_VEHICLE"


def test_not_ours_is_matched_before_ours():
    """The trap that would silently disable the whole feature: 'NOT-OURS'
    contains 'OURS'. Test the substring order explicitly, because the symptom --
    every stranger reading as the household car -- looks exactly like a working
    system that simply never sees a stranger."""
    assert vehicle_id.verdict("NOT-OURS") == "unknown"
    assert vehicle_id.verdict("not ours, a silver sedan") == "unknown"
    assert vehicle_id.verdict("OURS") == "known"


def test_word_boundaries_stop_accidental_matches():
    """'ours' must not fire inside 'hours', 'colours', 'yours'."""
    assert vehicle_id.verdict("visible for hours") == "unsure"
    assert vehicle_id.verdict("two colours") == "unsure"


# --- every doubt keeps call 1's answer --------------------------------------

def test_ollama_failure_on_call_two_never_changes_the_label():
    """A transport failure is not evidence about a vehicle."""
    out = vehicle_id.identify("KNOWN_VEHICLE", CFG, "img", lambda p, i: (None, 900))
    assert out["label"] == "KNOWN_VEHICLE"
    assert out["note"] == "vehicle-id=call-failed"


@pytest.mark.parametrize("reply", ["", "a vehicle", "I cannot tell", "banana", "UNSURE"])
def test_an_unparseable_answer_keeps_the_label(reply):
    """This is the 3am rule from WAR-STORIES.md: never raise an alarm because
    you could not see clearly."""
    out = vehicle_id.identify("KNOWN_VEHICLE", CFG, "img", answering(reply))
    assert out["label"] == "KNOWN_VEHICLE"
    assert out["note"].startswith("vehicle-id=unsure")


def test_unsure_does_not_downgrade_a_real_unfamiliar_either():
    """Symmetry: doubt keeps whatever call 1 said, in both directions."""
    out = vehicle_id.identify("UNFAMILIAR_VEHICLE", CFG, "img", answering("dunno"))
    assert out["label"] == "UNFAMILIAR_VEHICLE"


def test_missing_second_prompt_is_reported_not_guessed():
    cfg = {"prompt": "p", "vehicle_check": {k: v for k, v in CFG["vehicle_check"].items()
                                            if k != "prompt"}}
    out = vehicle_id.identify("KNOWN_VEHICLE", cfg, "img", boom)
    assert out["label"] == "KNOWN_VEHICLE" and "misconfigured" in out["note"]
