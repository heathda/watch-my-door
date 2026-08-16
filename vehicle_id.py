#!/usr/bin/env python3
"""
Second-call vehicle identity check -- ask the identity question on its own.

WHY THIS EXISTS (2026-08-16). `UNFAMILIAR_VEHICLE` had fired 0 times in 57,948
events. Six prompt rewrites over three weeks failed to change that, and the
measurement that explains why is in docs/PROMPT-LOG-driveway.md: told that the
household cars were "a red sedan and a dark green hatchback" and shown a white
pickup, the model still answered `KNOWN_VEHICLE`, in 47 of 48 draws. Deleting
the "if you cannot tell, choose KNOWN_VEHICLE" fallback did not help -- it
turned those frames into `NONE`.

In the single call, identity is one clause among fifteen labels and a numbered
procedure, and the model satisfices: `KNOWN_VEHICLE` becomes its word for "a
vehicle is present". So this module asks the identity question by itself:

    call 1 (unchanged)  -> "is anything there?"   -> a label
    call 2 (only if that label is a vehicle)
                        -> "is this one of OUR two vehicles?" -> OURS / NOT-OURS

WHAT WAS TRIED FIRST, AND WHY IT IS NOT THIS. The first build asked for
attributes -- "answer with its colour and body type" -- and compared them to a
configured list in Python, on the theory that a string comparison cannot decline
to run. Measured on the corpus, the comparison ran fine and the attributes were
wrong: the household pickup came back "white suv" and, cropped, "silver sedan",
both 4/4 unanimous. It produced two false alarms on the household truck.

The same frames under the direct question: 6/6 correct, 18/18 draws unanimous,
including four vehicles that are not ours. Its descriptions are wrong in the
same ways ("Black SUV" for a grey crossover) while the verdict is right, so the
model can judge identity without being able to name what it is looking at.
Asking for attributes threw that judgement away and rebuilt it from the least
reliable part of the answer.

COST. Call 2 fires only when call 1 returned a vehicle label -- a handful of
frames a day against ~200 alerts, so the ~85% of frames that are an empty
driveway pay nothing.

SAFETY. This can only ever swap one vehicle label for the other. It cannot
invent a label, cannot touch PERSON or NONE, and on any doubt -- an Ollama
failure, an unparseable answer -- it keeps the label call 1 produced. That
preserves the rule this project learned at 3am: never raise an alarm because you
could not see clearly.
"""

import re

# The model is asked to answer OURS or NOT-OURS. It writes NOT-OURS, NOT OURS,
# NOTOURS and "not ours" interchangeably, so match on a normalised string.
# ORDER MATTERS: "NOT-OURS" contains "OURS", so the negative must be tested
# first or every stranger reads as the household car -- a silent, total failure
# of the feature that would still pass a smoke test.
_NOT_OURS = re.compile(r"\bnot[\s_-]*ours\b")
_OURS = re.compile(r"\bours\b")


def check_config(cam_cfg: dict) -> dict | None:
    """The camera's vehicle_check block, or None when the feature is off.

    Absent block = feature off, which is how every camera behaves until it is
    deliberately switched on. A block with enabled: false is also off, so the
    config can stay in place while the check is disabled for a comparison.
    """
    cfg = (cam_cfg or {}).get("vehicle_check")
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return None
    return cfg


def triggers(cfg: dict) -> set[str]:
    """Labels from call 1 that are worth a second call."""
    return {t.upper() for t in cfg.get("triggers", [])}


def verdict(raw: str | None) -> str:
    """'known' | 'unknown' | 'unsure' from the second call's raw text.

    Tolerant of the model wrapping its answer in a sentence, because it does:
    "NOT-OURS White sedan parked in the driveway". Anything that does not
    clearly say one or the other is 'unsure', which the caller turns into "keep
    whatever call 1 said" -- never into an alarm.
    """
    text = (raw or "").lower()
    if _NOT_OURS.search(text):
        return "unknown"
    if _OURS.search(text):
        return "known"
    return "unsure"


def identify(label: str, cam_cfg: dict, img_b64: str, classify_fn) -> dict:
    """Run the check for one frame. Never raises.

    `classify_fn` is cam_watcher.classify, injected so this module has no
    opinion about transport and so the tests need no Ollama.

    Returns a dict that is always safe to apply:
        label      -- the label to use (call 1's unless we are confident)
        note       -- short, greppable audit string, or None when we did not run
        latency_ms -- cost of call 2, 0 when it did not happen
        ran        -- whether call 2 was made
    """
    unchanged = {"label": label, "note": None, "latency_ms": 0, "ran": False}
    cfg = check_config(cam_cfg)
    if cfg is None or label.upper() not in triggers(cfg):
        return unchanged

    prompt = cfg.get("prompt")
    if not prompt:
        return {**unchanged, "note": "vehicle-id=misconfigured(no prompt)"}

    raw, latency_ms = classify_fn(prompt, img_b64)
    if raw is None:
        # Ollama failed on call 2. Call 1 already gave a usable answer; a
        # transport failure must not be allowed to change a label.
        return {**unchanged, "note": "vehicle-id=call-failed",
                "latency_ms": latency_ms, "ran": True}

    outcome = verdict(raw)
    if outcome == "unsure":
        snippet = " ".join((raw or "").split())[:30]
        return {**unchanged, "note": f"vehicle-id=unsure({snippet})",
                "latency_ms": latency_ms, "ran": True}

    key = "known_label" if outcome == "known" else "unknown_label"
    new_label = (cfg.get(key) or label).upper()
    arrow = "" if new_label == label.upper() else f"->{new_label}"
    return {
        "label": new_label,
        "note": f"vehicle-id={outcome}{arrow}",
        "latency_ms": latency_ms,
        "ran": True,
    }
