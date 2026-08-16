#!/usr/bin/env python3
"""
Which labels a camera prompt actually offers.

The model is *told* to answer with one of the labels its prompt defines, but
nothing enforces it -- and an invented label is a silent failure: it matches no
`alert_on` entry, so the event is logged, dropped, and never alerts. The frame
that should have emailed you looks identical in the database to one that
correctly didn't.

This is not hypothetical. In production it is rare but real -- BIRD_FLYING,
MOWING_LAWN and MOWING between them account for five events no prompt ever
offered, each one logged and dropped. A bad prompt revision makes it common: a
single 104-frame A/B on 2026-07-26 produced PERSON_ON_PROPERTY, PERSON_IN_YARD
and PERSON_PASSING_BY (2.9% of the run), all three being the prompt's own
numbered-step text read back as a label. PERSON_ON_PROPERTY is exactly the case
that should have emailed.

Every camera prompt defines its labels the same way -- an UPPER_SNAKE_CASE name
followed by a parenthesised definition -- so the permitted set is derived from
the prompt itself. That means it cannot drift out of sync with the prompt the
way a hand-maintained list in cameras.yaml would.
"""

import re

# LABEL ( ... ) -- the definition form every camera prompt uses. Three chars
# minimum so prose like "IR" or "US" can't be mistaken for a label.
_LABEL_DEF = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*\(")


def known_labels(prompt: str) -> set[str]:
    """
    The labels a prompt defines, found as `LABEL (definition)` pairs.

    Returns an empty set if the prompt doesn't use that convention. Callers MUST
    treat empty as "cannot tell" and skip the check -- never as "no labels are
    valid", which would flag every event.
    """
    return set(_LABEL_DEF.findall(prompt or ""))


def is_unknown(label: str, prompt: str) -> bool:
    """
    True only when we can positively say this prompt never offered this label.

    Deliberately fails open: an unparseable prompt, or an empty label, is not
    flagged. A false "unknown-label" note would send someone chasing a prompt
    bug that isn't there, which is worse than missing one.
    """
    known = known_labels(prompt)
    if not known or not label:
        return False
    return label.upper() not in known

