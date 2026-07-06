"""parse_answer: splitting the model's LABEL + description, incl. malformed output."""
import pytest

from cam_watcher import parse_answer


@pytest.mark.parametrize(
    "raw, label, desc_contains",
    [
        ("OPEN\nThe garage door is wide open.", "OPEN", "wide open"),
        ("PERSON_LOITERING\nMan in a hoodie by the garage.", "PERSON_LOITERING", "hoodie"),
        ("NONE\nNothing relevant is visible.", "NONE", "Nothing"),
        ("OPEN", "OPEN", ""),                                # label only
        ("open\nlowercase label", "OPEN", "lowercase"),      # case-normalised
        ("CLOSED: the door is shut", "CLOSED", "the door is shut"),   # inline colon
        ("PACKAGE - a box on the porch", "PACKAGE", "a box on the porch"),  # inline dash
        ("  PERSON  \n  someone at the door  ", "PERSON", "someone at the door"),
        ("", "", ""),                                        # empty
    ],
)
def test_parse_answer(raw, label, desc_contains):
    got_label, got_desc = parse_answer(raw)
    assert got_label == label
    assert desc_contains in got_desc
