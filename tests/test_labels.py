"""
Label extraction and the unknown-label guard.

The guard exists because an invented label is invisible: it matches no alert_on
entry, so the event is logged and dropped exactly like a frame that correctly
didn't alert. These tests pin the two properties that matter -- it catches a
label the prompt never defined, and it fails OPEN when it can't be sure.
"""
import labels

PROMPT = """
    This camera watches a door.

    PRIORITY RULE, read before choosing: label the person, not the door.

    Choose the single best label:
    PERSON (a person is visible, whatever the door is doing),
    DOOR_OPEN (nobody is present and the door is open or partly
    open),
    NONE (nothing relevant visible).
"""


def test_known_labels_finds_every_defined_label():
    assert labels.known_labels(PROMPT) == {"PERSON", "DOOR_OPEN", "NONE"}


def test_known_labels_ignores_prose_in_caps():
    """'PRIORITY RULE' is instruction text, not a label -- only NAME( counts."""
    found = labels.known_labels(PROMPT)
    assert "PRIORITY" not in found and "RULE" not in found


def test_known_labels_spans_a_wrapped_definition():
    """YAML folding breaks definitions across lines; DOOR_OPEN's runs onto a
    second line above and must still be found."""
    assert "DOOR_OPEN" in labels.known_labels(PROMPT)


def test_unknown_label_is_flagged():
    assert labels.is_unknown("PERSON_AT_DOOR", PROMPT) is True


def test_defined_label_is_not_flagged():
    assert labels.is_unknown("DOOR_OPEN", PROMPT) is False


def test_match_is_case_insensitive():
    assert labels.is_unknown("door_open", PROMPT) is False


def test_unparseable_prompt_fails_open():
    """A prompt we can't read labels out of must flag NOTHING. Treating an
    empty set as 'no label is valid' would mark every event unknown."""
    assert labels.known_labels("just some prose, no labels here") == set()
    assert labels.is_unknown("ANYTHING", "just some prose, no labels here") is False


def test_empty_inputs_fail_open():
    assert labels.is_unknown("", PROMPT) is False
    assert labels.is_unknown("PERSON", "") is False
    assert labels.known_labels(None) == set()


def test_live_config_prompts_all_parse():
    """Every real camera prompt must yield labels, or the guard silently does
    nothing for that camera. Also asserts the alert_on labels are among them --
    an alert label the parser can't see would be flagged unknown on every hit."""
    import pathlib

    import yaml

    path = pathlib.Path(__file__).resolve().parent.parent / "cameras.yaml.example"
    config = yaml.safe_load(path.read_text())
    cameras = {k: v for k, v in config.items() if not k.startswith("_")}
    for name, cfg in cameras.items():
        known = labels.known_labels(cfg["prompt"])
        assert known, f"{name}: no labels parsed from prompt"
        for tag in cfg["alert_on"]:
            assert tag.upper() in known, (
                f"{name}: alert_on label {tag!r} not found as a 'LABEL (...)' "
                f"definition -- it would be flagged unknown on every hit"
            )
