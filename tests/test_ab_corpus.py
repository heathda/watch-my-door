"""Ground-truth corpus loading and accuracy scoring in ab_prompt.py.

The tuning tools are what every prompt verdict rests on, so a bug here is worse
than a bug in the alert path: it does not break anything visibly, it just makes
a wrong config look right. These run offline -- no Ollama, no images.
"""
from pathlib import Path

import pytest

import ab_prompt
import db

# Corpora live under docs/, which is gitignored, so a fresh clone has none.
CORPORA = sorted((Path(__file__).resolve().parent.parent / "docs" / "corpus").glob("*.txt"))


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("CAM_WATCHER_DB", str(tmp_path / "events.db"))
    con = db.connect()
    db.log_event(con, camera="driveway-cam", classification="NONE", description="",
                 raw_response="NONE", latency_ms=1,
                 image_path=r"C:\BlueIris\Alerts\driveway-cam.20260815_1.3-3.jpg", alerted=0)
    con.commit()
    return con


def write(tmp_path, text):
    p = tmp_path / "corpus.txt"
    p.write_text(text)
    return p


def test_reads_labels_and_ignores_comments(conn, tmp_path):
    rows, truth = ab_prompt.frames_from_file(conn, write(tmp_path, (
        "# a comment\n"
        "driveway-cam.20260815_1.3-3.jpg   KNOWN_VEHICLE   # trailing comment\n"
        "\n"
        "driveway-cam.20260815_2.3-3.jpg   NONE\n"
    )), "driveway-cam")
    assert truth == {"driveway-cam.20260815_1.3-3.jpg": "KNOWN_VEHICLE",
                     "driveway-cam.20260815_2.3-3.jpg": "NONE"}
    assert len(rows) == 2


def test_appends_jpg_and_finds_the_event_row(conn, tmp_path):
    """A corpus is written by hand, so the basename may arrive without .jpg --
    the same forgiveness resolve_image() extends to BlueIris."""
    rows, truth = ab_prompt.frames_from_file(
        conn, write(tmp_path, "driveway-cam.20260815_1.3-3  KNOWN_VEHICLE\n"), "driveway-cam")
    _, camera, stored, image_path = rows[0]
    assert "driveway-cam.20260815_1.3-3.jpg" in truth
    assert stored == "NONE" and camera == "driveway-cam"
    assert image_path.endswith("driveway-cam.20260815_1.3-3.jpg")


def test_frame_with_no_event_row_is_still_a_candidate(conn, tmp_path):
    """Event rows are not a prerequisite: resolve_image() works off a basename,
    so a corpus can include frames from before the camera was configured."""
    rows, _ = ab_prompt.frames_from_file(
        conn, write(tmp_path, "driveway-cam.never-logged.3-3.jpg  PERSON\n"), "driveway-cam")
    assert len(rows) == 1 and rows[0][2] is None


def test_duplicate_frame_is_an_error(conn, tmp_path):
    """A frame listed twice would be scored twice and silently skew accuracy."""
    with pytest.raises(SystemExit):
        ab_prompt.frames_from_file(
            conn, write(tmp_path, "a.jpg NONE\na.jpg NONE\n"), "driveway-cam")


def test_empty_corpus_is_an_error(conn, tmp_path):
    with pytest.raises(SystemExit):
        ab_prompt.frames_from_file(conn, write(tmp_path, "# nothing but a comment\n"), None)


def results(*specs):
    return [{"ts_iso": f"t{i}", "image": Path(name), "a": a, "b": b}
            for i, (name, a, b) in enumerate(specs)]


def test_accuracy_scores_against_truth_not_agreement():
    """The point of the whole feature: A and B agree perfectly and are both
    wrong. Agreement says 100%; accuracy says 0%."""
    truth = {"1.jpg": "KNOWN_VEHICLE", "2.jpg": "UNFAMILIAR_VEHICLE"}
    out = []
    ab_prompt.accuracy(results(("1.jpg", "NONE", "NONE"), ("2.jpg", "NONE", "NONE")),
                       truth, "b", "cand", echo=out.append)
    assert "correct on 0/2" in "\n".join(out)


def test_accuracy_breaks_down_per_true_label():
    truth = {"1.jpg": "KNOWN_VEHICLE", "2.jpg": "KNOWN_VEHICLE", "3.jpg": "NONE"}
    out = []
    ab_prompt.accuracy(results(("1.jpg", "NONE", "KNOWN_VEHICLE"),
                               ("2.jpg", "NONE", "NONE"),
                               ("3.jpg", "NONE", "NONE")),
                       truth, "b", "cand", echo=out.append)
    text = "\n".join(out)
    assert "correct on 2/3" in text
    assert "KNOWN_VEHICLE          1/2" in text
    assert "NONE                   1/1" in text


def test_unlabelled_frames_are_not_scored():
    """Frames can ride along in a corpus without a truth label; they must not
    count as wrong."""
    out = []
    ab_prompt.accuracy(results(("1.jpg", "NONE", "NONE"), ("2.jpg", "NONE", "PERSON")),
                       {"1.jpg": "NONE"}, "b", "cand", echo=out.append)
    assert "correct on 1/1" in "\n".join(out)


def test_frame_name_tolerates_a_plain_string():
    assert ab_prompt.frame_name({"image": Path("/x/y/z.jpg")}) == "z.jpg"
    assert ab_prompt.frame_name({"image": "z.jpg"}) == "z.jpg"


@pytest.mark.skipif(not CORPORA, reason="no corpus checked out (docs/ is gitignored)")
@pytest.mark.parametrize("path", CORPORA, ids=lambda p: p.name)
def test_committed_corpora_parse_and_cannot_be_gamed(path, conn):
    """A corpus is the artifact every prompt verdict rests on; a typo in one is
    a silently wrong accuracy number.

    The balance check matters as much as the parse: a corpus made only of the
    frames a candidate is supposed to newly catch rewards a candidate that
    answers 'something is there' to everything. There has to be a way to fail.
    """
    _, truth = ab_prompt.frames_from_file(conn, path, None)
    assert truth, f"{path.name}: no frame carries a true label -- nothing is scored"
    quiet = {label for label in truth.values() if label in ("NONE", "IDLE")}
    assert quiet, f"{path.name}: no non-alerting frames -- a shout-at-everything config scores 100%"
    assert len(set(truth.values())) >= 3, f"{path.name}: too few distinct labels to be a real test"
