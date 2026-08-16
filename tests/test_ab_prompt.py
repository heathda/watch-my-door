"""ab_prompt: frame selection, even-spread sampling, the A/B loop, and the
read-only guarantee. classify() is mocked -- no Ollama, no images beyond the
few bytes the fixtures write."""
import sqlite3

import pytest

import ab_prompt
import cam_watcher
import db


@pytest.fixture
def seeded_db(temp_db, tmp_path, monkeypatch):
    """An event log whose frames point at real files on disk."""
    monkeypatch.setattr(cam_watcher, "ALERT_IMAGE_DIR", "")  # no bare-name fallback
    conn = db.connect()
    for i in range(6):
        img = tmp_path / f"frame{i}.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")
        db.log_event(conn, "cam", "GATE_CLOSED" if i % 2 else None, "d", "r",
                     10, str(img), alerted=False)
    # A seventh whose image was pruned by BlueIris.
    db.log_event(conn, "cam", "GATE_CLOSED", "d", "r", 10,
                 str(tmp_path / "pruned.jpg"), alerted=False)
    conn.close()
    return temp_db


def test_candidates_filters_and_missing_images_are_dropped(seeded_db):
    conn = ab_prompt.open_ro(seeded_db)
    rows = ab_prompt.candidates(conn, camera="cam")
    assert len(rows) == 7
    kept, missing = ab_prompt.with_images(rows)
    assert len(kept) == 6 and missing == 1

    assert len(ab_prompt.candidates(conn, unlabeled=True)) == 3
    assert len(ab_prompt.candidates(conn, labels=["GATE_CLOSED"])) == 4
    assert ab_prompt.candidates(conn, camera="ghost") == []


def test_open_ro_cannot_write(seeded_db):
    """A tuning tool must not be able to corrupt the production event log."""
    conn = ab_prompt.open_ro(seeded_db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM events")


def test_spread_samples_across_the_range_not_the_tail():
    rows = list(range(100))
    picked = ab_prompt.spread(rows, 5)
    assert picked == [0, 20, 40, 60, 80]          # evenly spaced, not [95..99]
    assert ab_prompt.spread(rows, None) == rows   # no limit -> everything
    assert ab_prompt.spread(rows, 500) == rows    # limit above supply
    assert ab_prompt.spread(rows, 0) == []


def test_compare_runs_both_configs_on_identical_bytes(seeded_db, monkeypatch):
    seen = []

    def fake_classify(prompt, b64):
        seen.append((prompt, b64))
        return ("OLD\nthe gate is closed", 10) if prompt == "A" else ("NEW\na red car", 10)

    monkeypatch.setattr(cam_watcher, "classify", fake_classify)
    conn = ab_prompt.open_ro(seeded_db)
    rows, _ = ab_prompt.with_images(ab_prompt.candidates(conn, camera="cam"))
    cfg_a = {"cam": {"prompt": "A"}}
    cfg_b = {"cam": {"prompt": "B"}}

    results = list(ab_prompt.compare(rows[:2], cfg_a, cfg_b, echo=lambda *a: None))
    assert [(r["a"], r["b"]) for r in results] == [("OLD", "NEW"), ("OLD", "NEW")]
    assert results[0]["b_desc"] == "a red car"
    # Both configs must see byte-identical input -- the prompt is the only variable.
    assert seen[0][1] == seen[1][1]
    assert [p for p, _ in seen] == ["A", "B", "A", "B"]


def test_stored_as_a_skips_the_baseline_call(seeded_db, monkeypatch):
    calls = []
    monkeypatch.setattr(cam_watcher, "classify",
                        lambda prompt, b64: calls.append(prompt) or ("NEW\nx", 10))
    conn = ab_prompt.open_ro(seeded_db)
    rows, _ = ab_prompt.with_images(ab_prompt.candidates(conn, labels=["GATE_CLOSED"]))
    results = list(ab_prompt.compare(rows[:3], {}, {"cam": {"prompt": "B"}},
                                     stored_as_a=True, echo=lambda *a: None))
    assert calls == ["B", "B", "B"]                      # half the model calls
    assert all(r["a"] == "GATE_CLOSED" for r in results)  # baseline from the DB


def test_missing_camera_in_config_is_reported_not_crashed(seeded_db, monkeypatch):
    monkeypatch.setattr(cam_watcher, "classify", lambda p, b: ("X\nx", 1))
    label, desc = ab_prompt.classify_with({}, "cam", "b64")
    assert label is None and "not in config" in desc


def test_repeat_draws_each_config_n_times_and_reports_the_modal_label(
        seeded_db, monkeypatch):
    """The point of --repeat: one call is a sample, not the config's answer."""
    # B is unstable -- 2 of every 3 draws say GATE_CLOSED, the third says NONE.
    b_draws = iter(["GATE_CLOSED", "NONE", "GATE_CLOSED"] * 10)
    monkeypatch.setattr(cam_watcher, "classify", lambda prompt, b64: (
        (f"{next(b_draws)}\nd", 10) if prompt == "B" else ("PERSON\nd", 10)))

    conn = ab_prompt.open_ro(seeded_db)
    rows, _ = ab_prompt.with_images(ab_prompt.candidates(conn, camera="cam"))
    results = list(ab_prompt.compare(rows[:1], {"cam": {"prompt": "A"}},
                                     {"cam": {"prompt": "B"}},
                                     repeat=3, echo=lambda *a: None))
    r = results[0]
    assert len(r["a_draws"]) == 3 and len(r["b_draws"]) == 3
    assert r["a"] == "PERSON"                      # stable config -> unanimous
    assert r["b"] == "GATE_CLOSED"                 # modal, not first-past-the-post
    assert ab_prompt.purity(r["a_draws"]) == 1.0
    assert ab_prompt.purity(r["b_draws"]) == pytest.approx(2 / 3)


def test_stored_as_a_is_never_repeated(seeded_db, monkeypatch):
    """The stored label is one historical draw; it cannot be re-sampled."""
    calls = []
    monkeypatch.setattr(cam_watcher, "classify",
                        lambda prompt, b64: calls.append(prompt) or ("NEW\nx", 10))
    conn = ab_prompt.open_ro(seeded_db)
    rows, _ = ab_prompt.with_images(ab_prompt.candidates(conn, labels=["GATE_CLOSED"]))
    results = list(ab_prompt.compare(rows[:2], {}, {"cam": {"prompt": "B"}},
                                     stored_as_a=True, repeat=4,
                                     echo=lambda *a: None))
    assert calls == ["B"] * 8                       # 2 frames x 4 draws, B only
    assert all(r["a_draws"] == ["GATE_CLOSED"] for r in results)


def test_report_calls_a_run_below_its_own_noise_floor_unreadable(capsys):
    """The guard rail: B differs from A on 1/2 frames, but B disagrees with
    ITSELF on 1/2 frames too, so the run supports no verdict."""
    results = [
        {"ts_iso": "t1", "camera": "cam", "stored": None, "image": "i",
         "a": "GATE_OPEN", "a_desc": "d", "b": "GATE_CLOSED", "b_desc": "d",
         "a_draws": ["GATE_OPEN"] * 4, "b_draws": ["GATE_CLOSED", "GATE_OPEN"] * 2},
        {"ts_iso": "t2", "camera": "cam", "stored": None, "image": "i",
         "a": "PERSON", "a_desc": "d", "b": "PERSON", "b_desc": "d",
         "a_draws": ["PERSON"] * 4, "b_draws": ["PERSON"] * 4},
    ]
    ab_prompt.report(results, "old.yaml", "new.yaml")
    out = capsys.readouterr().out
    assert "unanimous on 1/2 frames (50%)" in out          # B's overall floor
    assert "unanimous on 2/2 frames (100%)" in out         # A is stable
    assert "BELOW the noise floor" in out
    # per-label split: the spread is in the gate label, not in PERSON
    assert "GATE_CLOSED            0/1 frames unanimous (0%)" in out
    assert "PERSON                 1/1 frames unanimous (100%)" in out
    assert "GATE_CLOSEDx2, GATE_OPENx2" in out    # the wobble itself is printed


def test_report_highlights_labels_only_b_produces(capsys):
    results = [
        {"ts_iso": "t", "camera": "cam", "stored": None, "image": "i",
         "a": "GATE_CLOSED", "a_desc": "gate shut", "b": "UNFAMILIAR_VEHICLE",
         "b_desc": "a silver sedan is parked in the driveway"},
        {"ts_iso": "t", "camera": "cam", "stored": None, "image": "i",
         "a": "GATE_CLOSED", "a_desc": "gate shut", "b": "GATE_CLOSED",
         "b_desc": "gate shut, driveway empty"},
    ]
    ab_prompt.report(results, "old.yaml", "new.yaml")
    out = capsys.readouterr().out
    assert "agreement: 1/2 (50%)" in out
    assert "UNFAMILIAR_VEHICLE" in out and "NEW under B" in out
    assert "silver sedan" in out  # the disagreement detail is the payload
