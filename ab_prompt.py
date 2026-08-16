#!/usr/bin/env python3
"""
A/B two camera configs against the SAME stored alert frames -- the prompt tool.

replay.py answers "what would these WEIGHTS have done?" by re-scoring labels
already in the database. It structurally cannot answer "what would this PROMPT
have done?", because a prompt change alters which labels get produced in the
first place -- the thing replay treats as fixed input. Answering that needs the
original images and a second trip through the vision model. That is this tool.

    events.db gives (image_path, the label the live prompt produced)
    config A + config B give two prompts
    -> run both over the same frames, diff the labels, print the disagreements

WHERE TO RUN THIS: on the machine holding the alert frames (the BlueIris box),
or anywhere ALERT_IMAGE_DIR can be pointed at a copy of them -- resolve_image()
falls back to a basename lookup, so an archived frame put back in that folder
becomes an A/B candidate again, with its original label as a free third opinion.

If BlueIris is set to PRUNE its alert folder on a size/age cap, the usable
window is only however far back the JPEGs still go: frames age out, event rows
do not. Setting BlueIris to move alerts to a backup folder instead removes that
ceiling and lets the corpus accumulate, which is what makes a stable regression
set possible.

Usage:
    # what did today's rewrite change, on the camera it targeted?
    python ab_prompt.py --a cameras-baseline-20260725.yaml --b cameras.yaml \
        --camera driveway-cam --limit 40

    # cheaper: trust the stored label as the "A" side (it WAS produced by the
    # old prompt) and only pay for the B run -- half the model calls
    python ab_prompt.py --b cameras.yaml --stored-as-a --camera driveway-cam --limit 60

    # the frames lost to the 07-25 Ollama outage have images but no label:
    python ab_prompt.py --a cameras-baseline-20260725.yaml --b cameras.yaml \
        --camera driveway-cam --since "2026-07-25 08:00" --until "2026-07-25 15:00"

    # measure the NOISE FLOOR before trusting any of the above: same config on
    # both sides, drawn 5x per frame. Whatever it disagrees with itself about,
    # a real A/B cannot attribute to a prompt
    python ab_prompt.py --a cameras.yaml --b cameras.yaml --camera driveway-cam \
        --repeat 5

    # score against GROUND TRUTH rather than agreement. Needed whenever the
    # defect is "the live prompt thinks nothing is there", because selecting
    # frames by their stored label can only return frames it already found
    # interesting -- and two configs can agree perfectly while both are wrong
    python ab_prompt.py --a cameras.yaml --b candidate.yaml \
        --frames-file docs/corpus/driveway-groundtruth.txt --repeat 3

The model samples at temperature 0.1, not 0, so one call is a sample and not
"the answer". Measured on a driveway camera: the live config, re-run on
identical images, agreed with itself on only 75% of frames -- and all of the
spread was in one label family, while person/vehicle labels never wavered. An A/B whose
disagreement rate sits under that floor is measuring the sampler, not the
prompt. --repeat is how you find out which you are looking at.

Strictly read-only: opens events.db with mode=ro, writes no files, sends no
email. Safe to run against the live database while BlueIris keeps firing.
"""

import argparse
import base64
import sqlite3
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

import cam_watcher


def open_ro(db_path: Path) -> sqlite3.Connection:
    """Read-only handle. NOT db.connect() -- that one creates/migrates schema,
    and a tuning tool has no business writing to the production event log."""
    return sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)


def candidates(conn, camera=None, since=None, until=None, labels=None, unlabeled=False):
    """Stored events worth re-running, oldest first:
    (ts_iso, camera, stored_label, image_path).

    labels    -- only frames the live prompt labeled one of these (e.g. the
                 GATE_CLOSED frames you suspect are hiding vehicles)
    unlabeled -- only frames with NO label (Ollama errored at the time); these
                 are pure signal for a prompt test, since nothing was learned
                 from them the first time.
    """
    where, params = ["image_path IS NOT NULL"], []
    if camera:
        where.append("camera = ?")
        params.append(camera)
    if since:
        where.append("ts_iso >= ?")
        params.append(since)
    if until:
        where.append("ts_iso <= ?")
        params.append(until)
    if unlabeled:
        where.append("classification IS NULL")
    elif labels:
        where.append("classification IN (%s)" % ",".join("?" * len(labels)))
        params.extend(labels)
    return conn.execute(
        "SELECT ts_iso, camera, classification, image_path FROM events"
        f" WHERE {' AND '.join(where)} ORDER BY ts_epoch",
        params,
    ).fetchall()


def frames_from_file(conn, path: Path, camera=None):
    """Load a hand-curated corpus: one frame per line, `<basename> [TRUTH]`.

    Selecting by camera/time/label can only ever return what the LIVE prompt
    already thought was interesting, which is useless for a defect whose whole
    shape is "the live prompt calls these frames empty". A ground-truth corpus
    is the other direction: a human opened the JPEG, wrote down what is actually
    in it, and the run is scored against that instead of against agreement.

    Blank lines and `#` comments are ignored. The second field, if present, is
    the true label; frames without one still run, they just don't score. A frame
    with no event row is kept too -- resolve_image() only needs the basename --
    so a corpus can include frames from before the camera was configured.

    Returns (rows, truth_by_basename).
    """
    rows, truth, seen = [], {}, set()
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, label = (f.strip() for f in line.partition(" "))
        if not name.lower().endswith(".jpg"):
            name += ".jpg"
        if name in seen:
            sys.exit(f"{path}:{lineno}: duplicate frame {name}")
        seen.add(name)
        if label:
            truth[name] = label
        row = conn.execute(
            "SELECT ts_iso, camera, classification, image_path FROM events"
            " WHERE image_path LIKE ? ORDER BY ts_epoch LIMIT 1", ("%" + name,),
        ).fetchone()
        if row is None:
            rows.append((f"(no event row) {name}", camera, None, name))
        else:
            rows.append(row)
    if not rows:
        sys.exit(f"{path}: no frames listed")
    return rows, truth


def frame_name(result) -> str:
    """Basename of a result's frame, which is the key a corpus is written in.
    Tolerates a plain string as well as a Path -- compare() yields Paths, but
    callers constructing results by hand should not have to know that."""
    return Path(result["image"]).name


def accuracy(results, truth, side, name, echo=print):
    """How often one config got the RIGHT answer, not merely the same answer.

    Agreement between two configs is the only thing this tool could report
    before a labelled corpus existed, and two configs can agree perfectly while
    both being wrong -- which is exactly what a camera whose defect is "it calls
    everything NONE" will do. Scored per truth label, because an overall number
    is dominated by whichever case is most common (here: an empty driveway).
    """
    scored = [r for r in results if frame_name(r) in truth]
    if not scored:
        return
    right = [r for r in scored if r[side] == truth[frame_name(r)]]
    echo(f"\naccuracy of {side.upper()} ({name}) against ground truth:")
    echo(f"  correct on {len(right)}/{len(scored)} frames "
         f"({100 * len(right) / len(scored):.0f}%)")
    by_truth = {}
    for r in scored:
        want = truth[frame_name(r)]
        n, ok = by_truth.get(want, (0, 0))
        by_truth[want] = (n + 1, ok + (r[side] == want))
    echo("  by true label:")
    for want, (n, ok) in sorted(by_truth.items(), key=lambda kv: -kv[1][0]):
        echo(f"    {want:22} {ok}/{n} correct ({100 * ok / n:.0f}%)")


def truth_table(results, truth, echo=print):
    """Every scored frame, so a wrong answer can be traced to its JPEG."""
    scored = [r for r in results if frame_name(r) in truth]
    if not scored:
        return
    echo("\nground truth, frame by frame  (. = correct, X = wrong):")
    for r in sorted(scored, key=lambda r: r["ts_iso"]):
        want = truth[frame_name(r)]
        marks = "".join("." if r[s] == want else "X" for s in ("a", "b"))
        echo(f"  {marks}  {r['ts_iso']}  want {want:20} "
             f"A {str(r['a']):20} B {str(r['b'])}")
        echo(f"        {r['image'].name}")


def with_images(rows):
    """Drop frames whose JPEG is gone (BlueIris pruned it). Returns
    (kept_rows_with_resolved_path, n_missing)."""
    kept, missing = [], 0
    for ts_iso, camera, label, image_path in rows:
        resolved = cam_watcher.resolve_image(image_path)
        if resolved is None:
            missing += 1
            continue
        kept.append((ts_iso, camera, label, resolved))
    return kept, missing


def spread(rows, limit):
    """Evenly-spaced sample across the whole range, not the newest N.

    Motion re-triggers every ~30s, so the newest N frames are usually one
    episode seen N times -- worthless for measuring a prompt. Striding covers
    the day: different light, different weather, different traffic.
    """
    if limit is None or limit >= len(rows):
        return rows
    if limit <= 0:
        return []
    stride = len(rows) / limit
    return [rows[int(i * stride)] for i in range(limit)]


def classify_with(config: dict, camera: str, img_b64: str):
    """One frame through one config's prompt. Returns (label, description);
    label is None if the camera isn't in that config or the call failed."""
    cfg = config.get(camera)
    if not cfg:
        return None, "(camera not in config)"
    raw, _ = cam_watcher.classify(cfg["prompt"], img_b64)
    if raw is None:
        return None, "(ollama call failed)"
    return cam_watcher.parse_answer(raw)


def draw_n(config: dict, camera: str, img_b64: str, n: int = 1):
    """n independent draws of one config over one frame.

    The Modelfile sets temperature 0.1, not 0, so a single call is a SAMPLE of
    what a config answers, not the answer. Where that sampling spread is wide,
    a one-draw A/B cannot tell a prompt effect from a coin flip. Returns
    (draws, modal_label, description_of_the_modal_draw).
    """
    draws = [classify_with(config, camera, img_b64) for _ in range(n)]
    labels = [label for label, _ in draws]
    # Counter ties resolve to first-seen, so a 1-1 split reports the first draw.
    label = Counter(labels).most_common(1)[0][0]
    desc = next(d for lab, d in draws if lab == label)
    return labels, label, desc


def purity(draws) -> float:
    """Share of draws that agree with the modal one. 1.0 = unanimous."""
    return Counter(draws).most_common(1)[0][1] / len(draws) if draws else 0.0


def compare(rows, cfg_a, cfg_b, stored_as_a=False, repeat=1, echo=print):
    """Run each frame through A and B. Yields one result dict per frame.

    Frames are read and encoded once and reused for every call, so A and B see
    byte-identical input -- the prompt is the only variable.

    repeat > 1 draws each config `repeat` times per frame. The reported label is
    the modal draw; the full draw list rides along in a_draws/b_draws so report()
    can show how self-consistent each config was. --stored-as-a is never
    repeated: the stored label is one historical draw and cannot be re-sampled.
    """
    total, t0 = len(rows), time.monotonic()
    for i, (ts_iso, camera, stored, image_file) in enumerate(rows, 1):
        try:
            img_b64 = base64.b64encode(image_file.read_bytes()).decode()
        except OSError as e:
            echo(f"  [{i}/{total}] {ts_iso} unreadable: {e}")
            continue

        if stored_as_a:
            a_draws, a_label, a_desc = [stored], stored, "(stored)"
        else:
            a_draws, a_label, a_desc = draw_n(cfg_a, camera, img_b64, repeat)
        b_draws, b_label, b_desc = draw_n(cfg_b, camera, img_b64, repeat)

        flag = " " if a_label == b_label else "*"
        shown_a, shown_b = str(a_label), str(b_label)
        if repeat > 1:
            shown_a += "" if stored_as_a else f"({Counter(a_draws)[a_label]}/{len(a_draws)})"
            shown_b += f"({Counter(b_draws)[b_label]}/{len(b_draws)})"
        echo(f"  [{i}/{total}] {flag} {ts_iso}  {shown_a:26} -> {shown_b}")
        if i == 1 and not (stored_as_a and repeat == 1):
            per = time.monotonic() - t0
            echo(f"      (~{per:.0f}s/frame, est. {per * total / 60:.0f} min total)")
        yield {
            "ts_iso": ts_iso, "camera": camera, "stored": stored,
            "image": image_file, "a": a_label, "a_desc": a_desc,
            "b": b_label, "b_desc": b_desc,
            "a_draws": a_draws, "b_draws": b_draws,
        }


def stability(results, side, name, show=8, echo=print):
    """How often one config, re-run on the SAME image, answers the same thing.

    This is the noise floor, and it is the number that makes an A/B readable: a
    disagreement rate below it says nothing about the prompt, because the prompt
    disagrees with itself that often. Broken out per modal label, because the
    spread is usually not uniform -- on a camera where one label is a borderline
    visual call, that label can be near a coin flip while every other label is
    rock solid, and only the per-label view shows it.
    """
    draws = [r for r in results if len(r.get(f"{side}_draws") or []) > 1]
    if not draws:
        return
    n_rep = len(draws[0][f"{side}_draws"])
    unanimous = [r for r in draws if purity(r[f"{side}_draws"]) == 1.0]
    echo(f"\nself-consistency of {side.upper()} ({name}), {n_rep} draws per frame:")
    echo(f"  unanimous on {len(unanimous)}/{len(draws)} frames "
         f"({100 * len(unanimous) / len(draws):.0f}%)")

    by_label = {}
    for r in draws:
        n, u = by_label.get(r[side], (0, 0))
        by_label[r[side]] = (n + 1, u + (purity(r[f"{side}_draws"]) == 1.0))
    echo("  by modal label:")
    for label, (n, u) in sorted(by_label.items(), key=lambda kv: -kv[1][0]):
        echo(f"    {str(label):22} {u}/{n} frames unanimous ({100 * u / n:.0f}%)")

    wobbly = sorted((r for r in draws if purity(r[f"{side}_draws"]) < 1.0),
                    key=lambda r: purity(r[f"{side}_draws"]))
    if wobbly:
        echo(f"\n  frames where {side.upper()} disagreed with itself "
             f"({len(wobbly)}; showing up to {show}) -- these cannot be read as"
             " a prompt effect:")
        for r in wobbly[:show]:
            spread = ", ".join(f"{lab}x{cnt}" for lab, cnt
                               in Counter(r[f"{side}_draws"]).most_common())
            echo(f"    {r['ts_iso']}  {spread}")


def report(results, cfg_a_name, cfg_b_name, show=12, echo=print, truth=None):
    if not results:
        echo("No frames compared.")
        return
    agree = sum(1 for r in results if r["a"] == r["b"])
    n = len(results)
    echo(f"\n{'=' * 70}\n{n} frames  |  A = {cfg_a_name}  ->  B = {cfg_b_name}")
    echo(f"agreement: {agree}/{n} ({100 * agree / n:.0f}%)")

    if truth:
        accuracy(results, truth, "a", cfg_a_name, echo=echo)
        accuracy(results, truth, "b", cfg_b_name, echo=echo)

    stability(results, "a", cfg_a_name, echo=echo)
    stability(results, "b", cfg_b_name, echo=echo)
    floors = [1 - len([r for r in results if purity(r.get(f"{s}_draws") or [1]) == 1.0])
              / n for s in ("a", "b") if len(results[0].get(f"{s}_draws") or []) > 1]
    if floors:
        floor = max(floors)
        seen = 1 - agree / n
        verdict = ("BELOW the noise floor -- this run cannot support a verdict"
                   if seen <= floor else
                   "above the noise floor -- the excess is what the prompt did")
        echo(f"\nA-vs-B disagreement {100 * seen:.0f}% vs noise floor "
             f"{100 * floor:.0f}%: {verdict}")
    echo("")

    echo("label flow (A -> B):")
    flow = Counter((r["a"], r["b"]) for r in results)
    a_labels = Counter(r["a"] for r in results)
    b_labels = Counter(r["b"] for r in results)
    for (a, b), count in flow.most_common():
        mark = "" if a == b else ("   <- NEW under B" if a_labels[b] == 0 else "   <- changed")
        echo(f"  {count:5}  {str(a):22} -> {str(b):22}{mark}")

    only_b = sorted(set(b_labels) - set(a_labels), key=lambda x: (x is None, x))
    only_a = sorted(set(a_labels) - set(b_labels), key=lambda x: (x is None, x))
    if only_b:
        echo("\nlabels B produces that A never did:")
        for lab in only_b:
            echo(f"  {b_labels[lab]:5}  {lab}")
    if only_a:
        echo("\nlabels A produced that B never does:")
        for lab in only_a:
            echo(f"  {a_labels[lab]:5}  {lab}")

    diffs = [r for r in results if r["a"] != r["b"]]
    if diffs:
        echo(f"\ndisagreements ({len(diffs)}; showing up to {show}) -- read these,"
             " they are the whole point:")
        for r in diffs[:show]:
            want = (truth or {}).get(frame_name(r))
            echo(f"\n  {r['ts_iso']}  {r['camera']}"
                 + (f"   [truth: {want}]" if want else ""))
            echo(f"    A {str(r['a']):22} {r['a_desc'][:110]}")
            echo(f"    B {str(r['b']):22} {r['b_desc'][:110]}")
            echo(f"    {r['image']}")

    if truth:
        truth_table(results, truth, echo=echo)
    echo(f"\n{'=' * 70}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Read-only: never writes events.db, never emails.",
    )
    ap.add_argument("--b", default="cameras.yaml", help="config under test (default: cameras.yaml)")
    ap.add_argument("--a", default=None, help="baseline config to compare against")
    ap.add_argument("--stored-as-a", action="store_true",
                    help="use each frame's stored label as the baseline instead of "
                         "re-running config A (half the model calls; valid only while "
                         "the stored labels came from the prompt you mean by 'A')")
    ap.add_argument("--db", default="events.db", help="event log to pull frames from")
    ap.add_argument("--camera", default=None, help="restrict to one camera")
    ap.add_argument("--since", default=None, help="ISO timestamp lower bound, e.g. '2026-07-25 07:20'")
    ap.add_argument("--until", default=None, help="ISO timestamp upper bound")
    ap.add_argument("--label", action="append", dest="labels",
                    help="only frames the live run labeled this (repeatable)")
    ap.add_argument("--unlabeled", action="store_true",
                    help="only frames with no label (lost to an Ollama outage)")
    ap.add_argument("--frames-file", default=None, metavar="PATH",
                    help="hand-curated corpus: one '<basename> [TRUE_LABEL]' per "
                         "line, # comments ignored. Overrides the selection flags "
                         "and --limit, and where a true label is given the run is "
                         "scored for ACCURACY instead of only A-vs-B agreement")
    ap.add_argument("--limit", type=int, default=25,
                    help="frames to sample, spread evenly over the range (default 25; 0 = all)")
    ap.add_argument("--show", type=int, default=12, help="disagreements to print in full")
    ap.add_argument("--repeat", type=int, default=1, metavar="N",
                    help="draw each config N times per frame and report how often it "
                         "agrees with ITSELF (the noise floor). Sampling is not "
                         "deterministic, so a disagreement rate under that floor means "
                         "nothing. Costs N x the model calls; 3-5 is usually enough")
    args = ap.parse_args()
    if args.repeat < 1:
        ap.error("--repeat must be at least 1")

    if not args.stored_as_a and not args.a:
        ap.error("pass --a <config> to compare two configs, or --stored-as-a "
                 "to compare against the labels already in the database")

    cfg_b = yaml.safe_load(Path(args.b).read_text()) or {}
    cfg_a = {} if args.stored_as_a else (yaml.safe_load(Path(args.a).read_text()) or {})
    a_name = "stored labels" if args.stored_as_a else args.a

    conn = open_ro(Path(args.db))
    truth = {}
    if args.frames_file:
        rows, truth = frames_from_file(conn, Path(args.frames_file), args.camera)
    else:
        rows = candidates(conn, args.camera, args.since, args.until, args.labels,
                          args.unlabeled)
    rows, missing = with_images(rows)
    if missing:
        print(f"note: {missing} frames skipped -- image no longer on disk "
              f"(BlueIris prunes its alert folder; event rows outlive the JPEGs)")
    if not rows:
        print("No frames with images match those filters. If you expected some, check "
              "that this is the box with ALERT_IMAGE_DIR on it.")
        return

    # A curated corpus is the sample; striding it would drop the very frames it
    # was assembled to test.
    if not args.frames_file:
        rows = spread(rows, args.limit or None)
    print(f"{len(rows)} frames, {rows[0][0]} .. {rows[-1][0]}")
    if truth:
        print(f"ground truth supplied for {len(truth)} of them")
    print(f"model: {cam_watcher.OLLAMA_MODEL} at {cam_watcher.OLLAMA_URL}")
    if args.repeat > 1:
        sides = 1 if args.stored_as_a else 2
        print(f"repeat: {args.repeat} draws per config per frame "
              f"({len(rows) * args.repeat * sides} model calls)")
        if args.stored_as_a:
            print("note: --stored-as-a is a single historical draw, so only B gets a "
                  "noise floor")
    print()

    results = []
    try:
        for r in compare(rows, cfg_a, cfg_b, args.stored_as_a, args.repeat):
            results.append(r)
    except KeyboardInterrupt:
        print("\ninterrupted -- reporting on what finished")
    report(results, a_name, args.b, args.show, truth=truth)

    lat = [r for r in results if r["b"] is None]
    if lat:
        print(f"warning: {len(lat)} frames got no label from B (model errors) -- "
              f"results are incomplete")


if __name__ == "__main__":
    main()
