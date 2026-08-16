#!/usr/bin/env python3
"""
Review recent classifications from events.db -- the tuning view.

Shows EVERY classification (not just alerts), so you can see how each camera's
prompt is actually labeling frames and refine the prompt / alert_on lists.

Usage:
    python review.py                 # last 25 events, all cameras
    python review.py 50              # last 50 events, all cameras
    python review.py driveway-cam    # last 25 events for one camera
    python review.py driveway-cam 50 # last 50 for one camera
    python review.py --counts        # label frequency per camera (great for tuning)
    python review.py --unknown       # labels no camera prompt defines (silent misses)
"""

import sys
from pathlib import Path

import yaml

import db
import labels

CONFIG_FILE = Path(__file__).parent / "cameras.yaml"


def parse_args(argv: list[str]) -> tuple[str | None, int, bool, bool]:
    camera, limit, counts, unknown = None, 25, False, False
    for a in argv:
        if a == "--counts":
            counts = True
        elif a == "--unknown":
            unknown = True
        elif a.isdigit():
            limit = int(a)
        else:
            camera = a
    return camera, limit, counts, unknown


def show_counts(conn) -> None:
    """Label frequency per camera -- shows what each prompt is producing."""
    rows = conn.execute(
        """
        SELECT camera, classification, COUNT(*) n, SUM(alerted) alerts
        FROM events
        GROUP BY camera, classification
        ORDER BY camera, n DESC
        """
    ).fetchall()
    cur = None
    for camera, label, n, alerts in rows:
        if camera != cur:
            print(f"\n{camera}")
            cur = camera
        flag = f"  ({alerts} alerted)" if alerts else ""
        print(f"  {n:4}  {label or '(none)'}{flag}")


def show_unknown(conn) -> None:
    """
    Labels in the log that the camera's CURRENT prompt does not define.

    These are the quiet failures: an invented label matches no alert_on entry,
    so the frame is logged and dropped exactly like one that correctly didn't
    alert. Audited against the live prompts rather than the stored note, so it
    covers history written before the check existed.

    Expect retired labels to show up here too (a *_NIGHT label was valid when it
    was emitted). Read the date range: recent dates mean the prompt is inventing
    labels now, old ones are just archaeology.
    """
    try:
        config = yaml.safe_load(CONFIG_FILE.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        print(f"Could not read {CONFIG_FILE}: {e}")
        return

    found = False
    for camera, cfg in config.items():
        if camera.startswith("_") or not isinstance(cfg, dict) or "prompt" not in cfg:
            continue
        known = labels.known_labels(cfg["prompt"])
        if not known:
            print(f"{camera}: no labels could be parsed from the prompt -- skipped")
            continue
        rows = conn.execute(
            """
            SELECT classification, COUNT(*) n, SUM(alerted) alerts,
                   MIN(ts_iso) first, MAX(ts_iso) last
            FROM events
            WHERE camera = ? AND classification IS NOT NULL
            GROUP BY classification ORDER BY n DESC
            """,
            (camera,),
        ).fetchall()
        odd = [r for r in rows if r[0].upper() not in known]
        if not odd:
            continue
        found = True
        print(f"\n{camera}")
        for label, n, alerts, first, last in odd:
            print(
                f"  {n:5}  {label:22} {alerts or 0} alerted"
                f"   {first[:10]} .. {last[:10]}"
            )
    if not found:
        print("No unrecognized labels -- every logged label is defined in its prompt.")


def show_recent(conn, camera: str | None, limit: int) -> None:
    where = "WHERE camera = ?" if camera else ""
    params = (camera, limit) if camera else (limit,)
    rows = conn.execute(
        f"""
        SELECT ts_iso, camera, classification, alerted, latency_ms, description
        FROM events {where}
        ORDER BY id DESC LIMIT ?
        """,
        params,
    ).fetchall()
    for ts, cam, label, alerted, ms, desc in rows:
        mark = "*" if alerted else " "
        ms_s = f"{ms}ms" if ms is not None else "-"
        print(f"{mark} {ts}  {cam:14} {(label or '-'):18} {ms_s:>7}  {desc or ''}")
    print(f"\n({len(rows)} rows; * = emailed)")


def main() -> None:
    camera, limit, counts, unknown = parse_args(sys.argv[1:])
    conn = db.connect()
    if unknown:
        show_unknown(conn)
    elif counts:
        show_counts(conn)
    else:
        show_recent(conn, camera, limit)


if __name__ == "__main__":
    main()
