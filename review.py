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
"""

import sys
import db


def parse_args(argv: list[str]) -> tuple[str | None, int, bool]:
    camera, limit, counts = None, 25, False
    for a in argv:
        if a == "--counts":
            counts = True
        elif a.isdigit():
            limit = int(a)
        else:
            camera = a
    return camera, limit, counts


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
    camera, limit, counts = parse_args(sys.argv[1:])
    conn = db.connect()
    if counts:
        show_counts(conn)
    else:
        show_recent(conn, camera, limit)


if __name__ == "__main__":
    main()
