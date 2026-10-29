"""Redraw the report for any recorded session.

    python report.py                # list sessions
    python report.py --last         # redraw the most recent one
    python report.py --session 3    # redraw session 3
"""
import argparse
from datetime import datetime

import session as session_mod


def main():
    ap = argparse.ArgumentParser(description="DriveSafe session report")
    ap.add_argument("--session", type=int, help="session id")
    ap.add_argument("--last", action="store_true", help="most recent session")
    ap.add_argument("--out", help="output png path")
    args = ap.parse_args()

    rows = session_mod.session_list()
    if not rows:
        raise SystemExit("No sessions recorded yet. Run app.py first.")

    if not args.session and not args.last:
        print(f"{'id':>3}  {'started':<20} {'minutes':>7} {'frames':>7}")
        for sid, started, ended, frames in rows:
            mins = ((ended or started) - started) / 60.0
            print(f"{sid:>3}  {datetime.fromtimestamp(started):%d %b %Y %H:%M}   "
                  f"{mins:7.1f} {frames:7d}")
        print("\nRedraw one with:  python report.py --session <id>")
        return

    sid = rows[-1][0] if args.last else args.session
    summary = session_mod.load_summary(sid)
    if summary is None:
        raise SystemExit(f"No session with id {sid}")
    if not summary["samples"]:
        raise SystemExit(f"Session {sid} has no samples (it was too short)")

    out = session_mod.write_report(summary, args.out)
    print(session_mod.recommendation(summary))
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
