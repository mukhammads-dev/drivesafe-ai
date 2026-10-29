"""SQLite log and the end-of-session report.

What is stored: timestamps, levels, scores. What is never stored: video,
frames, or anything a face could be recovered from.
"""
import sqlite3
import time
from datetime import datetime

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    frames      INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    ts          REAL NOT NULL,
    level       TEXT NOT NULL,
    score       INTEGER,
    perclos     REAL,
    detail      TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS samples (
    session_id  INTEGER NOT NULL,
    ts          REAL NOT NULL,
    score       INTEGER,
    perclos     REAL
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_samples_session ON samples(session_id);
"""

SAMPLE_EVERY = 2.0  # seconds


class Session:
    def __init__(self, db_path=None):
        path = db_path or config.DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        self.started = time.time()
        cur = self.db.execute("INSERT INTO sessions (started_at) VALUES (?)", (self.started,))
        self.id = cur.lastrowid
        self.db.commit()
        self.frames = 0
        self._last_sample = 0.0

    def tick(self, reading, now: float = None):
        now = now or time.time()
        self.frames += 1
        if now - self._last_sample >= SAMPLE_EVERY:
            self._last_sample = now
            self.db.execute(
                "INSERT INTO samples (session_id, ts, score, perclos) VALUES (?,?,?,?)",
                (self.id, now, reading.score, round(reading.perclos, 2)),
            )

    def event(self, level: str, reading, detail: str = "", now: float = None):
        now = now or time.time()
        self.db.execute(
            "INSERT INTO events (session_id, ts, level, score, perclos, detail) VALUES (?,?,?,?,?,?)",
            (self.id, now, level, reading.score, round(reading.perclos, 2), detail),
        )
        self.db.commit()

    def close(self):
        self.db.execute("UPDATE sessions SET ended_at = ?, frames = ? WHERE id = ?",
                        (time.time(), self.frames, self.id))
        self.db.commit()

    # -- reading back -------------------------------------------------------
    def summary(self):
        events = self.db.execute(
            "SELECT ts, level, score, perclos, detail FROM events WHERE session_id = ? ORDER BY ts",
            (self.id,)).fetchall()
        samples = self.db.execute(
            "SELECT ts, score, perclos FROM samples WHERE session_id = ? ORDER BY ts",
            (self.id,)).fetchall()
        return {"events": events, "samples": samples,
                "started": self.started, "duration": time.time() - self.started}


def session_list(db_path=None):
    """Every recorded session, newest last."""
    import sqlite3
    db = sqlite3.connect(str(db_path or config.DB_PATH))
    rows = db.execute(
        "SELECT id, started_at, ended_at, frames FROM sessions ORDER BY id").fetchall()
    db.close()
    return rows


def load_summary(session_id, db_path=None):
    """Read one past session back out of the database."""
    import sqlite3
    db = sqlite3.connect(str(db_path or config.DB_PATH))
    row = db.execute("SELECT started_at, ended_at FROM sessions WHERE id = ?",
                     (session_id,)).fetchone()
    if row is None:
        db.close()
        return None
    started, ended = row
    events = db.execute(
        "SELECT ts, level, score, perclos, detail FROM events WHERE session_id = ? ORDER BY ts",
        (session_id,)).fetchall()
    samples = db.execute(
        "SELECT ts, score, perclos FROM samples WHERE session_id = ? ORDER BY ts",
        (session_id,)).fetchall()
    db.close()
    return {"events": events, "samples": samples, "started": started,
            "duration": (ended or started) - started, "id": session_id}


def write_report(summary, out_path=None):
    """Two stacked panels plus a numbers strip - the image that goes in the README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or (config.DOCS_DIR / "report.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = summary["samples"]
    if not samples:
        return None

    t0 = samples[0][0]
    minutes = [(ts - t0) / 60.0 for ts, _, _ in samples]
    scores = [s for _, s, _ in samples]
    perclos = [p for _, _, p in samples]
    span = max(minutes[-1], 0.5)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})

    # --- fatigue score with level bands ---------------------------------
    ax1.axhspan(0, 35, color="#2e7d32", alpha=0.07)
    ax1.axhspan(35, 60, color="#f9a825", alpha=0.10)
    ax1.axhspan(60, 100, color="#c62828", alpha=0.10)
    ax1.plot(minutes, scores, color="#1565c0", linewidth=2.2)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Fatigue score")
    ax1.text(span * 0.995, 17, "normal", ha="right", va="center", fontsize=8, color="#2e7d32")
    ax1.text(span * 0.995, 47, "caution", ha="right", va="center", fontsize=8, color="#b28704")
    ax1.text(span * 0.995, 80, "warning", ha="right", va="center", fontsize=8, color="#c62828")

    for ts, level, _, _, _ in summary["events"]:
        x = (ts - t0) / 60.0
        color = "#c62828" if level == "CRITICAL" else "#ef6c00"
        ax1.axvline(x, color=color, linewidth=1.2, alpha=0.8)
        ax1.annotate(level.title(), xy=(x, 96), fontsize=7, color=color,
                     rotation=90, va="top", ha="right")

    # --- PERCLOS --------------------------------------------------------
    ax2.plot(minutes, perclos, color="#6a1b9a", linewidth=1.8)
    ax2.axhline(config.PERCLOS_WARNING, color="#c62828", linewidth=1,
                linestyle="--", label=f"warning {config.PERCLOS_WARNING:.0f}%")
    ax2.axhline(config.PERCLOS_CAUTION, color="#f9a825", linewidth=1,
                linestyle=":", label=f"caution {config.PERCLOS_CAUTION:.0f}%")
    ax2.set_ylim(0, max(25, max(perclos) * 1.3 if perclos else 25))
    ax2.set_ylabel("PERCLOS %")
    ax2.set_xlabel("Minutes into session")
    ax2.legend(loc="upper right", fontsize=7, frameon=False)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_xlim(0, span)

    # --- header ---------------------------------------------------------
    warnings = [e for e in summary["events"] if e[1] in ("WARNING", "CRITICAL")]
    critical = [e for e in summary["events"] if e[1] == "CRITICAL"]
    headline = (
        f"{summary['duration'] / 60.0:.0f} min   |   "
        f"peak score {max(scores)}   |   "
        f"avg PERCLOS {sum(perclos) / len(perclos):.1f}%   |   "
        f"{len(warnings)} warning(s), {len(critical)} microsleep(s)"
    )
    fig.suptitle("DriveSafe session - " +
                 datetime.fromtimestamp(t0).strftime("%d %b %Y, %H:%M"),
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.06, 0.915, headline, ha="left", fontsize=9, color="#555555")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def recommendation(summary) -> str:
    """One sentence a person can act on."""
    warnings = [e for e in summary["events"] if e[1] in ("WARNING", "CRITICAL")]
    minutes = summary["duration"] / 60.0
    if not warnings:
        return f"No fatigue warnings in {minutes:.0f} minutes."
    first = (warnings[0][0] - summary["started"]) / 60.0
    return (f"{len(warnings)} warning(s) in {minutes:.0f} minutes; the first at "
            f"{first:.0f} min. Consider a break every {max(15, int(first)) } minutes.")
