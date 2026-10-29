"""The measurement layer: PERCLOS, microsleep, yawns, head nods, and a score.

The one idea that separates this from a blink counter:

    PERCLOS = share of the LAST 60 SECONDS in which the eyes were closed.

A blink closes the eyes for ~0.15 s, which is well under 1% of a minute, so
blinking never raises PERCLOS. Drowsy eyelids hang half shut for seconds at a
time, which does. 15% is the threshold used in driver-fatigue research.

Microsleep is handled separately because it must not wait for a window to
fill: eyes shut for 1.5 s is an emergency on its own.
"""
import time
from collections import deque
from dataclasses import dataclass, field

import config


@dataclass
class Reading:
    """Everything the UI, the alert layer and the log need for one frame."""
    perclos: float = 0.0
    eyes_closed: bool = False
    closed_for: float = 0.0
    microsleep: bool = False
    yawning: bool = False
    yawn_count: int = 0
    nod_count: int = 0
    blink_rate: float = 0.0
    pitch: float = 0.0
    ear: float = 0.0
    mar: float = 0.0
    closed_ear: float = 0.0
    yawn_mar: float = 0.0
    score: int = 0
    warming_up: bool = True
    contributions: dict = field(default_factory=dict)
    suggested_level: str = "NORMAL"


class FatigueMonitor:
    def __init__(self, profile):
        self.p = profile
        self._closed_window = deque()      # (timestamp, closed_bool)
        self._blinks = deque()             # timestamps
        self._yawns = deque()              # timestamps
        self._nods = deque()               # timestamps
        self._closed_since = None
        self._was_closed = False
        self._started = None
        self._mouth_open_since = None
        self._nod_since = None
        self._blink_ready = False
        self._yawn_grace_until = 0.0

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _trim(dq, now, span):
        while dq and now - (dq[0][0] if isinstance(dq[0], tuple) else dq[0]) > span:
            dq.popleft()

    def open_ratio(self, seconds: float, now: float = None) -> float:
        """Share of the last `seconds` with the eyes open.

        Used to let the driver dismiss an alarm by being visibly awake. A
        ratio, not an unbroken streak: an alert person blinks every few
        seconds, and demanding zero blinks made the alarm undismissable.
        """
        now = now or time.time()
        recent = [c for t, c in self._closed_window if now - t <= seconds]
        if len(recent) < 5:
            return 0.0
        return sum(1 for c in recent if not c) / len(recent)

    # -- main ---------------------------------------------------------------
    def update(self, ear: float, mar: float, pitch: float, now: float = None) -> Reading:
        now = now or time.time()
        if self._started is None:
            self._started = now
        r = Reading()
        # PERCLOS is a share of a 60-second window. Reporting it after two
        # frames would mean one closed frame reads as 100%.
        elapsed = now - self._started
        r.warming_up = elapsed < config.PERCLOS_MIN_WINDOW
        r.ear, r.mar = ear, mar
        r.closed_ear, r.yawn_mar = self.p.closed_ear, self.p.yawn_mar
        # The blink-rate window is a full minute, so that signal stays out of
        # the score until a minute has actually passed. Otherwise every
        # session opens with "you are blinking too little".
        self._blink_ready = elapsed >= 60.0

        # --- mouth FIRST --------------------------------------------------
        # The eye logic below needs to know whether a yawn is in progress.
        yawn_now = mar >= self.p.yawn_mar
        if yawn_now:
            if self._mouth_open_since is None:
                self._mouth_open_since = now
            elif now - self._mouth_open_since >= config.YAWN_MIN_SECONDS:
                # count once per opening, not once per frame
                if not self._yawns or now - self._yawns[-1] > config.YAWN_MIN_SECONDS * 2:
                    self._yawns.append(now)
                r.yawning = True
            self._yawn_grace_until = now + config.YAWN_EYE_GRACE
        else:
            self._mouth_open_since = None

        self._trim(self._yawns, now, config.YAWN_WINDOW)
        r.yawn_count = len(self._yawns)
        in_yawn = now < self._yawn_grace_until

        # --- eyes ---------------------------------------------------------
        # Eyes shut during a yawn are not drowsiness.
        closed = (ear < self.p.closed_ear) and not in_yawn
        self._closed_window.append((now, closed))
        self._trim(self._closed_window, now, config.PERCLOS_WINDOW)

        if self._closed_window:
            closed_frames = sum(1 for _, c in self._closed_window if c)
            r.perclos = 100.0 * closed_frames / len(self._closed_window)

        if closed:
            if self._closed_since is None:
                self._closed_since = now
            r.closed_for = now - self._closed_since
        else:
            # a short closure that just ended is a blink
            if self._was_closed and self._closed_since is not None:
                if now - self._closed_since < 0.5:
                    self._blinks.append(now)
            self._closed_since = None

        self._was_closed = closed
        r.eyes_closed = closed
        r.microsleep = r.closed_for >= config.MICROSLEEP_SECONDS

        self._trim(self._blinks, now, 60.0)
        r.blink_rate = float(len(self._blinks))

        # --- head nod -----------------------------------------------------
        r.pitch = pitch
        dropped = ((self.p.pitch_baseline - pitch) * config.NOD_SIGN) >= config.NOD_PITCH_DROP
        if dropped:
            if self._nod_since is None:
                self._nod_since = now
            elif now - self._nod_since >= config.NOD_MIN_SECONDS:
                if not self._nods or now - self._nods[-1] > 3.0:
                    self._nods.append(now)
        else:
            self._nod_since = None

        self._trim(self._nods, now, config.YAWN_WINDOW)
        r.nod_count = len(self._nods)

        # --- score --------------------------------------------------------
        r.contributions, r.score = self._score(r)
        r.suggested_level = self._level(r)
        return r

    def _score(self, r: Reading):
        """Every point is traceable to one signal — that is the point."""
        perclos_pts = min(r.perclos / config.PERCLOS_WARNING, 1.0) * config.W_PERCLOS
        yawn_pts = min(r.yawn_count / 3.0, 1.0) * config.W_YAWN
        nod_pts = min(r.nod_count / 2.0, 1.0) * config.W_NOD

        # Both directions matter: drowsy people blink more, then stop blinking.
        if self._blink_ready:
            base = max(self.p.blink_rate, 1.0)
            drift = abs(r.blink_rate - base) / base
            blink_pts = min(drift, 1.0) * config.W_BLINK_RATE
        else:
            blink_pts = 0.0

        parts = {
            "PERCLOS": round(perclos_pts),
            "Yawns": round(yawn_pts),
            "Head nods": round(nod_pts),
            "Blink rate": round(blink_pts),
        }
        return parts, int(min(100, sum(parts.values())))

    @staticmethod
    def _level(r: Reading) -> str:
        # Microsleep never waits for the window: eyes shut right now is an
        # emergency even one second after startup.
        if r.microsleep:
            return "CRITICAL"
        if r.warming_up:
            return "NORMAL"
        if r.perclos >= config.PERCLOS_WARNING or r.score >= 60:
            return "WARNING"
        if r.perclos >= config.PERCLOS_CAUTION or r.yawn_count >= config.YAWN_CAUTION_COUNT or r.score >= 35:
            return "CAUTION"
        return "NORMAL"
