"""Thirty seconds of watching the person, so the thresholds are theirs.

Why this exists: a fixed "eye is closed below EAR 0.25" fails constantly.
Narrow eyes, glasses, a camera mounted low — all shift the number. Nearly
every drowsiness demo on GitHub skips this and then needs its threshold
hand-tuned for the one person in the demo video.

DriveSafe measures the person first and stores the result, so a second run
starts instantly.
"""
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

import config


@dataclass
class Profile:
    open_ear: float          # this person's EAR with eyes comfortably open
    closed_ear: float        # below this we call the eye closed
    rest_mar: float          # mouth at rest
    yawn_mar: float          # above this the mouth counts as wide open
    pitch_baseline: float    # how they naturally hold their head, in degrees
    blink_rate: float        # blinks per minute while alert
    samples: int

    def save(self, path=None):
        path = path or config.PROFILE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path=None):
        path = path or config.PROFILE_PATH
        if not path.exists():
            return None
        try:
            return cls(**json.loads(path.read_text()))
        except (TypeError, ValueError):
            return None

    @classmethod
    def fallback(cls):
        """Used only when someone runs app.py without calibrating."""
        open_ear = config.FALLBACK_OPEN_EAR
        return cls(
            open_ear=open_ear,
            closed_ear=open_ear * config.CLOSED_RATIO,
            rest_mar=0.15,
            yawn_mar=config.FALLBACK_MAR_YAWN,
            pitch_baseline=0.0,
            blink_rate=17.0,
            samples=0,
        )


class Calibrator:
    """Collects samples while the person simply looks at the camera."""

    def __init__(self, seconds: float = None):
        self.seconds = seconds or config.CALIBRATION_SECONDS
        self.started = time.time()
        self.ear, self.mar, self.pitch = [], [], []
        self.open_mouth = []          # second phase: mouth held wide
        self._was_closed = False
        self._blinks = 0

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def progress(self) -> float:
        return min(1.0, self.elapsed / self.seconds)

    @property
    def phase(self) -> str:
        """'face' for the long first part, 'mouth' for the short second one."""
        left = self.seconds - self.elapsed
        return "mouth" if left <= config.MOUTH_PHASE_SECONDS else "face"

    @property
    def instruction(self) -> str:
        if self.phase == "mouth":
            return "Now open your mouth WIDE, like a yawn - and hold it"
        return "Look at the camera and behave normally. Blink as usual."

    @property
    def done(self) -> bool:
        return self.progress >= 1.0 and len(self.ear) >= config.MIN_CALIBRATION_SAMPLES

    def update(self, ear: float, mar: float, pitch: float):
        if self.phase == "mouth":
            # Only the mouth matters here; the person is deliberately gaping,
            # so these frames must not pollute the resting measurements.
            self.open_mouth.append(mar)
            return
        self.ear.append(ear)
        self.mar.append(mar)
        self.pitch.append(pitch)
        # rough blink count: EAR dipping under 75% of the running median
        if len(self.ear) > 20:
            ref = float(np.median(self.ear[-20:]))
            closed = ear < ref * 0.75
            if closed and not self._was_closed:
                self._blinks += 1
            self._was_closed = closed

    def finish(self) -> Profile:
        if not self.ear:
            return Profile.fallback()
        ear = np.asarray(self.ear, dtype=np.float64)
        mar = np.asarray(self.mar, dtype=np.float64)
        pitch = np.asarray(self.pitch, dtype=np.float64)

        # 85th percentile, not the mean: blinks during calibration would drag
        # a mean down and make the closed threshold far too low.
        open_ear = float(np.percentile(ear, 85))
        rest_mar = float(np.median(mar))
        minutes = max(self.elapsed / 60.0, 1e-3)

        # Yawn threshold, measured rather than assumed: halfway between this
        # person's resting mouth and how wide they actually open it.
        if len(self.open_mouth) >= 20:
            wide = float(np.percentile(np.asarray(self.open_mouth), 80))
            yawn_mar = rest_mar + config.YAWN_FRACTION * max(wide - rest_mar, 0.05)
        else:
            yawn_mar = rest_mar + 0.25     # fallback if the second phase was skipped

        return Profile(
            open_ear=round(open_ear, 4),
            closed_ear=round(open_ear * config.CLOSED_RATIO, 4),
            rest_mar=round(rest_mar, 4),
            yawn_mar=round(yawn_mar, 4),
            pitch_baseline=round(float(np.median(pitch)), 2),
            blink_rate=round(self._blinks / minutes, 1),
            samples=len(self.ear),
        )
