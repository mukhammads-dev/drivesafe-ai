"""How much the system trusts its own input right now.

A monitoring tool that keeps reporting numbers after it has lost the face is
worse than one that stops: the driver sees a calm green screen and assumes
they are being watched. DriveSafe reports its own blindness instead.
"""
import time
from collections import deque

import cv2
import numpy as np

import config

GOOD, FAIR, POOR = "good", "fair", "poor"


class QualityMeter:
    def __init__(self):
        self._seen = deque()        # (timestamp, face_found)
        self._brightness = deque()  # (timestamp, mean luma)

    def update(self, frame, face, now: float = None):
        now = now or time.time()
        self._seen.append((now, face is not None))

        if face is not None:
            x, y, w, h = face.box
            roi = frame[max(0, y):y + h, max(0, x):x + w]
            if roi.size:
                luma = float(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
                self._brightness.append((now, luma))

        for dq in (self._seen, self._brightness):
            while dq and now - dq[0][0] > config.QUALITY_WINDOW:
                dq.popleft()
        return self.state()

    def face_rate(self) -> float:
        if not self._seen:
            return 0.0
        return sum(1 for _, ok in self._seen if ok) / len(self._seen)

    def brightness(self) -> float:
        if not self._brightness:
            return 0.0
        return float(np.mean([b for _, b in self._brightness]))

    def state(self):
        """Returns (level, reason). The reason is shown to the user verbatim."""
        rate, bright = self.face_rate(), self.brightness()
        if rate < 0.35:
            return POOR, "face not visible"
        if bright and bright < config.QUALITY_MIN_BRIGHTNESS * 0.6:
            return POOR, "too dark"
        if rate < config.QUALITY_MIN_FACE_RATE:
            return FAIR, "face intermittent"
        if bright and bright < config.QUALITY_MIN_BRIGHTNESS:
            return FAIR, "low light"
        return GOOD, ""
