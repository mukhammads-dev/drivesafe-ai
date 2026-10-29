"""Face detection with YuNet (a small CNN that ships inside OpenCV).

DriveSafe only needs ONE face — the person in front of the camera — so we
always return the largest detection. Nothing about the face is stored:
the frame is used, measured and dropped.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Face:
    raw: np.ndarray  # YuNet row: [x, y, w, h, 5 landmark pairs, score]

    @property
    def box(self) -> tuple:
        x, y, w, h = self.raw[:4].astype(int)
        return int(x), int(y), int(w), int(h)

    @property
    def score(self) -> float:
        return float(self.raw[14])

    @property
    def area(self) -> int:
        _, _, w, h = self.box
        return w * h


class FaceDetector:
    def __init__(self, model_path, score_threshold: float = 0.7):
        self.model = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000
        )
        self._size = None

    def detect(self, frame) -> list:
        h, w = frame.shape[:2]
        if self._size != (w, h):          # setInputSize is not free, so only on change
            self.model.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.model.detect(frame)
        if faces is None:
            return []
        return [Face(row.astype(np.float32)) for row in faces]

    def largest(self, frame):
        """The driver is the closest face. Returns None when nobody is there."""
        faces = self.detect(frame)
        return max(faces, key=lambda f: f.area) if faces else None
