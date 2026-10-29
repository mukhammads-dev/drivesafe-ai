"""68 facial landmarks (OpenCV contrib LBF) and the two ratios we measure.

YuNet gives only 5 points — enough to find a face, not enough to tell whether
an eye is open. LBF fits 68 points inside the detected box, which gives us
6 points per eye and 8 around the inner lips.

    EAR (eye aspect ratio)   - how open an eye is
    MAR (mouth aspect ratio) - how open the mouth is

Both are ratios, so they do not change when the person moves closer to the
camera. That is the whole reason we use ratios instead of pixel distances.
"""
import cv2
import numpy as np

# 68-point model indices
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
OUTER_MOUTH = [48, 50, 51, 52, 54, 56, 57, 58]
EYES_ALL = LEFT_EYE + RIGHT_EYE


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


class LandmarkFitter:
    def __init__(self, model_path):
        if not hasattr(cv2, "face"):
            raise SystemExit(
                "cv2.face is missing — install the contrib build:\n"
                "  pip install opencv-contrib-python"
            )
        self.model = cv2.face.createFacemarkLBF()
        self.model.loadModel(str(model_path))

    def fit(self, frame, face):
        """Returns a (68, 2) float array, or None if the fit fails.

        OpenCV has shipped this result as (1, 68, 2), (68, 2) and nested
        variants across versions, so we flatten to (N, 2) and check the
        count instead of trusting one shape.
        """
        x, y, w, h = face.box
        if w <= 0 or h <= 0:
            return None
        boxes = np.array([[x, y, w, h]], dtype=np.int32)
        ok, result = self.model.fit(frame, boxes)
        if not ok or result is None or len(result) == 0:
            return None
        points = np.asarray(result[0], dtype=np.float32).reshape(-1, 2)
        return points if points.shape[0] >= 68 else None


def eye_aspect_ratio(points, eye_idx) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Open eye ~0.30, fully closed ~0.05-0.10. The exact number differs per
    person, which is why calibration.py measures it instead of assuming it.
    """
    p = [points[i] for i in eye_idx]
    vertical = _dist(p[1], p[5]) + _dist(p[2], p[4])
    horizontal = _dist(p[0], p[3]) + 1e-6
    return vertical / (2.0 * horizontal)


def both_eyes_ear(points) -> float:
    """Average of the two eyes — one eye can be hidden by head rotation."""
    return (eye_aspect_ratio(points, LEFT_EYE) + eye_aspect_ratio(points, RIGHT_EYE)) / 2.0


def mouth_aspect_ratio(points) -> float:
    """MAR from the OUTER lips.

    The inner-lip points collapse to almost nothing on a closed mouth and the
    landmark model tracks them poorly on a wide-open one, which made yawns
    hard to separate from noise. The outer contour has a much wider and more
    reliable range: roughly 0.1 closed, 0.6+ mid-yawn.
    """
    p = points
    vertical = _dist(p[50], p[58]) + _dist(p[51], p[57]) + _dist(p[52], p[56])
    horizontal = _dist(p[48], p[54]) + 1e-6
    return vertical / (3.0 * horizontal)


def eye_strip(frame, points, pad_ratio: float = 0.6, out_width: int = 420):
    """Crop a wide strip containing just the two eyes.

    This is what the UI shows instead of the whole face: the fatigue signal
    lives in the eyes, and cropping keeps the person's face off the screen
    (and out of any demo recording).
    """
    pts = np.array([points[i] for i in EYES_ALL], dtype=np.float32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = x1 - x0, y1 - y0
    px, py = w * 0.12, h * pad_ratio
    x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
    x1 = int(min(frame.shape[1], x1 + px))
    y1 = int(min(frame.shape[0], y1 + py))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame[y0:y1, x0:x1]
    scale = out_width / crop.shape[1]
    return cv2.resize(crop, (out_width, max(1, int(crop.shape[0] * scale))))
