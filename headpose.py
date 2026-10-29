"""Head pose (pitch / yaw / roll) from 6 landmarks with cv2.solvePnP.

A drowsy head drops forward, then jerks back up. Pitch catches that even
when the eyes stay open, so it is an independent second signal.

Absolute pitch depends on where the camera sits, so DriveSafe never uses the
raw number: calibration records the person's natural pitch and everything
downstream works with the difference from that baseline.
"""
import math

import cv2
import numpy as np

# A generic 3D head, in millimetres. Good enough for relative angles.
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),      # 30 nose tip
    (0.0,   -330.0, -65.0),     # 8  chin
    (-225.0, 170.0, -135.0),    # 36 left eye, outer corner
    (225.0,  170.0, -135.0),    # 45 right eye, outer corner
    (-150.0, -150.0, -125.0),   # 48 mouth, left corner
    (150.0,  -150.0, -125.0),   # 54 mouth, right corner
], dtype=np.float64)

LANDMARK_IDS = [30, 8, 36, 45, 48, 54]


def camera_matrix(frame_shape):
    """Approximate intrinsics: focal length ~= image width, centre = image centre."""
    h, w = frame_shape[:2]
    f = float(w)
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def estimate(points, frame_shape):
    """Returns (pitch, yaw, roll) in degrees, or None when solvePnP fails."""
    image_points = np.array([points[i] for i in LANDMARK_IDS], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix(frame_shape),
        np.zeros((4, 1)),                 # assume no lens distortion
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    return _euler_degrees(rmat)


def _euler_degrees(r):
    """Rotation matrix -> (pitch, yaw, roll) in degrees.

    Done by hand rather than with cv2.RQDecomp3x3: that function has
    returned a different number of values across OpenCV releases, and this
    is six lines of arithmetic.
    """
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:                       # gimbal lock, rare in practice
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0

    pitch, yaw, roll = (math.degrees(a) for a in (pitch, yaw, roll))
    # A head looking straight ahead should read near 0, not near 180.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return pitch, yaw, roll
