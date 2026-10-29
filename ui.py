"""The dashboard.

Deliberate choice: the window shows a CROP OF THE EYES, never the whole face.

Two reasons, and both are worth saying out loud in a README. The fatigue
signal lives in the eyelids, so the rest of the face is not needed — and a
tool that watches someone for an hour should show, on its own screen, that it
is not keeping their face.
"""
import math

import cv2
import numpy as np

W, H = 900, 560
BG = (28, 28, 32)
PANEL = (40, 40, 46)
TEXT = (235, 235, 240)
MUTED = (150, 150, 160)
LINE = (62, 62, 70)

LEVEL_COLOR = {
    "NORMAL":   (110, 190, 90),
    "CAUTION":  (60, 190, 240),
    "WARNING":  (60, 110, 240),
    "CRITICAL": (60, 60, 240),
}
QUALITY_COLOR = {"good": (110, 190, 90), "fair": (60, 190, 240), "poor": (120, 120, 130)}

F = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.5, color=TEXT, thick=1):
    cv2.putText(img, s, org, F, scale, color, thick, cv2.LINE_AA)


def _panel(img, x, y, w, h, color=PANEL):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)


def blank():
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = BG
    return canvas


def draw_calibration(frame, face, progress: float, samples: int,
                     instruction: str = "", phase: str = "face"):
    """Shown for the first 30 seconds. The preview is blurred on purpose —
    enough to position yourself, not enough to be a face recording."""
    canvas = blank()
    accent = LEVEL_COLOR["CAUTION"] if phase == "mouth" else LEVEL_COLOR["NORMAL"]
    title = "CALIBRATING - MOUTH" if phase == "mouth" else "CALIBRATING"
    _text(canvas, title, (40, 60), 1.0, TEXT, 2)
    _text(canvas, instruction or "Look at the camera and behave normally.",
          (40, 95), 0.55, accent if phase == "mouth" else MUTED,
          2 if phase == "mouth" else 1)

    if frame is not None:
        thumb = cv2.resize(frame, (360, 270))
        thumb = cv2.GaussianBlur(thumb, (0, 0), 7)
        canvas[130:400, 40:400] = thumb
        if face is not None:
            x, y, w, h = face.box
            sx, sy = 360 / frame.shape[1], 270 / frame.shape[0]
            cv2.rectangle(canvas,
                          (40 + int(x * sx), 130 + int(y * sy)),
                          (40 + int((x + w) * sx), 130 + int((y + h) * sy)),
                          accent, 2)

    _panel(canvas, 440, 130, 420, 270)
    _text(canvas, "Why this step", (460, 165), 0.6, TEXT)
    lines = ([
        "A yawn threshold cannot be guessed",
        "either: how wide a mouth opens differs",
        "as much as eye shape does.",
        "",
        "So the last seconds measure it. Open",
        "wide and hold - the threshold is set",
        "halfway between your resting mouth",
        "and this.",
    ] if phase == "mouth" else [
        "Eye size differs between people, and",
        "glasses and camera angle shift it further.",
        "A fixed threshold would be wrong for you.",
        "",
        "DriveSafe measures YOUR open eye, your",
        "resting mouth and your natural head",
        "angle, then sets every threshold",
        "relative to those numbers.",
    ])
    for i, line in enumerate(lines):
        _text(canvas, line, (460, 200 + i * 22), 0.48, MUTED)

    bx, by, bw, bh = 40, 440, W - 80, 26
    _panel(canvas, bx, by, bw, bh, LINE)
    _panel(canvas, bx, by, int(bw * max(0.0, min(1.0, progress))), bh, accent)
    _text(canvas, f"{int(progress * 100)}%   {samples} samples", (bx, by + bh + 28), 0.55, MUTED)
    _text(canvas, "q  quit", (40, H - 20), 0.45, MUTED)
    return canvas


def draw_monitor(strip, reading, level: str, quality, history, fps: float, sound: bool,
                 awaiting_ack: bool = False, unresponsive: float = 0.0, message: str = ""):
    canvas = blank()
    color = LEVEL_COLOR.get(level, TEXT)
    q_level, q_reason = quality

    # --- eyes ------------------------------------------------------------
    _text(canvas, "EYES", (40, 42), 0.5, MUTED)
    _panel(canvas, 40, 56, 420, 150, (18, 18, 22))
    if strip is not None:
        s = strip[: min(150, strip.shape[0])]
        canvas[56:56 + s.shape[0], 40:40 + s.shape[1]] = s
    else:
        _text(canvas, "no face", (190, 136), 0.6, MUTED)
    cv2.rectangle(canvas, (40, 56), (460, 206), LINE, 1)

    # --- score -----------------------------------------------------------
    _panel(canvas, 490, 56, 370, 150)
    _text(canvas, "FATIGUE", (510, 88), 0.5, MUTED)
    _text(canvas, str(reading.score), (510, 160), 2.2, color, 4)
    _text(canvas, "/100", (640, 160), 0.7, MUTED)
    _text(canvas, level, (700, 100), 0.75, color, 2)
    if reading.microsleep:
        _text(canvas, "MICROSLEEP", (700, 130), 0.55, LEVEL_COLOR["CRITICAL"], 2)

    # --- contributions ----------------------------------------------------
    _text(canvas, "WHERE THE SCORE COMES FROM", (40, 248), 0.5, MUTED)
    y = 270
    for name, pts in reading.contributions.items():
        _text(canvas, name, (40, y + 12), 0.5, TEXT)
        _panel(canvas, 200, y, 300, 16, LINE)
        _panel(canvas, 200, y, int(300 * min(pts / 40.0, 1.0)), 16, color)
        _text(canvas, f"+{pts}", (515, y + 13), 0.5, MUTED)
        y += 26

    # --- numbers ----------------------------------------------------------
    _panel(canvas, 570, 240, 290, 164)
    eye_hot = reading.ear and reading.ear < reading.closed_ear
    mouth_hot = reading.mar and reading.mar >= reading.yawn_mar
    rows = [
        ("PERCLOS", f"{reading.perclos:4.1f} %", False),
        ("Eyes closed", f"{reading.closed_for:4.1f} s", reading.closed_for > 0),
        ("EAR / closed", f"{reading.ear:.3f} / {reading.closed_ear:.3f}", bool(eye_hot)),
        ("MAR / yawn", f"{reading.mar:.3f} / {reading.yawn_mar:.3f}", bool(mouth_hot)),
        ("Yawns (5 min)", str(reading.yawn_count), reading.yawn_count > 0),
        ("Head nods", str(reading.nod_count), reading.nod_count > 0),
        ("Blinks / min", f"{reading.blink_rate:.0f}", False),
        ("Head pitch", f"{reading.pitch:5.1f} deg", False),
    ]
    for i, (k, v, hot) in enumerate(rows):
        _text(canvas, k, (590, 264 + i * 18), 0.44, MUTED)
        _text(canvas, v, (762, 264 + i * 18), 0.44, color if hot else TEXT)

    # --- history ----------------------------------------------------------
    _text(canvas, "LAST 2 MINUTES", (40, 414), 0.5, MUTED)
    gx, gy, gw, gh = 40, 426, W - 80, 70
    _panel(canvas, gx, gy, gw, gh, (22, 22, 26))
    if len(history) > 1:
        pts = []
        for i, s in enumerate(history):
            px = gx + int(gw * i / max(1, len(history) - 1))
            py = gy + gh - int(gh * min(s, 100) / 100.0)
            pts.append((px, py))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    cv2.line(canvas, (gx, gy + gh - int(gh * 0.6)), (gx + gw, gy + gh - int(gh * 0.6)), LINE, 1)

    # --- status bar -------------------------------------------------------
    qc = QUALITY_COLOR.get(q_level, MUTED)
    if reading.warming_up:
        qc = MUTED
        label = "monitoring: warming up (PERCLOS needs 15s)"
    else:
        label = f"monitoring: {q_level}" + (f" ({q_reason})" if q_reason else "")
    cv2.circle(canvas, (48, H - 24), 5, qc, -1)
    _text(canvas, label, (62, H - 20), 0.47, qc)
    _text(canvas, f"{fps:4.1f} fps", (300, H - 20), 0.47, MUTED)
    _text(canvas, f"sound: {'on' if sound else 'off'}", (400, H - 20), 0.47, MUTED)
    _text(canvas, "q quit   r recal   s sound   SPACE ack", (600, H - 20), 0.45, MUTED)

    if reading.suggested_level != "NORMAL" and level != "NORMAL":
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), color, 4)

    if awaiting_ack:
        canvas = _draw_wake_overlay(canvas, unresponsive, message)
    return canvas


def _draw_wake_overlay(canvas, unresponsive: float, message: str):
    """Full-screen pulsing red, shown while the siren is waiting for a reply.

    The alarm cannot be silenced by waiting - only by the driver doing
    something, so the screen says exactly what that something is.
    """
    import time as _time

    pulse = 0.35 + 0.30 * abs(math.sin(_time.time() * 6.0))
    layer = canvas.copy()
    cv2.rectangle(layer, (0, 0), (W, H), LEVEL_COLOR["CRITICAL"], -1)
    canvas = cv2.addWeighted(layer, pulse, canvas, 1 - pulse, 0)

    _text(canvas, "WAKE UP", (W // 2 - 185, H // 2 - 30), 2.6, (255, 255, 255), 6)
    _text(canvas, "press SPACE  -  or hold your eyes open for 3 seconds",
          (W // 2 - 255, H // 2 + 20), 0.62, (255, 255, 255), 2)
    _text(canvas, f"no response for {unresponsive:4.1f}s",
          (W // 2 - 110, H // 2 + 56), 0.55, (255, 255, 255), 1)
    if message:
        _text(canvas, message, (W // 2 - 190, H // 2 + 96), 0.7, (255, 255, 255), 2)
    return canvas
