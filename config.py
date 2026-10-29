"""Every tunable number lives here, so the logic files stay readable.

The thresholds below are *starting points*. Most of them are replaced at
runtime by values measured for the actual person during calibration
(see calibration.py) — a fixed EAR threshold never works across faces,
glasses and camera angles.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

# Pretrained models (download_models.py fetches both)
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
LBF_PATH = MODELS_DIR / "lbfmodel.yaml"

DB_PATH = DATA_DIR / "drivesafe.db"
PROFILE_PATH = DATA_DIR / "profile.json"

# --- Detection -------------------------------------------------------------
DETECTION_SCORE = 0.7      # YuNet confidence needed to accept a face

# --- Calibration -----------------------------------------------------------
CALIBRATION_SECONDS = 30   # total; the last MOUTH_PHASE_SECONDS ask for a wide-open mouth
MOUTH_PHASE_SECONDS = 8    # we cannot guess a yawn threshold, so we measure one
YAWN_FRACTION = 0.5        # yawn starts halfway between resting and wide-open
# "Closed" is defined relative to THIS person's open eye, not an absolute number.
CLOSED_RATIO = 0.60        # closed when EAR < 60% of their own open-eye EAR
MIN_CALIBRATION_SAMPLES = 150

# --- Fallbacks (only used if no calibration profile exists) -----------------
FALLBACK_OPEN_EAR = 0.30
FALLBACK_MAR_YAWN = 0.60

# --- PERCLOS ---------------------------------------------------------------
PERCLOS_WINDOW = 60.0      # seconds; the rolling window PERCLOS is measured over
PERCLOS_MIN_WINDOW = 15.0  # a 60s metric is meaningless in the first seconds:
                           # until this much has been seen, PERCLOS is not trusted
PERCLOS_CAUTION = 8.0      # %
PERCLOS_WARNING = 15.0     # %  industry threshold for drowsiness
PERCLOS_RELEASE = 11.0     # %  drop below this to leave WARNING (hysteresis)

# --- Microsleep ------------------------------------------------------------
MICROSLEEP_SECONDS = 1.5   # eyes continuously closed this long -> immediate alarm

# --- Yawning ---------------------------------------------------------------
YAWN_MIN_SECONDS = 0.8     # mouth open at least this long (so talking does not count)
YAWN_WINDOW = 300.0        # seconds; yawns are counted over the last 5 minutes
YAWN_CAUTION_COUNT = 2
# A yawn squeezes the eyes shut for a second or two. That is not drowsiness,
# and counting it fakes a microsleep and inflates PERCLOS, so eye closure is
# ignored while the mouth is wide open and for a moment afterwards.
YAWN_EYE_GRACE = 1.5

# --- Head nod --------------------------------------------------------------
NOD_PITCH_DROP = 12.0      # degrees below the person's natural pitch
# Pitch sign depends on the camera mount. The live pitch is shown in the UI:
# if a nod makes it go UP instead of down, set this to -1.0.
NOD_SIGN = 1.0
NOD_MIN_SECONDS = 0.5

# --- Fatigue score weights (0..100) ----------------------------------------
# Kept here so the score stays explainable: every point on screen can be
# traced back to one of these.
W_PERCLOS = 40
W_YAWN = 25
W_NOD = 25
W_BLINK_RATE = 10

# --- Alerting --------------------------------------------------------------
ALERT_COOLDOWN = 20.0      # seconds of silence after an alert fires
STABLE_FRAMES = 5          # a state must hold this many frames before it counts

# --- Escalation ------------------------------------------------------------
# A beep that stops by itself teaches the driver that silence means safe, so
# a CRITICAL alarm keeps sounding until they prove they are awake.
ACK_EYES_OPEN = 3.0        # window of open eyes that counts as "I am awake"
# Not an unbroken streak: an awake person blinks every few seconds, so
# requiring zero blinks made the alarm impossible to acknowledge.
ACK_OPEN_RATIO = 0.8
NO_RESPONSE_SECONDS = 5.0  # no acknowledgement this long -> tell the dispatcher

# Where the dispatcher alert goes. Empty = local log only, which is enough
# to demo the feature.  export DRIVESAFE_WEBHOOK="https://..."
WEBHOOK_URL = os.environ.get("DRIVESAFE_WEBHOOK", "")
DRIVER_NAME = os.environ.get("DRIVESAFE_DRIVER", "driver-01")

# --- Monitoring quality ----------------------------------------------------
# The system says "I cannot see properly" instead of guessing.
QUALITY_WINDOW = 10.0      # seconds
QUALITY_MIN_FACE_RATE = 0.70   # face found in at least 70% of frames
QUALITY_MIN_BRIGHTNESS = 40    # mean luma of the face region


def check_models():
    """Fail with a clear message instead of a cryptic OpenCV error."""
    missing = [p.name for p in (YUNET_PATH, LBF_PATH)
               if not p.exists() or p.stat().st_size < 100_000]
    if missing:
        raise SystemExit(
            "Model file(s) missing: " + ", ".join(missing) +
            "\nRun: python download_models.py"
        )
