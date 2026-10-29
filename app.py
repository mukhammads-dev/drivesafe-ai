"""DriveSafe AI - fatigue monitoring station.

    python app.py                    # webcam
    python app.py --source clip.mp4  # a recorded clip
    python app.py --recalibrate      # measure this person again

Keys:  q quit   r recalibrate   s toggle sound   SPACE acknowledge an alarm
"""
import argparse
import time
from collections import deque

import cv2

import config
import headpose
import session as session_mod
import ui
from alert import AlertManager
from calibration import Calibrator, Profile
from detector import FaceDetector
from fatigue import FatigueMonitor, Reading
from landmarks import LandmarkFitter, both_eyes_ear, eye_strip, mouth_aspect_ratio
from quality import QualityMeter

HISTORY_SECONDS = 120
WINDOW = "DriveSafe AI"


def parse_source(value: str):
    """'0' means the default webcam; anything else is a file path."""
    try:
        return int(value)
    except ValueError:
        return value


def main():
    ap = argparse.ArgumentParser(description="DriveSafe AI - fatigue monitoring")
    ap.add_argument("--source", default="0", help="webcam index or video file")
    ap.add_argument("--recalibrate", action="store_true", help="ignore the saved profile")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    config.check_models()

    source = parse_source(args.source)
    is_webcam = isinstance(source, int)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    detector = FaceDetector(config.YUNET_PATH, config.DETECTION_SCORE)
    fitter = LandmarkFitter(config.LBF_PATH)
    quality = QualityMeter()

    profile = None if args.recalibrate else Profile.load()
    calibrator = None if profile else Calibrator()
    monitor = FatigueMonitor(profile) if profile else None
    alerts = AlertManager(sound=not args.no_sound)
    sess = session_mod.Session()

    history = deque(maxlen=HISTORY_SECONDS)
    last_history_push = 0.0
    last_reading = Reading()
    eyes_open_since = None          # used to let the driver acknowledge by staying awake
    fps, last_t = 0.0, time.time()

    if profile:
        print(f"[profile] open EAR {profile.open_ear}  closed below {profile.closed_ear}  "
              f"({profile.samples} samples)")
    else:
        print(f"[calibration] {config.CALIBRATION_SECONDS}s - look at the camera normally")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_webcam:
                frame = cv2.flip(frame, 1)   # mirror, so it feels like a mirror

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            face = detector.largest(frame)
            points = fitter.fit(frame, face) if face is not None else None
            quality.update(frame, face, now)

            ear = mar = pitch = None
            if points is not None:
                ear = both_eyes_ear(points)
                mar = mouth_aspect_ratio(points)
                pose = headpose.estimate(points, frame.shape)
                pitch = pose[0] if pose else 0.0

            # ---------------- calibration ----------------
            if monitor is None:
                if points is not None:
                    calibrator.update(ear, mar, pitch)
                canvas = ui.draw_calibration(
                    frame, face, calibrator.progress,
                    len(calibrator.ear) + len(calibrator.open_mouth),
                    calibrator.instruction, calibrator.phase)

                if calibrator.progress >= 1.0:
                    if len(calibrator.ear) >= config.MIN_CALIBRATION_SAMPLES:
                        profile = calibrator.finish()
                        profile.save()
                        monitor = FatigueMonitor(profile)
                        print(f"[profile] saved - open EAR {profile.open_ear}, "
                              f"closed below {profile.closed_ear}, "
                              f"yawn above MAR {profile.yawn_mar}, "
                              f"blink rate {profile.blink_rate}/min")
                    else:
                        print("[calibration] not enough samples, restarting")
                        calibrator = Calibrator()

            # ---------------- monitoring ----------------
            else:
                if points is not None:
                    last_reading = monitor.update(ear, mar, pitch, now)
                    level, fired = alerts.update(last_reading, now)
                    sess.tick(last_reading, now)
                    if fired:
                        sess.event(level, last_reading, alerts.message, now)
                        print(f"[{time.strftime('%H:%M:%S')}] {level}: {alerts.message}")

                    # Acknowledge by staying awake: eyes open without a break.
                    if last_reading.eyes_closed:
                        eyes_open_since = None
                    elif eyes_open_since is None:
                        eyes_open_since = now
                    if (alerts.awaiting_ack and eyes_open_since is not None
                            and now - eyes_open_since >= config.ACK_EYES_OPEN):
                        if alerts.acknowledge("eyes open"):
                            sess.event("ACK", last_reading, "acknowledged by eyes open", now)
                            print(f"[{time.strftime('%H:%M:%S')}] acknowledged (eyes open)")
                else:
                    # No face: do not guess. The quality meter says why.
                    level = alerts.level
                    eyes_open_since = None

                if now - last_history_push >= 1.0:
                    last_history_push = now
                    history.append(last_reading.score)

                strip = eye_strip(frame, points) if points is not None else None
                canvas = ui.draw_monitor(
                    strip, last_reading, level, quality.state(), list(history), fps,
                    alerts.sound,
                    awaiting_ack=alerts.awaiting_ack,
                    unresponsive=alerts.unresponsive_for(now),
                    message=alerts.message,
                )

            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                calibrator, monitor = Calibrator(), None
                print("[calibration] restarted")
            if key == ord("s"):
                alerts.sound = not alerts.sound
            if key == ord(" ") and alerts.acknowledge("key press"):
                sess.event("ACK", last_reading, "acknowledged by key", time.time())
                print(f"[{time.strftime('%H:%M:%S')}] acknowledged (key)")

    finally:
        alerts.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        sess.close()
        summary = sess.summary()
        print("\n" + session_mod.recommendation(summary))
        if not args.no_report and summary["samples"]:
            path = session_mod.write_report(summary)
            if path:
                print(f"Report written to {path}")


if __name__ == "__main__":
    main()
