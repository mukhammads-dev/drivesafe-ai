"""Tests for the measurement logic - no camera, no model files needed.

These exist because the project's central claim is testable:
blinking must NOT raise PERCLOS, and drowsy eyelids must.

    python tests/test_fatigue.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                      # noqa: E402
from alert import AlertManager     # noqa: E402
from calibration import Calibrator, Profile  # noqa: E402
from fatigue import FatigueMonitor  # noqa: E402

FPS = 30.0
STEP = 1.0 / FPS

OPEN, CLOSED = 0.30, 0.10
RESTING_MOUTH, OPEN_MOUTH = 0.15, 0.70


def make_profile():
    return Profile(open_ear=OPEN, closed_ear=OPEN * config.CLOSED_RATIO,
                   rest_mar=RESTING_MOUTH, yawn_mar=0.45,
                   pitch_baseline=0.0, blink_rate=15.0, samples=900)


def run(monitor, seconds, ear_fn, mar_fn=None, pitch_fn=None, t0=1000.0):
    """Feed the monitor a synthetic signal and return the last reading."""
    reading = None
    frames = int(seconds * FPS)
    for i in range(frames):
        t = t0 + i * STEP
        reading = monitor.update(
            ear_fn(t - t0),
            mar_fn(t - t0) if mar_fn else RESTING_MOUTH,
            pitch_fn(t - t0) if pitch_fn else 0.0,
            now=t,
        )
    return reading


def blinking(period=4.0, duration=0.15):
    """A normal blink: eyes shut ~0.15 s every few seconds."""
    return lambda s: CLOSED if (s % period) < duration else OPEN


def drowsy(period=4.0, duration=1.2):
    """Heavy eyelids: shut for over a second at a time."""
    return lambda s: CLOSED if (s % period) < duration else OPEN


def check(name, condition, detail=""):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -> {detail}" if detail else ""))
    return condition


def main():
    failures = 0
    print("\nPERCLOS")

    r = run(FatigueMonitor(make_profile()), 70, blinking())
    failures += not check("normal blinking stays under the caution line",
                          r.perclos < config.PERCLOS_CAUTION, f"PERCLOS {r.perclos:.1f}%")
    failures += not check("normal blinking is not an alert",
                          r.suggested_level == "NORMAL", r.suggested_level)
    failures += not check("blink rate is measured",
                          10 <= r.blink_rate <= 20, f"{r.blink_rate:.0f}/min")

    r = run(FatigueMonitor(make_profile()), 70, drowsy())
    failures += not check("drowsy eyelids cross the warning line",
                          r.perclos >= config.PERCLOS_WARNING, f"PERCLOS {r.perclos:.1f}%")
    failures += not check("drowsy state is escalated",
                          r.suggested_level in ("WARNING", "CRITICAL"), r.suggested_level)

    print("\nMicrosleep")
    m = FatigueMonitor(make_profile())
    r = run(m, 2.0, lambda s: CLOSED)
    failures += not check("eyes shut 2 s is critical",
                          r.microsleep and r.suggested_level == "CRITICAL",
                          f"closed_for {r.closed_for:.1f}s")
    r = run(FatigueMonitor(make_profile()), 1.0, lambda s: CLOSED)
    failures += not check("eyes shut 1 s is not yet critical", not r.microsleep)

    print("\nYawning")
    m = FatigueMonitor(make_profile())
    r = run(m, 3.0, lambda s: OPEN, mar_fn=lambda s: OPEN_MOUTH if s < 1.5 else RESTING_MOUTH)
    failures += not check("a 1.5 s wide-open mouth counts as a yawn",
                          r.yawn_count == 1, f"{r.yawn_count}")

    m = FatigueMonitor(make_profile())
    r = run(m, 6.0, lambda s: OPEN,
            mar_fn=lambda s: OPEN_MOUTH if (s % 1.0) < 0.3 else RESTING_MOUTH)
    failures += not check("talking (short mouth movements) is not a yawn",
                          r.yawn_count == 0, f"{r.yawn_count}")

    print("\nYawning does not fake a microsleep")
    m = FatigueMonitor(make_profile())
    r = run(m, 1.9, lambda s: CLOSED, mar_fn=lambda s: OPEN_MOUTH)
    failures += not check("eyes shut during a yawn is not a microsleep",
                          not r.microsleep, f"closed_for {r.closed_for:.1f}s")
    failures += not check("the yawn itself is still counted", r.yawn_count == 1, f"{r.yawn_count}")

    # a yawn every 10 s, eyes shut through each one
    m = FatigueMonitor(make_profile())
    r = run(m, 70,
            lambda s: CLOSED if (s % 10.0) < 2.0 else OPEN,
            mar_fn=lambda s: OPEN_MOUTH if (s % 10.0) < 2.0 else RESTING_MOUTH)
    failures += not check("repeated yawns do not inflate PERCLOS",
                          r.perclos < config.PERCLOS_CAUTION, f"PERCLOS {r.perclos:.1f}%")

    print("\nAcknowledging while blinking normally")
    m = FatigueMonitor(make_profile())
    run(m, 2.0, lambda s: CLOSED, t0=1000.0)                 # microsleep
    run(m, 4.0, blinking(), t0=1002.0)                       # awake again
    ratio = m.open_ratio(config.ACK_EYES_OPEN, now=1006.0)
    failures += not check("an awake, blinking driver can dismiss the alarm",
                          ratio >= config.ACK_OPEN_RATIO, f"open {ratio * 100:.0f}% of 3 s")
    m2 = FatigueMonitor(make_profile())
    run(m2, 4.0, lambda s: CLOSED, t0=2000.0)
    failures += not check("a driver with shut eyes cannot",
                          m2.open_ratio(config.ACK_EYES_OPEN, now=2004.0) < config.ACK_OPEN_RATIO)

    print("\nHead nod")
    m = FatigueMonitor(make_profile())
    r = run(m, 3.0, lambda s: OPEN, pitch_fn=lambda s: -20.0 if s < 1.0 else 0.0)
    failures += not check("head dropping 20 deg counts as a nod", r.nod_count == 1, f"{r.nod_count}")

    print("\nScore is explainable")
    failures += not check("every point traces to a named signal",
                          sum(r.contributions.values()) == r.score,
                          f"{r.contributions} = {r.score}")

    print("\nAlerting")
    m = FatigueMonitor(make_profile())
    alerts = AlertManager(sound=False)
    fired, t = 0, 2000.0
    for i in range(int(70 * FPS)):
        t += STEP
        reading = m.update(drowsy()(i * STEP), RESTING_MOUTH, 0.0, now=t)
        _, did = alerts.update(reading, now=t)
        fired += int(did)
    failures += not check("cooldown keeps 70 s of drowsiness to a few alerts",
                          1 <= fired <= 4, f"{fired} alerts")

    m2 = FatigueMonitor(make_profile())
    quiet = AlertManager(sound=False)
    fired2, t = 0, 3000.0
    for i in range(int(70 * FPS)):
        t += STEP
        reading = m2.update(blinking()(i * STEP), RESTING_MOUTH, 0.0, now=t)
        _, did = quiet.update(reading, now=t)
        fired2 += int(did)
    failures += not check("70 s of normal blinking fires nothing", fired2 == 0, f"{fired2} alerts")

    print("\nEscalation ladder")
    import notify
    sent = []
    notify.dispatcher_alert = lambda reading, secs: sent.append(secs)

    m3 = FatigueMonitor(make_profile())
    ladder = AlertManager(sound=False)
    t = 4000.0
    # eyes shut and staying shut
    for _ in range(int(3.0 * FPS)):
        t += STEP
        reading = m3.update(CLOSED, RESTING_MOUTH, 0.0, now=t)
        ladder.update(reading, now=t)
    failures += not check("a microsleep arms the siren and waits for a reply",
                          ladder.awaiting_ack, ladder.level)
    failures += not check("no dispatcher alert yet", not sent)

    for _ in range(int(4.0 * FPS)):      # still no reply
        t += STEP
        reading = m3.update(CLOSED, RESTING_MOUTH, 0.0, now=t)
        ladder.update(reading, now=t)
    failures += not check("no reply for 5 s escalates to the dispatcher",
                          len(sent) == 1, f"{len(sent)} alert(s)")

    for _ in range(int(3.0 * FPS)):      # still shut - must not spam
        t += STEP
        ladder.update(m3.update(CLOSED, RESTING_MOUTH, 0.0, now=t), now=t)
    failures += not check("the dispatcher is told once, not every frame",
                          len(sent) == 1, f"{len(sent)} alert(s)")

    failures += not check("acknowledging disarms it", ladder.acknowledge("test"))
    failures += not check("and it stays disarmed", not ladder.awaiting_ack)

    print("\nCalibration")
    cal = Calibrator(seconds=0.1)
    blink = blinking()
    for i in range(900):
        cal.update(blink(i * STEP), RESTING_MOUTH, 2.0)
    p = cal.finish()
    failures += not check("open EAR is found despite blinks",
                          abs(p.open_ear - OPEN) < 0.02, f"{p.open_ear}")
    failures += not check("closed threshold sits between open and closed",
                          CLOSED < p.closed_ear < OPEN, f"{p.closed_ear}")

    print(f"\n{'ALL PASSED' if failures == 0 else str(failures) + ' FAILED'}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
