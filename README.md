# DriveSafe AI — a fatigue meter, not a blink counter

Most webcam drowsiness demos alarm the moment an eye closes, so they fire on
every ordinary blink and get switched off within a minute. **DriveSafe** measures
**PERCLOS** — the share of the last 60 seconds the eyes spent closed — which is
the metric used in driver-fatigue research. A blink lasts ~0.15 s and barely
moves it. Heavy eyelids move it a lot.

It calibrates to the person first, explains every point of the score it shows,
says so when it cannot see properly, and writes a session report at the end.

<!-- docs/demo.gif -->

## What makes it different

| | |
|---|---|
| **PERCLOS, not per-frame** | A 60 s rolling window. Blinking never triggers it; drowsy eyelids do. |
| **Calibrated per person** | 30 s of watching sets *your* open-eye EAR, resting mouth and natural head angle. A fixed EAR threshold is wrong for narrow eyes, glasses or a low camera. |
| **Explainable score** | 0–100, and the screen shows which signal contributed each point. |
| **Admits blindness** | Face lost, too dark, half occluded → it reports `monitoring: poor` instead of a confident wrong answer. |
| **Shows eyes only** | The window displays a crop of the eye region — never the whole face, and no video is ever stored. |
| **Microsleep is separate** | Eyes shut ≥ 1.5 s alarms immediately, bypassing window and cooldown. |

## How it works

```
camera frame
   │
   ▼
YuNet ──► face box          (OpenCV DNN)
   │
   ▼
LBF facemark ──► 68 landmarks
   │
   ├─► EAR  (eyes)   ─┐
   ├─► MAR  (mouth)  ─┤
   └─► solvePnP      ─┤
         pitch        │
                      ▼
            PERCLOS · microsleep · yawns · nods · blink rate
                      │
                      ▼
            score 0–100 + contributions
                      │
                      ▼
      stability → hysteresis → cooldown → alert
                      │
                      ▼
              SQLite  →  session report
```

**EAR** = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖) over the 6 landmarks of each eye.
A ratio, so it does not change when you lean toward the camera.

**PERCLOS** = closed frames ÷ frames in the last 60 s. Caution at 8 %, warning at 15 %.

## Calibration — why it exists

Narrow eyes, glasses, a camera mounted below eye level: all of them shift EAR.
A hard-coded `EAR < 0.25` works for the one person in the demo video and fails
for everyone else. DriveSafe spends 30 seconds measuring the person, stores the
result in `data/profile.json`, and sets every threshold relative to it —
closed means *below 60 % of your own open-eye EAR*.

## Escalation — an alarm a sleeping driver cannot ignore

A single chime is useless twice over: a driver habituates to it within a few
events, and a driver who is actually falling asleep never hears it. Real fleet
systems answer this with a ladder, and so does DriveSafe.

| Step | Trigger | What happens |
|---|---|---|
| 1 · Caution | PERCLOS ≥ 8 %, or 2 yawns | Chime and on-screen text |
| 2 · Warning | PERCLOS ≥ 15 %, or score ≥ 60 | **Spoken** warning — a voice is far harder to tune out than a tone |
| 3 · Critical | Eyes shut ≥ 1.5 s | Siren that **does not stop on its own**, full-screen pulsing red, spoken "wake up" |
| 4 · No reply | 5 s with no acknowledgement | The driver is probably asleep: the alert leaves the cabin and goes to the **dispatcher** (webhook + local log) |

Step 3 is the one that matters. The siren stops only when the driver proves they
are awake — pressing `SPACE`, or holding their eyes open for three seconds. An
alarm that silences itself teaches the driver that silence means safe.

Step 4 reflects how fleets actually work: the intervention that reliably helps is
not a louder noise, it is a human being phoning the driver.

```bash
export DRIVESAFE_WEBHOOK="https://your-api/fatigue-events"   # optional
export DRIVESAFE_DRIVER="driver-01"
```
Without a webhook the alert is still written to `data/notifications.log`, so the
step is demonstrable with no server running.

**Hardware honesty:** a production system also vibrates the seat or the steering
wheel and can slow the vehicle through the CAN bus. A laptop and a webcam cannot.
This project implements the decision logic, not the actuators.

## Not crying wolf

Three rules, all of them there because a naive version fires constantly:

- **stability** — a level must hold for several frames before it counts
- **hysteresis** — leaving WARNING needs PERCLOS to fall well below the entry threshold
- **cooldown** — 20 s of silence after an alert, except for microsleep

The test suite checks the result: 70 seconds of normal blinking fires **zero** alerts.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_models.py        # ~55 MB: YuNet + LBF landmarks

python app.py                    # webcam; calibrates on first run
python app.py --source clip.mp4  # a recorded clip
python app.py --recalibrate      # measure again
python tests/test_fatigue.py     # logic tests, no camera needed
python report.py                 # list recorded sessions
python report.py --last          # redraw the latest report
```

Keys: `q` quit · `r` recalibrate · `s` sound on/off · `SPACE` acknowledge an alarm

## Tests

The central claim is testable without a camera, so it is tested:

```
PERCLOS
  [PASS] normal blinking stays under the caution line  -> PERCLOS 4.2%
  [PASS] drowsy eyelids cross the warning line         -> PERCLOS 30.0%
Microsleep
  [PASS] eyes shut 2 s is critical
  [PASS] eyes shut 1 s is not yet critical
Yawning
  [PASS] a 1.5 s wide-open mouth counts as a yawn
  [PASS] talking (short mouth movements) is not a yawn
Alerting
  [PASS] cooldown keeps 70 s of drowsiness to a few alerts  -> 1 alert
  [PASS] 70 s of normal blinking fires nothing              -> 0 alerts
Escalation ladder
  [PASS] a microsleep arms the siren and waits for a reply
  [PASS] no reply for 5 s escalates to the dispatcher
  [PASS] the dispatcher is told once, not every frame
  [PASS] acknowledging disarms it
```

## Results

<!-- Fill in after recording the 12 evaluation clips. Do not guess these. -->

| | Measured |
|---|---|
| Microsleep episodes detected | — / — |
| False alarms per hour | — |
| Ordinary blinks that raised an alert | — |
| Latency from eye closure to alert | — s |
| FPS (MacBook Air, CPU only) | — |

## Project structure

```
app.py            camera loop and state machine
config.py         every threshold, in one place
detector.py       YuNet face detection
landmarks.py      68 landmarks, EAR, MAR, eye crop
headpose.py       solvePnP → pitch / yaw / roll
calibration.py    30 s personal baseline → data/profile.json
fatigue.py        PERCLOS, microsleep, yawns, nods, score
alert.py          levels, hysteresis, cooldown, the escalation ladder
voice.py          spoken warnings (say / espeak / SAPI)
siren.py          generated two-tone alarm that runs until acknowledged
notify.py         dispatcher webhook + local notification log
quality.py        can the system see? (face rate, brightness)
session.py        SQLite log and the end-of-session report
report.py         redraw the report for any past session
ui.py             dashboard drawing
tests/            logic tests, no camera required
```

## Limitations

- Acted drowsiness is not real drowsiness. Proper evaluation needs a dataset
  such as NTHU-DDD or a supervised road test.
- Sunglasses hide the eyes entirely; the system falls back to head pose and says
  its monitoring quality is poor.
- A plain webcam is unreliable at night. Production fleet systems use IR cameras.
- The head-pose estimate uses a generic 3D head and uncalibrated intrinsics, so
  it is good for relative change, not absolute angles.
- This is an assistive tool. It does not transfer responsibility away from the driver.

## Privacy

No video and no images are written to disk. The database stores timestamps,
levels and scores only. The on-screen preview is a crop of the eye region, so the
face is not shown even to the operator. Everything runs locally; nothing is sent
anywhere.

## Models

- [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) — face detection
- [LBF facemark](https://github.com/kurnianggoro/GSOC2017) — 68 facial landmarks
