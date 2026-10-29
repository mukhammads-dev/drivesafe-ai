"""Turning a per-frame level into alerts a person will not learn to ignore.

Three rules stop the noise, and an escalation ladder stops the opposite
problem - an alarm too weak to wake anyone.

    stability  - a level must hold for a few frames before it counts
    hysteresis - it takes more to enter a level than to stay in it
    cooldown   - after an alert, stay quiet for a while

The ladder:

    CAUTION   chime + on-screen text
    WARNING   spoken warning - a voice is harder to tune out than a tone
    CRITICAL  siren that does NOT stop until the driver acknowledges,
              by pressing a key or holding their eyes open
    NO REPLY  still no acknowledgement after a few seconds: the driver is
              probably asleep, so the alert leaves the cabin and goes to
              the dispatcher

That last step matters. In real fleets the intervention that works is not a
louder noise - it is a human being phoning the driver.
"""
import platform
import subprocess
import time

import config
import notify
import voice
from siren import Siren

LEVELS = ["NORMAL", "CAUTION", "WARNING", "CRITICAL"]
RANK = {name: i for i, name in enumerate(LEVELS)}

MESSAGES = {
    "CAUTION": "Signs of fatigue - consider a break soon",
    "WARNING": "You are drowsy - take a break",
    "CRITICAL": "WAKE UP - eyes closed",
}
SPOKEN = {
    "WARNING": "You are showing signs of fatigue. Take a break.",
    "CRITICAL": "Wake up. Wake up. Pull over.",
}


def _chime():
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["afplay", "/System/Library/Sounds/Sosumi.aiff"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif system == "Windows":
            import winsound
            winsound.Beep(880, 400)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


class AlertManager:
    def __init__(self, sound: bool = True):
        self.sound = sound
        self.level = "NORMAL"
        self.message = ""
        self._candidate = "NORMAL"
        self._streak = 0
        self._last_alert = 0.0

        # escalation state
        self.awaiting_ack = False
        self.ack_source = ""
        self._critical_since = None
        self._dispatcher_sent = False
        self._siren = Siren() if sound else None

    # -- acknowledgement ----------------------------------------------------
    def unresponsive_for(self, now: float = None) -> float:
        if not self.awaiting_ack or self._critical_since is None:
            return 0.0
        return (now or time.time()) - self._critical_since

    def acknowledge(self, source: str = "key"):
        """The driver proved they are awake. Stop the siren."""
        if not self.awaiting_ack:
            return False
        self.awaiting_ack = False
        self.ack_source = source
        self._critical_since = None
        self._dispatcher_sent = False
        if self._siren:
            self._siren.stop()
        return True

    def shutdown(self):
        if self._siren:
            self._siren.stop()

    # -- main ---------------------------------------------------------------
    def update(self, reading, now: float = None):
        """Returns (level, fired). fired is True only on a new alert."""
        now = now or time.time()
        proposed = reading.suggested_level

        # hysteresis: stepping DOWN needs the signals to be clearly lower
        if RANK[proposed] < RANK[self.level]:
            if self.level == "WARNING" and reading.perclos > config.PERCLOS_RELEASE:
                proposed = "WARNING"
            elif reading.score > 30 and self.level in ("WARNING", "CAUTION"):
                proposed = self.level

        # stability: ignore single-frame flickers
        if proposed == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = proposed, 1

        fired = False
        if self._streak >= config.STABLE_FRAMES and proposed != self.level:
            rising = RANK[proposed] > RANK[self.level]
            self.level = proposed
            if rising and proposed != "NORMAL":
                urgent = proposed == "CRITICAL"
                if urgent or now - self._last_alert >= config.ALERT_COOLDOWN:
                    self._last_alert = now
                    self.message = MESSAGES.get(proposed, "")
                    fired = True
                    self._escalate(proposed, now)

        if self.level == "NORMAL" and not self.awaiting_ack:
            self.message = ""

        # step four: nobody responded
        if self.awaiting_ack and not self._dispatcher_sent:
            if self.unresponsive_for(now) >= config.NO_RESPONSE_SECONDS:
                self._dispatcher_sent = True
                notify.dispatcher_alert(reading, self.unresponsive_for(now))
                self.message = "NO RESPONSE - dispatcher alerted"
                if self.sound:
                    voice.speak("No response. Alerting the dispatcher.")

        return self.level, fired

    def _escalate(self, level: str, now: float):
        if level == "CAUTION":
            if self.sound:
                _chime()
            return

        if level == "WARNING":
            if self.sound:
                voice.speak(SPOKEN["WARNING"])
            return

        if level == "CRITICAL":
            self.awaiting_ack = True
            self._critical_since = now
            self._dispatcher_sent = False
            if self.sound:
                self._siren.start()
                voice.speak(SPOKEN["CRITICAL"])
