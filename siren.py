"""A siren that does not stop on its own.

The third escalation step is the important one. Every other alarm in this
project fires once and goes quiet; this one repeats until the driver proves
they are awake - a key press, or eyes held open for a few seconds.

An alarm that silences itself teaches the driver that silence means safe.

The tone is generated once into data/siren.wav with the standard library, so
the repository carries no audio files.
"""
import array
import math
import platform
import shutil
import subprocess
import threading
import wave

import config

SAMPLE_RATE = 22050
TONE_A, TONE_B = 760.0, 1040.0   # alternating two-tone, like an alarm clock
SEGMENT = 0.18                   # seconds per tone
REPEATS = 4                      # segments per clip -> ~0.72 s


def _build_wav(path):
    """Two-tone square-ish wave. Loud, narrow band, hard to sleep through."""
    frames = array.array("h")
    for seg in range(REPEATS):
        freq = TONE_A if seg % 2 == 0 else TONE_B
        for i in range(int(SAMPLE_RATE * SEGMENT)):
            t = i / SAMPLE_RATE
            value = math.sin(2 * math.pi * freq * t)
            value += 0.35 * math.sin(4 * math.pi * freq * t)   # harmonic = harsher
            # short fade so segments do not click
            fade = min(1.0, (i / (SAMPLE_RATE * 0.01)), ((SAMPLE_RATE * SEGMENT - i) / (SAMPLE_RATE * 0.01)))
            frames.append(int(max(-1.0, min(1.0, value)) * fade * 30000))

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(frames.tobytes())


def _player(path):
    system = platform.system()
    if system == "Darwin" and shutil.which("afplay"):
        return ["afplay", "-v", "1", str(path)]
    for exe in ("paplay", "aplay", "ffplay"):
        if system == "Linux" and shutil.which(exe):
            return ([exe, "-nodisp", "-autoexit", str(path)] if exe == "ffplay"
                    else [exe, str(path)])
    return None


class Siren:
    """Plays until stop() is called. Safe to start twice."""

    def __init__(self, path=None):
        self.path = path or (config.DATA_DIR / "siren.wav")
        if not self.path.exists():
            _build_wav(self.path)
        self.cmd = _player(self.path)
        self._thread = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                if self.cmd:
                    proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)
                    while proc.poll() is None:
                        if self._stop.wait(0.05):
                            proc.terminate()
                            return
                else:
                    print("\a", end="", flush=True)
                    if self._stop.wait(0.7):
                        return
            except Exception:
                if self._stop.wait(0.7):
                    return
            if self._stop.wait(0.15):
                return
