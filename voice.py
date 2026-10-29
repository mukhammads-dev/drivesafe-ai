"""Spoken warnings.

A tone is easy to tune out; a human voice is not. The second escalation step
speaks instead of beeping, because a drowsy brain filters repeated beeps long
before it filters speech.

Uses whatever the OS already provides - no extra dependency, no audio assets.
"""
import platform
import shutil
import subprocess

_SYSTEM = platform.system()


def available() -> bool:
    if _SYSTEM == "Darwin":
        return shutil.which("say") is not None
    if _SYSTEM == "Linux":
        return shutil.which("espeak") is not None or shutil.which("espeak-ng") is not None
    return _SYSTEM == "Windows"


def speak(text: str, rate: int = 190):
    """Never blocks the camera loop, never raises."""
    try:
        if _SYSTEM == "Darwin" and shutil.which("say"):
            subprocess.Popen(["say", "-r", str(rate), text],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        if _SYSTEM == "Linux":
            exe = shutil.which("espeak") or shutil.which("espeak-ng")
            if exe:
                subprocess.Popen([exe, "-s", str(rate), text],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
        if _SYSTEM == "Windows":
            subprocess.Popen(
                ["powershell", "-Command",
                 f"Add-Type -AssemblyName System.Speech; "
                 f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    except Exception:
        pass
    print(f"[voice] {text}")
