"""Step four: when the driver does not respond, tell someone who can act.

In real fleets the most effective intervention is not a louder noise - it is
a dispatcher phoning the driver. This module is the hook for that: it POSTs
the event as JSON to a webhook (your own API, a Telegram bot, Slack) and
always appends it to a local log so the feature is demonstrable with no
server running.

Set the target with an environment variable:

    export DRIVESAFE_WEBHOOK="https://example.com/fatigue-events"
    export DRIVESAFE_DRIVER="driver-01"
"""
import json
import ssl
import threading
import time
import urllib.request
from datetime import datetime

import config

LOG_PATH = config.DATA_DIR / "notifications.log"


def _context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _post(payload):
    if not config.WEBHOOK_URL:
        return
    try:
        request = urllib.request.Request(
            config.WEBHOOK_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(request, timeout=5, context=_context())
    except Exception as err:
        _append(f"webhook failed: {type(err).__name__}")


def _append(line: str):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as handle:
        handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {line}\n")


def dispatcher_alert(reading, seconds_unresponsive: float):
    """Fire and forget - the camera loop must not wait on the network."""
    payload = {
        "event": "NO_RESPONSE",
        "driver": config.DRIVER_NAME,
        "ts": time.time(),
        "unresponsive_for": round(seconds_unresponsive, 1),
        "score": reading.score,
        "perclos": round(reading.perclos, 1),
        "eyes_closed_for": round(reading.closed_for, 1),
    }
    _append(f"DISPATCHER ALERT  driver={payload['driver']}  "
            f"unresponsive={payload['unresponsive_for']}s  "
            f"score={payload['score']}  perclos={payload['perclos']}%")
    threading.Thread(target=_post, args=(payload,), daemon=True).start()
    return payload
