# core/utils/events.py
import json, os
from datetime import datetime
from typing import Dict, Any

EVENTS_PATH = os.environ.get("COPRIA_EVENTS_PATH", "data/events.jsonl")


def publish(event_type: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(EVENTS_PATH), exist_ok=True)
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "type": event_type,
        "payload": payload,
    }
    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
