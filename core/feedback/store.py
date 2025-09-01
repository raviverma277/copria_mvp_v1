# core/feedback/store.py
from pathlib import Path
from typing import List, Optional, Any, Dict
from dataclasses import asdict, is_dataclass, fields
from datetime import datetime, date
import json

from core.schemas.contracts import RiskFeedback

_FEEDBACK_PATH = Path("data/feedback.jsonl")


def _json_default(o: Any):
    # Serialize datetimes & dates as ISO-8601 strings
    if isinstance(o, (datetime, date)):
        # Keep 'Z' suffix for UTC-naive values to be explicit
        try:
            # If it already has tzinfo, isoformat() includes offset
            return o.isoformat()
        except Exception:
            return str(o)
    # Fallback stringification for any other non-JSON-serializable object
    return str(o)


def _coerce_datetime_fields(obj: Dict[str, Any], cls) -> Dict[str, Any]:
    """If the dataclass declares datetime fields, parse ISO strings back to datetime."""
    try:
        field_types = {f.name: f.type for f in fields(cls)}
    except Exception:
        return obj

    for k, t in field_types.items():
        if k in obj and t is datetime and isinstance(obj[k], str):
            v = obj[k]
            try:
                # Accept 'Z' and offset formats
                if v.endswith("Z"):
                    v = v.replace("Z", "+00:00")
                obj[k] = datetime.fromisoformat(v)
            except Exception:
                # Leave as-is if it can't be parsed
                pass
    return obj


def log_feedback(record: RiskFeedback) -> None:
    """
    Append-only write. Supports dataclass (preferred) and falls back to dict-like.
    Serializes datetimes to ISO-8601.
    """
    _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(record):
        payload = asdict(record)
    elif hasattr(record, "model_dump"):  # future-proof if you switch to Pydantic
        payload = record.model_dump()
    elif isinstance(record, dict):
        payload = record
    else:
        raise TypeError(f"Unsupported record type for RiskFeedback: {type(record)}")

    with _FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def load_feedback(run_id: Optional[str] = None, risk_uid: Optional[str] = None) -> List[RiskFeedback]:
    """
    Read all lines, coerce datetime fields from ISO strings where needed,
    and filter by optional (run_id, risk_uid).
    """
    if not _FEEDBACK_PATH.exists():
        return []

    out: List[RiskFeedback] = []
    with _FEEDBACK_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj = _coerce_datetime_fields(obj, RiskFeedback)
                fb = RiskFeedback(**obj)
                if run_id and fb.run_id != run_id:
                    continue
                if risk_uid and fb.risk_uid != risk_uid:
                    continue
                out.append(fb)
            except Exception:
                # ignore malformed lines to keep log append-only tolerant
                continue
    return out
