from pathlib import Path
from typing import List, Optional
import json
from dataclasses import asdict, is_dataclass

from core.schemas.contracts import RiskFeedback

_FEEDBACK_PATH = Path("data/feedback.jsonl")


def log_feedback(record: RiskFeedback) -> None:
    """
    Append-only write. Supports dataclass (preferred) and falls back to dict-like.
    """
    _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(record):
        payload = asdict(record)
    elif hasattr(record, "model_dump"):  # if we ever swap to Pydantic in future
        payload = record.model_dump()
    elif isinstance(record, dict):
        payload = record
    else:
        raise TypeError(f"Unsupported record type for RiskFeedback: {type(record)}")

    with _FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_feedback(run_id: Optional[str] = None, risk_uid: Optional[str] = None) -> List[RiskFeedback]:
    """
    Read all lines, coerce each JSON object into a RiskFeedback dataclass,
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
                fb = RiskFeedback(**obj)  # dataclass construction
                if run_id and fb.run_id != run_id:
                    continue
                if risk_uid and fb.risk_uid != risk_uid:
                    continue
                out.append(fb)
            except Exception:
                # ignore malformed lines (keeps append-only audit tolerant)
                continue
    return out
