from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal, Dict, Any
import hashlib, json

Severity = Literal["low","medium","high","critical"]

EvidenceRole = Literal["primary", "context"]

class EvidenceRef(BaseModel):
    source: Literal["sov", "loss_run", "notes"]
    locator: str
    snippet: Optional[str] = None
    role: Optional[EvidenceRole] = "primary"   # <-- NEW
    
class RiskItem(BaseModel):
    code: str                       # e.g., "SPRINKLER_ABSENT"
    title: str                      # "No sprinklers at WH-B"
    severity: Severity
    rationale: str                  # short, human-readable justification
    evidence: List[EvidenceRef] = []
    confidence: float = Field(ge=0, le=1, default=0.65)
    tags: List[str] = []            # ["fire","protection","warehouse"]
    rule_hits: List[str] = []       # which deterministic rules triggered
    llm_notes: Optional[str] = None # optional extra notes from LLM (kept short)

    # NEW: stable identifier for feedback/merging
    uid: Optional[str] = None

    @model_validator(mode="after")
    def _auto_uid(self):
        """Compute uid from (code, title, evidence anchors) if not provided."""
        if not self.uid:
            anchors = [(e.source, e.locator) for e in (self.evidence or [])]
            raw = {"code": self.code, "title": self.title, "anchors": anchors}
            s = json.dumps(raw, sort_keys=True, ensure_ascii=False)
            self.uid = hashlib.md5(s.encode("utf-8")).hexdigest()
        return self





