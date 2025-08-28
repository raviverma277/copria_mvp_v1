from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

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





