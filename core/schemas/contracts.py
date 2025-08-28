# core/schemas/contracts.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
import hashlib
import json

class IngestSource(str, Enum):
    LOCAL = "local"
    CYTORA = "cytora"
    AUTO = "auto"

@dataclass
class Provenance:
    run_id: str                          # caller-provided or generated
    source: IngestSource                 # "local" | "cytora" | "auto"
    received_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    checksum: Optional[str] = None       # sha256 of normalized payload
    raw_meta: Dict[str, Any] = field(default_factory=dict)

    def idempotency_key(self) -> str:
        base = f"{self.source}:{self.run_id}:{self.checksum or ''}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

@dataclass
class SubmissionBundle:
    """Contract-first bundle that the pipeline ingests."""
    sov: Optional[Dict[str, Any]] = None          # parsed SOV (rows/records or normalized)
    loss_run: Optional[Dict[str, Any]] = None     # parsed loss runs
    notes: Optional[str] = None                   # freeform notes/emails
    attachments: List[Dict[str, Any]] = field(default_factory=list)  # metadata for files
    provenance: Provenance = field(default_factory=lambda: Provenance(run_id="unknown", source=IngestSource.LOCAL))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def compute_checksum(payload: Dict[str, Any]) -> str:
        normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    def finalize(self) -> "SubmissionBundle":
        payload = {k: v for k, v in self.to_dict().items() if k != "provenance"}
        self.provenance.checksum = self.compute_checksum(payload)
        return self
