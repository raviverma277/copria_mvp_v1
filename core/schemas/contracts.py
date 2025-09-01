# core/schemas/contracts.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Union, Literal
from datetime import datetime
import hashlib
import json

# --- Ingest source (unchanged) ------------------------------------------------

class IngestSource(str, Enum):
    LOCAL = "local"
    CYTORA = "cytora"
    AUTO = "auto"

# --- New: attachment roles & email envelope -----------------------------------

class AttachmentRole(str, Enum):
    """Semantic role of an attachment so the pipeline can route parsers cleanly."""
    SUBMISSION_PDF = "submission_pdf"
    SOV = "sov"
    LOSS_RUN = "loss_run"
    EMAIL = "email"                # the .eml/.msg itself
    EMAIL_ATTACHMENT = "email_attachment"
    OTHER = "other"

@dataclass
class EmailEnvelope:
    """Metadata for an email (when the attachment itself is the email message)."""
    subject: Optional[str] = None
    from_addr: Optional[str] = None
    to: List[str] = field(default_factory=list)
    cc: List[str] = field(default_factory=list)
    sent_at: Optional[str] = None        # ISO 8601 string
    message_id: Optional[str] = None

@dataclass
class AttachmentMeta:
    """
    Rich file metadata, supporting nested attachments via parent_id.
    - For an .eml/.msg: role=EMAIL and fill email_envelope.
    - For files inside that email: set parent_id to the email attachment's id.
    """
    id: str                                # stable within this bundle (e.g., "att-1" or "att-<hash>")
    name: str                              # original filename
    mime_type: Optional[str] = None
    role: AttachmentRole = AttachmentRole.OTHER
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    storage_path: Optional[str] = None     # local path or object key if persisted
    origin: Optional[str] = None           # "upload" | "cytora" | "email"
    parent_id: Optional[str] = None        # link email attachments to the EMAIL attachment
    email_envelope: Optional[EmailEnvelope] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AttachmentMeta":
        """Coerce a plain dict into AttachmentMeta (backward-compatible)."""
        if isinstance(d, AttachmentMeta):
            return d
        # Best-effort enum coercion
        role_val = d.get("role", AttachmentRole.OTHER)
        try:
            role = role_val if isinstance(role_val, AttachmentRole) else AttachmentRole(str(role_val))
        except Exception:
            role = AttachmentRole.OTHER

        env = d.get("email_envelope")
        if env and not isinstance(env, EmailEnvelope):
            env = EmailEnvelope(**env)

        return AttachmentMeta(
            id=d.get("id") or d.get("sha256") or "att-unknown",
            name=d.get("name") or d.get("filename") or "unknown",
            mime_type=d.get("mime_type"),
            role=role,
            size_bytes=d.get("size_bytes"),
            sha256=d.get("sha256"),
            storage_path=d.get("storage_path"),
            origin=d.get("origin"),
            parent_id=d.get("parent_id"),
            email_envelope=env,
        )

# --- Provenance (unchanged behavior) ------------------------------------------

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

# --- Submission bundle (extended but backward-compatible) ---------------------

@dataclass
class SubmissionBundle:
    """Contract-first bundle that the pipeline ingests."""
    sov: Optional[Dict[str, Any]] = None          # parsed SOV (rows/records or normalized)
    loss_run: Optional[Dict[str, Any]] = None     # parsed loss runs
    notes: Optional[str] = None                   # freeform notes/emails
    # Backward-compat: callers can still pass a List[Dict[str, Any]]
    attachments: List[Union[AttachmentMeta, Dict[str, Any]]] = field(default_factory=list)
    provenance: Provenance = field(default_factory=lambda: Provenance(run_id="unknown", source=IngestSource.LOCAL))

    # --- helpers --------------------------------------------------------------

    def _normalize_attachments(self) -> List[AttachmentMeta]:
        """Ensure every attachment is an AttachmentMeta."""
        normalized: List[AttachmentMeta] = []
        for a in self.attachments:
            if isinstance(a, AttachmentMeta):
                normalized.append(a)
            elif isinstance(a, dict):
                normalized.append(AttachmentMeta.from_dict(a))
            else:
                # Unknown type; wrap minimal info
                normalized.append(AttachmentMeta(id="att-unknown", name=str(a)))
        self.attachments = normalized
        return normalized

    def _asdict_payload(self) -> Dict[str, Any]:
        """
        Convert to a dict for checksum calculation but exclude provenance.
        Ensures dataclasses (attachments, envelope) become plain dicts.
        """
        # Make sure attachments are dataclasses first
        self._normalize_attachments()
        d = asdict(self)
        d.pop("provenance", None)
        return d

    def to_dict(self) -> Dict[str, Any]:
        """Public serialization (includes provenance)."""
        # Ensure attachments are normalized before serialization
        self._normalize_attachments()
        return asdict(self)

    @staticmethod
    def compute_checksum(payload: Dict[str, Any]) -> str:
        normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    def finalize(self) -> "SubmissionBundle":
        # Normalize attachments and compute a stable checksum over the payload (excluding provenance)
        payload = self._asdict_payload()
        self.provenance.checksum = self.compute_checksum(payload)
        return self

# ---------------------------
# Adaptive Red-Flag Feedback
# ---------------------------

# Verdict choices captured from the UI
Verdict = Literal["confirm", "dismiss", "downgrade", "upgrade", "needs-more-info"]

@dataclass
class RiskFeedback:
    """
    Minimal, append-only feedback record for a single risk item as seen by an underwriter.
    Stored as JSONL; last record per (run_id, risk_uid) is treated as the current verdict in the UI.
    """
    # identity for joining back to a run + risk
    run_id: str
    risk_uid: str  # stable id you compute when building each risk item

    # who/when (simple for local dev; later can be SSO user)
    user: str = "local-user"
    submitted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # decision
    verdict: Verdict = "confirm"
    rationale: str = ""

    # frozen context for reproducibility/auditing
    context_signature: str = ""           # SHA-256 over (code/title + evidence anchors)
    risk_snapshot: Dict[str, Any] = field(default_factory=dict)  # full risk item as seen at decision time

    # provenance
    source: str = "ui"
    version: str = "v1"


def make_context_signature(risk_item: Dict[str, Any]) -> str:
    """
    Deterministic, order-insensitive signature over the parts of the item that identify its context.
    Safe to display and stable across processes.
    """
    core = {
        "code": risk_item.get("code"),
        "title": risk_item.get("title"),
        "evidence": [
            {"source": e.get("source"), "locator": e.get("locator")}
            for e in (risk_item.get("evidence") or [])
        ],
    }
    blob = json.dumps(core, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

