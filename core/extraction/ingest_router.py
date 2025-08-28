# core/extraction/ingest_router.py
import os
from typing import Optional
from core.schemas.contracts import SubmissionBundle, Provenance, IngestSource
from core.extraction.cytora_client import fetch_latest as cytora_fetch
from core.utils.events import publish

def select_source() -> IngestSource:
    val = (os.environ.get("INGEST_SOURCE") or os.environ.get("COPRIA_INGEST_SOURCE") or "local").lower()
    try:
        return IngestSource(val)
    except Exception:
        return IngestSource.LOCAL

def from_local(parsed_sov, parsed_loss, notes, attachments, run_id: str) -> SubmissionBundle:
    prov = Provenance(run_id=run_id, source=IngestSource.LOCAL)
    bundle = SubmissionBundle(
        sov=parsed_sov,
        loss_run=parsed_loss,
        notes=notes,
        attachments=attachments or [],
        provenance=prov,
    ).finalize()
    publish("SubmissionReceived", {"provenance": bundle.provenance.__dict__})
    return bundle

def from_cytora(run_id: Optional[str] = None) -> SubmissionBundle:
    bundle = cytora_fetch(run_id=run_id)
    publish("SubmissionReceived", {"provenance": bundle.provenance.__dict__})
    return bundle

def get_bundle_auto(local_args: dict, run_id: str) -> SubmissionBundle:
    """
    Very simple heuristic for AUTO: prefer local if any local inputs exist,
    else fall back to cytora.
    """
    has_local = any(local_args.get(k) for k in ("parsed_sov", "parsed_loss", "attachments"))
    return from_local(**local_args, run_id=run_id) if has_local else from_cytora(run_id=run_id)
