# core/extraction/cytora_client.py
import json, os
from typing import Optional, Dict, Any
from core.schemas.contracts import SubmissionBundle, Provenance, IngestSource

SAMPLE_PATH = os.environ.get("CYTORA_SAMPLE_PATH", "data/sample_cytora_bundle.json")

def fetch_latest(run_id: Optional[str] = None) -> SubmissionBundle:
    """
    Temporary stub: reads a sample bundle JSON from disk.
    Replace later with webhook/queue/pull.
    """
    if not os.path.exists(SAMPLE_PATH):
        # minimal placeholder
        payload = {
            "sov": {"records": []},
            "loss_run": {"records": []},
            "notes": "No sample file found; using empty bundle.",
            "attachments": [],
        }
    else:
        with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)

    prov = Provenance(run_id=run_id or "cytora-sample", source=IngestSource.CYTORA)
    bundle = SubmissionBundle(
        sov=payload.get("sov"),
        loss_run=payload.get("loss_run"),
        notes=payload.get("notes"),
        attachments=payload.get("attachments", []),
        provenance=prov,
    ).finalize()

    return bundle
