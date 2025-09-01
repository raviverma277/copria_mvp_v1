# tests/test_submission_bundle.py
from core.schemas.contracts import SubmissionBundle, Provenance, IngestSource


def test_checksum_and_idempotency():
    payload = {
        "sov": {"records": [{"id": 1}]},
        "loss_run": None,
        "notes": "n",
        "attachments": [],
    }
    prov = Provenance(run_id="abc123", source=IngestSource.LOCAL)
    bundle = SubmissionBundle(provenance=prov, **payload).finalize()

    assert bundle.provenance.checksum is not None
    key1 = bundle.provenance.idempotency_key()

    # same content -> same key
    prov2 = Provenance(run_id="abc123", source=IngestSource.LOCAL)
    bundle2 = SubmissionBundle(provenance=prov2, **payload).finalize()
    assert key1 == bundle2.provenance.idempotency_key()

    # different run_id -> different key
    prov3 = Provenance(run_id="zzz999", source=IngestSource.LOCAL)
    bundle3 = SubmissionBundle(provenance=prov3, **payload).finalize()
    assert key1 != bundle3.provenance.idempotency_key()

    print("test_submission_bundle.py passed")
