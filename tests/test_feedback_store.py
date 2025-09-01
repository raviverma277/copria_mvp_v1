from core.schemas.contracts import RiskFeedback
from core.feedback.store import log_feedback, load_feedback
from datetime import datetime
import uuid


def test_log_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rid = str(uuid.uuid4())
    fb = RiskFeedback(
        run_id=rid,
        risk_uid="abc123",
        verdict="confirm",
        rationale="ok",
        context_signature="sig",
        risk_snapshot={"code": "RF_X", "title": "Test", "evidence": []},
        submitted_at=datetime.utcnow(),
    )
    log_feedback(fb)
    got = load_feedback(run_id=rid, risk_uid="abc123")
    assert len(got) == 1
    assert got[0].verdict == "confirm"
