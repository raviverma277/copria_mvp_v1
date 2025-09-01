from collections import Counter
from .store import load_feedback


def verdict_counts(run_id: str):
    c = Counter(fb.verdict for fb in load_feedback(run_id=run_id))
    total = sum(c.values())
    return {"total": total, **c}
