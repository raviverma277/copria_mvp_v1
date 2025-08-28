# core/risk/cope_rules.py
from typing import Dict, Any

def compute_cope_score(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tiny, explainable COPE scorer.
    Inputs: pipeline 'results' dict (the same one you render in tabs).
    Returns: {"score": int|str, "breakdown": {...}}  -- matches your app.py usage.
    """

    # Pull normalized rows your pipeline already emits
    sov_rows = results.get("sov", []) or []          # list[dict]
    loss_rows = results.get("loss_runs", []) or []   # list[dict]

    # --- Construction (C) ---
    good_const = {"rcc", "reinforced concrete"}
    c_points = 0
    for r in sov_rows:
        cons = str(r.get("construction","")).lower()
        if any(g in cons for g in good_const):
            c_points += 1
    c_score = min(10, c_points)   # cap

    # --- Occupancy (O) ---
    # crude heuristic: office = safer, warehouse = neutral, unknown = neutral
    o_score = 0
    for r in sov_rows:
        occ = str(r.get("occupancy","")).lower()
        if "office" in occ:
            o_score += 2
        elif "warehouse" in occ:
            o_score += 1
    o_score = min(10, o_score)

    # --- Protection (P) ---
    # sprinklers add, lack subtracts
    p_score = 0
    for r in sov_rows:
        sprink = str(r.get("sprinklered","")).lower()
        if sprink in ("true", "1", "yes"):
            p_score += 2
        elif sprink in ("false", "0", "no"):
            p_score -= 1
    p_score = max(0, min(10, p_score))

    # --- Exposure (E) ---
    # simple proxy: open losses reduce; closed small losses mild impact
    e_score = 10
    for r in loss_rows:
        status = str(r.get("status","")).lower()
        incurred = float(r.get("incurred", 0) or 0)
        if status == "open":
            e_score -= 2
        e_score -= min(2, incurred / 250000)  # scale down to avoid big swings
    e_score = max(0, min(10, int(round(e_score))))

    total = c_score + o_score + p_score + e_score
    breakdown = {
        "construction": c_score,
        "occupancy": o_score,
        "protection": p_score,
        "exposure": e_score
    }
    return {"score": total, "breakdown": breakdown}
