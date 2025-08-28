
from typing import Dict, Any

def price_submission(bundle: Dict[str, Any], cope: Dict[str, Any], concerns: Dict[str, Any]) -> Dict[str, Any]:
    # Simple demo pricing
    total_tiv = bundle.get("submission_core", {}).get("total_tiv", 10_000_000)
    base_rate = 0.25  # per 100 of TIV
    premium = (total_tiv / 100.0) * base_rate
    modifiers = {"cope": 1.0, "concerns": 1.0}
    return {"premium": round(premium, 2), "currency": bundle.get("submission_core", {}).get("currency","GBP"), "modifiers": modifiers}
