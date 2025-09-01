# core/risk/nuanced.py
from typing import Dict, Any, List


def generate_nuanced_concerns(
    results: Dict[str, Any], cope: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Produces a small set of human-readable concerns using the same 'results' dict
    and (optionally) the COPE breakdown.
    Returns: {"concerns": [ {title, severity, rationale}, ... ] }
    """
    sov_rows = results.get("sov", []) or []
    loss_rows = results.get("loss_runs", []) or []
    concerns: List[Dict[str, str]] = []

    # 1) Non-sprinklered locations
    for i, r in enumerate(sov_rows):
        spr = str(r.get("sprinklered", "")).lower()
        if spr in ("", "false", "0", "no"):
            loc = r.get("location_id") or r.get("address") or f"row {i+1}"
            concerns.append(
                {
                    "title": f"Sprinklers absent at {loc}",
                    "severity": "High",
                    "rationale": "Absence of sprinklers elevates fire severity and downtime risk.",
                }
            )

    # 2) Open theft claim
    for r in loss_rows:
        if (
            str(r.get("status", "")).lower() == "open"
            and "theft" in str(r.get("cause", "")).lower()
        ):
            concerns.append(
                {
                    "title": "Open theft claim on schedule",
                    "severity": "Medium",
                    "rationale": "Open loss indicates unresolved exposure; verify mitigation and security controls.",
                }
            )
            break

    # 3) If COPE exposure is low, note that
    try:
        if (cope or {}).get("breakdown", {}).get("exposure", 10) <= 4:
            concerns.append(
                {
                    "title": "Elevated exposure score",
                    "severity": "Medium",
                    "rationale": "Recent losses or open reserves reduce overall risk tolerance until remedied.",
                }
            )
    except Exception:
        pass

    return {"concerns": concerns}
