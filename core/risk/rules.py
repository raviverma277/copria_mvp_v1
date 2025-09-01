# core/risk/rules.py

from typing import Dict, Any, List
from .models import RiskItem, EvidenceRef
import hashlib, json

def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "y"}

def _falsey(v) -> bool:
    if isinstance(v, bool):
        return not v
    if v is None:
        # Unknown should NOT be treated as false
        return False
    s = str(v).strip().lower()
    return s in {"false", "0", "no", "n"}

def rules_from_bundle(bundle: Dict[str, Any]) -> List[RiskItem]:
    """
    bundle expected shape:
      {
        "sov":      [ { "sheets": [ { "rows": [ {...}, ... ] } ] } ],
        "loss_run": [ { "sheets": [ { "rows": [ {...}, ... ] } ] } ],
      }
    """
    out: List[RiskItem] = []

    # --- Build a quick index: location_id -> address (for nicer titles) ---
    loc_to_addr = {}
    for sov_file in bundle.get("sov", []) or []:
        sheets = sov_file.get("sheets") or []
        for sh in sheets[:1]:
            for r in (sh.get("rows") or []):
                loc = r.get("location_id")
                if loc:
                    loc_to_addr[str(loc)] = r.get("address")

    # --- Rule 1: Sprinklers absent (explicitly false only) ---
    for sov_file in bundle.get("sov", []) or []:
        sheets = sov_file.get("sheets") or []
        for sh in sheets[:1]:
            rows = sh.get("rows") or []
            for i, r in enumerate(rows, start=1):
                val = r.get("sprinklered", None)

                # Flag ONLY when explicitly false (False, "no", "false", "0", ...)
                if _falsey(val):
                    loc = r.get("location_id") or r.get("address") or f"row{i}"
                    title_loc = str(loc)
                    title_addr = loc_to_addr.get(title_loc)
                    title = f"Sprinklers absent at {title_loc}"
                    if title_addr:
                        title += f" ({title_addr})"

                    item = RiskItem(
                        code="SPRINKLER_ABSENT",
                        title=title,
                        severity="high",
                        rationale="Location appears not sprinklered, increasing fire severity.",
                        evidence=[EvidenceRef(
                            source="sov",
                            locator=f"sheet=1,row={i}",
                            snippet=str({k: r.get(k) for k in ["location_id","address","sprinklered"]}),
                            role="primary"
                        )],
                        tags=["fire", "protection"],
                        rule_hits=["R_SOV_SprinklerAbsent"],
                        confidence=0.75
                    )
                    out.append(item)
            # If explicitly true or unknown -> do not flag, do nothing           

    # --- Rule 2: Open theft claim (include which location) ---
    for loss_file in bundle.get("loss_run", []) or []:
        sheets = loss_file.get("sheets") or []
        for sh in sheets[:1]:
            rows = sh.get("rows") or []
            for i, r in enumerate(rows, start=1):
                status = str(r.get("status", "")).strip().lower()
                cause = str(r.get("cause", "")).strip().lower()
                if status == "open" and "theft" in cause:
                    loc = r.get("location_id") or "unknown location"
                    addr = loc_to_addr.get(str(loc))
                    at_where = f" at {loc}" + (f" ({addr})" if addr else "")
                    item = RiskItem(
                        code="OPEN_THEFT_CLAIM",
                        title=f"Open theft claim{at_where}",
                        severity="medium",
                        rationale=f"Open theft loss{at_where} increases expected frequency/severity until mitigated.",
                        evidence=[EvidenceRef(
                            source="loss_run",
                            locator=f"row={i}",
                            snippet=str({k: r.get(k) for k in ["cause","status","location_id","incurred"]}),
                            role="primary"
                        )],
                        tags=["crime"],
                        rule_hits=["R_Loss_OpenTheft"],
                        confidence=0.7
                    )                    
                    out.append(item)


    return out


