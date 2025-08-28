from __future__ import annotations
import re
from typing import Dict, List, Any

_CURR_RE = re.compile(r"[^\d\.\-]")

def _to_bool(v: Any) -> bool | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("yes", "y", "true", "t", "1"):
        return True
    if s in ("no", "n", "false", "f", "0"):
        return False
    return None  # leave as None if ambiguous

def _to_float(v: Any) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    # strip currency symbols, commas, spaces
    s = _CURR_RE.sub("", str(v))
    try:
        return float(s) if s not in ("", "-", ".", "-.") else None
    except Exception:
        return None

def _clean_str(v: Any) -> Any:
    return v.strip() if isinstance(v, str) else v

def normalize_sov_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        # 1) shallow copy + trim strings
        rr = {k: _clean_str(v) for k, v in r.items()}

        # 2) known booleans
        if "sprinklers" in rr:
            b = _to_bool(rr["sprinklers"])
            if b is not None:
                rr["sprinklers"] = b
        if "alarm" in rr:
            b = _to_bool(rr["alarm"])
            if b is not None:
                rr["alarm"] = b
        if "heritage_status" in rr:
            b = _to_bool(rr["heritage_status"])
            if b is not None:
                rr["heritage_status"] = b
        if "_provisional_heritage_status" in rr:
            b = _to_bool(rr["_provisional_heritage_status"])
            if b is not None:
                rr["_provisional_heritage_status"] = b

        # 3) known numerics (keep optional for distance)
        if "tiv_building" in rr:
            n = _to_float(rr["tiv_building"])
            if n is not None:
                rr["tiv_building"] = n
        if "tiv_content" in rr:
            n = _to_float(rr["tiv_content"])
            if n is not None:
                rr["tiv_content"] = n
        if "tiv_bi" in rr:
            n = _to_float(rr["tiv_bi"])
            if n is not None:
                rr["tiv_bi"] = n
        if "fire_station_distance_km" in rr:
            n = _to_float(rr["fire_station_distance_km"])
            if n is not None:
                rr["fire_station_distance_km"] = n

        out.append(rr)
    return out

def normalize_loss_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        rr = {k: _clean_str(v) for k, v in r.items()}

        # status → uppercase enum expected by schema
        if "status" in rr and isinstance(rr["status"], str):
            rr["status"] = rr["status"].strip().upper()

        # gross fields might be currency-formatted
        for k in ("gross_paid", "gross_outstanding", "incurred"):
            if k in rr:
                n = _to_float(rr[k])
                if n is not None:
                    rr[k] = n

        # booleans
        if "reopened_flag" in rr:
            b = _to_bool(rr["reopened_flag"])
            if b is not None:
                rr["reopened_flag"] = b

        # keep reported/close dates as strings; schema/validators handle format
        out.append(rr)
    return out
