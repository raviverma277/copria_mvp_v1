# core/schemas/profiler.py
from __future__ import annotations
from typing import Any, Dict, List
from collections import Counter
from datetime import date, datetime

def _is_floaty(x: Any) -> bool:
    try: float(x); return True
    except: return False

def _is_inty(x: Any) -> bool:
    try: return float(x).is_integer()
    except: return False

def _is_booly(x: Any) -> bool:
    s = str(x).strip().lower()
    return s in {"true","false","yes","no","y","n","1","0"}

def _is_datey(x: Any) -> bool:
    if isinstance(x, (date, datetime)): return True
    s = str(x)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            from datetime import datetime as dt
            dt.strptime(s, fmt); return True
        except: pass
    return False

def profile_rows(rows: List[Dict[str, Any]], max_enum: int = 12) -> Dict[str, Dict[str, Any]]:
    """
    Returns metrics per field:
      { field: {count, empty, uniques, sample_values, type_guess, min, max, maxLength, enum?, nullable} }
    """
    if not rows: return {}
    fields = set().union(*[r.keys() for r in rows])
    out: Dict[str, Dict[str, Any]] = {}

    for f in sorted(fields):
        vals = [r.get(f) for r in rows]
        non_empty = [v for v in vals if v not in (None, "", " ")]
        empties = len(vals) - len(non_empty)
        uniques = Counter(non_empty)
        sample = list(uniques.keys())[:5]

        # type guess (conservative)
        guess = "string"
        if non_empty and all(_is_booly(v) for v in non_empty):
            guess = "boolean"
        elif non_empty and all(_is_floaty(v) for v in non_empty):
            guess = "number" if not all(_is_inty(v) for v in non_empty) else "integer"
        elif non_empty and all(_is_datey(v) for v in non_empty):
            guess = "string"  # keep dates as strings for MVP

        # ranges / lengths
        min_v = max_v = None
        if guess in {"number","integer"}:
            try:
                nums = [float(v) for v in non_empty]
                min_v, max_v = (min(nums), max(nums)) if nums else (None, None)
            except: pass

        max_len = None
        if guess == "string":
            try: max_len = max(len(str(v)) for v in non_empty) if non_empty else 0
            except: pass

        enum_vals = None
        if 0 < len(uniques) <= max_enum and guess == "string":
            enum_vals = list(uniques.keys())

        out[f] = {
            "count": len(vals),
            "empty": empties,
            "nullable": empties > 0,
            "uniques": len(uniques),
            "sample_values": sample,
            "type_guess": guess,
            "min": min_v, "max": max_v,
            "maxLength": max_len,
            "enum": enum_vals,
        }
    return out

def profile_to_suggestions(profile: dict, active_schema) -> dict:
    """
    Convert profile into schema suggestions, skipping fields already
    present in the active schema. `active_schema` may be:
      - dict (schema JSON)
      - (dict, meta) tuple
      - None
    """
    # Normalize active_schema to a dict
    schema_dict = None
    if isinstance(active_schema, dict):
        schema_dict = active_schema
    elif isinstance(active_schema, (list, tuple)) and active_schema:
        # common pattern: (schema_dict, schema_name/path)
        maybe_dict = active_schema[0]
        if isinstance(maybe_dict, dict):
            schema_dict = maybe_dict
    # else: leave as None (treat as empty schema)

    active_fields = set()
    if isinstance(schema_dict, dict):
        props = schema_dict.get("properties", {}) or {}
        active_fields = set(props.keys())

    suggestions = {}
    for field, stats in (profile or {}).items():
        if field in active_fields:
            continue  # already in schema

        # conservative proposal from profile stats
        suggestion = {
            "type": stats.get("type_guess", "string"),
        }
        # examples
        if stats.get("sample_values"):
            suggestion["examples"] = [stats["sample_values"][0]]

        # strings
        if stats.get("maxLength") and suggestion["type"] == "string":
            suggestion["maxLength"] = stats["maxLength"]
        if stats.get("enum") and suggestion["type"] == "string":
            # keep enums small
            suggestion["enum"] = stats["enum"][:10]

        # numeric bounds
        if stats.get("min") is not None and suggestion["type"] in {"number", "integer"}:
            suggestion["minimum"] = stats["min"]
        if stats.get("max") is not None and suggestion["type"] in {"number", "integer"}:
            suggestion["maximum"] = stats["max"]

        suggestions[field] = suggestion

    return suggestions



