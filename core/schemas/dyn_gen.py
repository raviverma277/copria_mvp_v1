# core/schemas/dyn_gen.py
from typing import Dict, Any


def suggestions_to_schema_props(profile: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    for field, m in (profile or {}).items():
        typ = m.get("type_guess", "string")
        p: Dict[str, Any] = {"type": "string"}

        if typ in {"number", "integer"}:
            p["type"] = "number" if typ == "number" else "integer"
            if m.get("min") is not None:
                p["minimum"] = m["min"]
            if m.get("max") is not None:
                p["maximum"] = m["max"]
        elif typ == "boolean":
            p["type"] = "boolean"
        else:
            p["type"] = "string"
            if m.get("maxLength"):
                p["maxLength"] = m["maxLength"]
            enum_vals = m.get("enum")
            if enum_vals and 2 <= len(enum_vals) <= 8:
                p["enum"] = [str(v) for v in enum_vals]

        if m.get("sample_values"):
            p["examples"] = [m["sample_values"][0]]

        props[field] = p
    return props
