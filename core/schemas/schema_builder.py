# core/schemas/schema_builder.py
# -----------------------------------------------------------------------------
# Schema builder & versioning utilities
# - Works with repo layout: core/schemas/json/*.schema*.json + active_schema.json
# - Supports array-of-objects schemas (items.properties) and object.properties
# - Public API (compat): load_active_schema, propose_vnext_schema, write_vnext_and_point_active
# - New API: preview_vnext_schema, generate_vnext_schema, build_property
# -----------------------------------------------------------------------------

from __future__ import annotations
from core.utils.llm_status import record_llm_call_start, record_llm_call_end
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import os, json
import time
from datetime import datetime as _dt


SCHEMA_DIR = Path("core/schemas/json")
ACTIVE_PTR = Path("core/schemas/active_schema.json")  # {"sov":"...", "loss_run":"...", ...}

PROV_PREFIX = "_provisional_"

# -------- I/O helpers --------

# ---- LLM diagnostics (for UI) ----
_LLM_LAST_ERROR = None

def _dbg(msg: str):
    # lightweight console logging for Streamlit's terminal
    try:
        print(f"[schema_builder] {msg}", flush=True)
    except Exception:
        pass

def _set_llm_last_error(msg: str):
    global _LLM_LAST_ERROR
    _LLM_LAST_ERROR = msg

def get_llm_last_error() -> str | None:
    return _LLM_LAST_ERROR

# ---- LLM meta (model/usage) for UI diagnostics ----
_LLM_LAST_META = None

def _set_llm_last_meta(meta):
    global _LLM_LAST_META
    _LLM_LAST_META = meta

def get_llm_last_meta():
    return _LLM_LAST_META


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

# -------- active pointer helpers --------

def _get_active_pointer() -> Dict[str, str]:
    return _read_json(ACTIVE_PTR) if ACTIVE_PTR.exists() else {}

def _set_active_pointer(ptr: Dict[str, str]) -> None:
    _write_json(ACTIVE_PTR, ptr)

def get_active_name(doc_type: str) -> str:
    """Return active filename for a doc type (fallback to <doc_type>.schema.json)."""
    ptr = _get_active_pointer()
    return ptr.get(doc_type, f"{doc_type}.schema.json")

def _set_active_name(doc_type: str, filename: str) -> None:
    ptr = _get_active_pointer()
    ptr[doc_type] = filename
    _set_active_pointer(ptr)

# -------- schema-shape helpers --------

def _get_props_node(schema: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Return (properties_dict, shape) where shape is 'array' or 'object'.
    If neither exists, initialize object.properties and return it.
    """
    if isinstance(schema.get("items"), dict) and isinstance(schema["items"].get("properties"), dict):
        return schema["items"]["properties"], "array"
    if isinstance(schema.get("properties"), dict):
        return schema["properties"], "object"
    # init object.properties if nothing present
    schema.setdefault("properties", {})
    return schema["properties"], "object"

def _set_props_node(schema: Dict[str, Any], props: Dict[str, Any], shape: str) -> None:
    if shape == "array":
        schema.setdefault("items", {}).setdefault("type", "object")
        schema["items"]["properties"] = props
    else:
        schema["properties"] = props

# -------- NEW: utilities to strip provisional and merge safely --------

def _strip_provisional(name: str) -> str:
    """Return clean field name (without _provisional_ prefix)."""
    return name[len(PROV_PREFIX):] if isinstance(name, str) and name.startswith(PROV_PREFIX) else name

def _merge_new_properties(schema: Dict[str, Any], new_props: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep copy schema and merge properties, stripping _provisional_ from keys.
    If a clean key already exists, the provisional duplicate is ignored.
    """
    s = json.loads(json.dumps(schema))  # deepcopy
    props, shape = _get_props_node(s)

    clean_props: Dict[str, Any] = {}
    for raw_k, v in (new_props or {}).items():
        base = _strip_provisional(raw_k)
        # if base already present, skip the provisional duplicate
        if base in props:
            continue
        clean_props[base] = v

    # If you also want to ensure "type" is set on each property:
    for k, v in list(clean_props.items()):
        if isinstance(v, dict) and "type" not in v:
            v["type"] = "string"

    props.update(clean_props)
    _set_props_node(s, props, shape)
    return s

# --- ADD directly under _merge_new_properties(...) ---

def _rename_provisional_keys_inplace(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rename any property keys that start with _provisional_ to their clean base.
    If a clean key already exists, shallow-merge and prefer the existing clean definition.
    Works for both object.properties and items.properties (array-of-objects).
    Returns the mutated schema for convenience.
    """
    props, shape = _get_props_node(schema)
    if not isinstance(props, dict):
        return schema

    to_move = []
    for k in list(props.keys()):
        base = _strip_provisional(k)
        if base != k:
            to_move.append((k, base))

    for old, new in to_move:
        if new in props and isinstance(props[new], dict) and isinstance(props[old], dict):
            # prefer existing clean, but bring over any missing keys
            merged = {**props[old], **props[new]}  # keep clean's values on conflict
            props[new] = merged
        else:
            props[new] = props.get(old)
        props.pop(old, None)

    _set_props_node(schema, props, shape)
    return schema

def _coerce_new_fields(new_fields: Any) -> Dict[str, Any]:
    """
    Normalize 'new_fields' to a dict: { clean_key: property_obj }
    Accepts:
      A) dict: { "field_key": {<json-schema-prop>}, ... }
      B) list: [
            {"field_name": "...", "type": "string", "description": "...", "examples": [...], "enum": [...]},
            {"name": "..."},  # alt key
            {"field_name": "...", "property": {...}},  # explicit property object
            "plain_field_name"
         ]
      C) str: "plain_field_name"
    - Strips '_provisional_' from keys.
    - Ensures each property has at least {"type":"string"} if missing.
    """
    out: Dict[str, Any] = {}
    if not new_fields:
        return out

    # A) dict shape
    if isinstance(new_fields, dict):
        for k, prop in new_fields.items():
            key = _strip_provisional((k or "").strip())
            if not key:
                continue
            p = prop or {}
            if isinstance(p, dict) and "type" not in p:
                p["type"] = "string"
            out[key] = p
        return out

    # B) list shape
    if isinstance(new_fields, list):
        for nf in new_fields:
            if isinstance(nf, dict):
                key = _strip_provisional((nf.get("field_name") or nf.get("name") or "").strip())
                if not key:
                    continue
                # allow explicit property dict or synthesize from simple fields
                prop = (
                    nf.get("property")
                    or nf.get("schema")
                    or {
                        "type": nf.get("type", "string"),
                        **({"description": nf["description"]} if nf.get("description") else {}),
                        **({"examples": nf["examples"]} if nf.get("examples") else {}),
                        **({"enum": nf["enum"]} if nf.get("enum") else {}),
                    }
                )
                if isinstance(prop, dict) and "type" not in prop:
                    prop["type"] = "string"
                out[key] = prop
            else:
                key = _strip_provisional(str(nf).strip())
                if key:
                    out[key] = {"type": "string"}
        return out

    # C) single string
    if isinstance(new_fields, str):
        key = _strip_provisional(new_fields.strip())
        if key:
            out[key] = {"type": "string"}
    return out


# -------- flatten & diff (for preview UI) --------

def _flatten_prop_types(schema: Dict[str, Any]) -> Dict[str, str]:
    props, _ = _get_props_node(schema)
    out: Dict[str, str] = {}
    for k, v in (props or {}).items():
        out[k] = v.get("type", "any") if isinstance(v, dict) else "any"
    return out

def _compute_diff(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    o = _flatten_prop_types(old)
    n = _flatten_prop_types(new)
    added = {k: n[k] for k in n.keys() - o.keys()}
    removed = {k: o[k] for k in o.keys() - n.keys()}
    changed = {k: {"from": o[k], "to": n[k]} for k in o.keys() & n.keys() if o[k] != n[k]}
    return {"added": added, "removed": removed, "changed": changed}

# ========= LLM enrichment for JSON-Schema properties =========
import os


def _render_schema_prompt(doc_type: str, field: str, samples: List[Any]) -> str:
    # keep short; your model/context window is precious
    sample_str = "\n".join([f"- {repr(s)}" for s in samples[:15]])
    return f"""You are a data schema expert creating JSON Schema properties for an {doc_type.upper()} ingestion system.
Given the field name and its sample values, propose a concise JSON Schema property.

Requirements:
- Provide a JSON object with keys: type (string), description (string), examples (array; optional), enum (array; optional), format (string; optional)
- Do NOT include $id, $schema, or title.
- Prefer "number" vs "integer" if values include decimals; otherwise "integer".
- For booleans, return type "boolean".
- For ISO dates / timestamps, set type "string" and an appropriate "format" (e.g., "date" or "date-time").

Field: {field}
Sample values:
{sample_str}
Return ONLY the JSON object.
"""

def _llm_property_for_field(doc_type: str, field: str, samples: List[Any]) -> Dict[str, Any]:
    """
    Best-effort call to LLM to infer a JSON-Schema property for a field given example values.
    Falls back to {"type": "string"} on any error.
    """
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            _set_llm_last_error("Missing OPENAI_API_KEY") 
            return {"type": "string"}  # no-op if key missing

        # Minimal OpenAI client; replace with your existing wrapper if you have one.
        try:
            from openai import OpenAI
        except Exception as e:
            _set_llm_last_error(f"OpenAI import failed: {e}")
            return {"type": "string"} 
         # or {"type":"object","properties":{}}
        client = OpenAI(api_key=api_key)
        prompt = _render_schema_prompt(doc_type, field, samples or [])
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        txt = (resp.choices[0].message.content or "").strip()
        import json
        if "```" in txt:
            txt = txt.split("```")[-2]  # take fenced block if present
        obj = json.loads(txt)
        if not isinstance(obj, dict):
            return {"type": "string"}
        # minimal hardening
        if "type" not in obj:
            obj["type"] = "string"
        return obj
    except Exception as e:
         _set_llm_last_error(f"OpenAI call failed: {e}")
         return {"type": "string"}

def enrich_properties_with_llm(
    doc_type: str,
    queued_props: Dict[str, Dict[str, Any]],
    field_samples: Dict[str, List[Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    For every queued field with missing/minimal property, call LLM to infer a better property.
    'Minimal' now means: property lacks description/examples/enum/format (title alone doesn't block enrichment).
    Returns a NEW dict with enriched properties; input dict is NOT mutated.
    """
    ENRICH_KEYS = {"description", "examples", "enum", "format"}
    out: Dict[str, Dict[str, Any]] = {}
    for field, prop in (queued_props or {}).items():
        prop = dict(prop or {})
        keys = set(prop.keys())
        # treat as minimal if it lacks any "enrichment" keys; 'title' is allowed without blocking enrichment
        needs_enrich = not (keys & ENRICH_KEYS)
        if needs_enrich:
            samples = field_samples.get(field, [])
            enriched = _llm_property_for_field(doc_type, field, samples or [])
            # ensure type always present
            if "type" not in enriched:
                enriched["type"] = prop.get("type", "string")
            # preserve a pre-supplied title if present
            if "title" in prop and "title" not in enriched:
                enriched["title"] = prop["title"]
            out[field] = enriched
        else:
            # already has richer info; keep as-is
            if "type" not in prop:
                prop["type"] = "string"
            out[field] = prop
    return out

# ========= LLM full-schema draft (experimental) =========
def _guess_py_type(val: Any) -> str:
    """Very lightweight type guess for prompt context."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, int):
        return "integer"
    if isinstance(val, float):
        return "number"
    s = str(val).strip()
    # crude date/datetime sniff (keep it cheap)
    if len(s) in (10, 19, 20, 24) and s[4] == "-" and s[7] == "-":
        return "date/date-time?"
    return "string"

# ---- JSON sanitization helpers for prompts ----
from datetime import date, datetime
from decimal import Decimal

try:
    import numpy as _np
except Exception:
    _np = None

try:
    import pandas as _pd
except Exception:
    _pd = None


def _to_jsonable(val):
    """Coerce non-JSON-serializable objects (Timestamp, numpy scalars, Decimal, datetime) to safe JSON types."""
    # None, bool, int, float, str are fine
    if val is None or isinstance(val, (bool, int, float, str)):
        return val

    # numpy scalars -> python scalars
    if _np is not None and isinstance(val, (_np.integer, _np.floating, _np.bool_)):
        return val.item()

    # pandas Timestamp / datetime / date -> ISO strings
    if (_pd is not None and isinstance(val, _pd.Timestamp)) or isinstance(val, (datetime, date)):
        try:
            # Prefer date-time with 'Z' for UTC, else plain ISO
            iso = val.isoformat()
            return iso if isinstance(val, date) else iso
        except Exception:
            return str(val)

    # Decimal -> float (lossy but OK for examples)
    if isinstance(val, Decimal):
        try:
            return float(val)
        except Exception:
            return str(val)

    # Fallback
    return str(val)


def _json_sanitize_field_summaries(summaries: dict) -> dict:
    """Deep-sanitize summaries so json.dumps never fails."""
    out = {}
    for k, info in (summaries or {}).items():
        if not isinstance(info, dict):
            out[k] = _to_jsonable(info)
            continue
        cleaned = {}
        for kk, vv in info.items():
            if isinstance(vv, list):
                cleaned[kk] = [_to_jsonable(x) for x in vv]
            elif isinstance(vv, dict):
                cleaned[kk] = {sk: _to_jsonable(sv) for sk, sv in vv.items()}
            else:
                cleaned[kk] = _to_jsonable(vv)
        out[k] = cleaned
    return out


def _guess_py_type(val) -> str:
    """Lightweight type guess; improved for pandas.Timestamp and datetime/date."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, int):
        return "integer"
    if isinstance(val, float):
        return "number"
    # pandas Timestamp / datetime / date
    if (_pd is not None and isinstance(val, _pd.Timestamp)) or isinstance(val, (datetime, date)):
        return "date-time"
    # strings that look like ISO dates
    s = str(val).strip()
    if len(s) in (10, 19, 20, 24) and s[4:5] == "-" and s[7:8] == "-":
        # e.g., 2024-06-01 / 2024-06-01T00:00:00Z
        return "date/date-time?"
    return "string"

def polish_full_schema_draft(doc_type: str, draft: dict) -> dict:
    if not isinstance(draft, dict):
        return {"type": "object", "properties": {}}
    draft = {**draft}
    draft.setdefault("type", "object")
    props = draft.get("properties") or {}
    if not isinstance(props, dict):
        props = {}
    draft["properties"] = props

    # --- NEW: rename provisional keys to clean base if no collision ---
    if props:
        renamed = {}
        for k, v in list(props.items()):
            if isinstance(k, str) and k.startswith("_provisional_"):
                base = k[len("_provisional_"):]
                if base and base not in props and base not in renamed:
                    renamed[base] = v
                    del props[k]
        props.update(renamed)
    # ------------------------------------------------------------------

    # Minimal required sets by bucket (keep intersection with props)
    minimal_required_map = {
        "sov": ["address", "country", "tiv_building"],
        "loss_run": ["date_of_loss", "cause"],
    }
    requested = minimal_required_map.get(doc_type, [])
    draft["required"] = [k for k in requested if k in props]

    def _dedup_examples(p):
        ex = p.get("examples")
        if isinstance(ex, list):
            seen, out = set(), []
            for v in ex:
                key = repr(v)
                if key not in seen:
                    seen.add(key)
                    out.append(v)
            p["examples"] = out[:5]

    def _normalize_bool_examples(p):
        ex = p.get("examples")
        if isinstance(ex, list):
            has_t = any(x is True or (isinstance(x, str) and x.strip().lower() == "true") for x in ex)
            has_f = any(x is False or (isinstance(x, str) and x.strip().lower() == "false") for x in ex)
            if has_t or has_f:
                p["examples"] = ([True] if has_t else []) + ([False] if has_f else [])

    for k, p in props.items():
        if not isinstance(p, dict):
            continue
        t = p.get("type")

        # Geo/codes/commons
        if k == "lat" and t == "number":
            p.setdefault("minimum", -90)
            p.setdefault("maximum", 90)
        if k == "lng" and t == "number":
            p.setdefault("minimum", -180)
            p.setdefault("maximum", 180)
        if k == "country" and t == "string":
            p.setdefault("pattern", "^[A-Z]{2}$")
        if k == "roof_age_years" and t in ("integer", "number"):
            p.setdefault("minimum", 0)
        if k in ("tiv_building", "tiv_content", "tiv_bi") and t in ("integer", "number"):
            p.setdefault("minimum", 0)

        _dedup_examples(p)
        if t == "boolean":
            _normalize_bool_examples(p)

        # Loss-run specifics (kept if you have them already)
        if doc_type == "loss_run":
            if k in ("gross_paid", "gross_outstanding", "incurred", "net_paid", "net_outstanding") and t in ("integer", "number"):
                p.setdefault("minimum", 0)
            if k == "date_of_loss" and t == "string":
                p.setdefault("format", "date")
            if k in ("reported_date", "close_date") and t == "string":
                p.setdefault("format", "date-time")
            if k == "status" and t == "string":
                p.setdefault("enum", ["OPEN", "CLOSED", "UNKNOWN"])
                if isinstance(p.get("examples"), list):
                    p["examples"] = [str(x).upper() for x in p["examples"]][:5]
            if k == "litigation_status" and t == "string":
                p.setdefault("enum", ["Open", "Closed"])

    return draft



def _extract_first_json_object(txt: str) -> str | None:
    """
    Return the first complete top-level JSON object substring from txt.
    Handles code-fences, leading prose, and extra trailing text/objects.
    """
    if not txt:
        return None

    # If fenced, prefer content inside fences
    if "```" in txt:
        parts = [p.strip() for p in txt.split("```") if p.strip()]
        # try to pick a json block or any block starting with '{'
        for p in parts:
            if p.lstrip().startswith("{"):
                txt = p
                break  # use the first fenced JSON-looking block

    # Trim leading text until first '{'
    start = txt.find("{")
    if start == -1:
        return None
    s = txt[start:]

    # Scan with brace depth & quote awareness to find matching end
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # end of the first complete JSON object
                    return s[: i + 1]
    return None  # no complete object found


def summarize_rows_for_prompt(rows: List[Dict[str, Any]], max_fields: int = 60, samples_per_field: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Create a compact summary per field for the LLM prompt:
      { field: { "sample_types": [...], "examples": [...], "non_null_ratio": 0.87 } }
    JSON-sanitized so json.dumps always works.
    """
    if not rows:
        return {}
    # collect union of keys, limit max_fields
    fields = set()
    for r in rows:
        fields.update(r.keys())
    fields = list(fields)[:max_fields]

    summary: Dict[str, Dict[str, Any]] = {}
    total = max(1, len(rows))
    for f in fields:
        ex_raw, types, count = [], set(), 0
        for r in rows:
            if f in r and r[f] is not None:
                v = r[f]
                types.add(_guess_py_type(v))
                if len(ex_raw) < samples_per_field:
                    ex_raw.append(v)
                count += 1
        if count == 0:
            continue
        # sanitize examples for JSON
        ex = [_to_jsonable(x) for x in ex_raw]
        summary[f] = {
            "sample_types": sorted(types),
            "examples": ex,
            "non_null_ratio": round(count / total, 3),
        }
    # final deep sanitize (paranoid)
    return _json_sanitize_field_summaries(summary)


def _render_full_schema_prompt(doc_type: str, field_summaries: Dict[str, Dict[str, Any]]) -> str:
    """
    Compact instruction for the LLM: return a JSON Schema object with 'type': 'object' and a 'properties' map.
    """
    body = json.dumps(field_summaries, ensure_ascii=False, indent=2)
    return f"""You are a data schema expert. Generate a concise JSON Schema for an {doc_type.upper()} ingestion system.

INPUT: A set of fields with sample values, rough type hints, and presence ratios.
OUTPUT: A single JSON object with:
- "type": "object"
- "properties": object whose keys are field names, values are property objects.
- Each property should include at least "type". Add "format" for dates/timestamps, "description" (1 sentence), and "examples" (small).
- Prefer "number" for decimals, "integer" for whole numbers, "boolean" for true/false.
- Do NOT include $schema or $id. Do NOT include "title" at the top-level.
- Keep it succinct and machine-usable.

Field context:
{body}

Return ONLY the JSON object (no extra text).
"""

def llm_generate_full_schema(
    doc_type: str,
    rows: List[Dict[str, Any]],
    max_fields: int = 60,
    samples_per_field: int = 10
) -> Dict[str, Any]:
    """
    Build field summaries from rows and ask the LLM to produce a full JSON Schema draft:
    {
      "type": "object",
      "properties": { ... }
    }
    - Sanitizes samples (Timestamps, numpy scalars, Decimal, etc.) via summarize_rows_for_prompt
    - Hardened parsing with clear LLM diagnostics via _set_llm_last_error
    Falls back to an empty 'properties' map on failure.
    """
    # --- helper: extract first complete top-level JSON object from text ---
    def _extract_first_json_object(txt: str) -> Optional[str]:
        if not txt:
            return None
        # prefer fenced blocks if present
        if "```" in txt:
            parts = [p.strip() for p in txt.split("```") if p.strip()]
            for p in parts:
                if p.lstrip().startswith("{"):
                    txt = p
                    break
        # trim to first '{'
        start = txt.find("{")
        if start == -1:
            return None
        s = txt[start:]

        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(s):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[: i + 1]
        return None

    # Summarize and sanitize example values for the prompt
    summaries = summarize_rows_for_prompt(rows, max_fields=max_fields, samples_per_field=samples_per_field)
    if not summaries:
        _set_llm_last_error(None)
        _set_llm_last_meta(None)
        _dbg(f"no summaries for {doc_type}; returning empty properties")
        return {"type": "object", "properties": {}}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        _set_llm_last_error("Missing OPENAI_API_KEY")
        _set_llm_last_meta(None)
        _dbg("missing OPENAI_API_KEY")
        return {"type": "object", "properties": {}}

    try:
        from openai import OpenAI
    except Exception as e:
        _set_llm_last_error(f"OpenAI import failed: {e}")
        _set_llm_last_meta(None)
        _dbg(f"OpenAI import failed: {e}")
        return {"type": "object", "properties": {}}

    try:
        client = OpenAI(api_key=api_key)
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        prompt = _render_full_schema_prompt(doc_type, summaries)
        _dbg(f"calling LLM for {doc_type}: model={model}, fields={len(summaries)}")

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        # record model/usage for sidebar
        meta = {
            "model": getattr(resp, "model", None),
            "usage": getattr(resp, "usage", None),
        }
        _set_llm_last_meta(meta)
        _dbg(f"LLM returned (model={meta['model']})")

        txt = (resp.choices[0].message.content or "").strip()
        if not txt:
            _set_llm_last_error("Empty response from model")
            _dbg("empty model response")
            return {"type": "object", "properties": {}}

        json_str = _extract_first_json_object(txt)
        if not json_str:
            _set_llm_last_error("Model did not return a complete JSON object")
            _dbg("no JSON block found in response")
            return {"type": "object", "properties": {}}

        try:
            obj = json.loads(json_str)
        except Exception as e:
            _set_llm_last_error(f"Failed to parse model JSON: {e}")
            _dbg(f"JSON parse error: {e}")
            return {"type": "object", "properties": {}}

        if not isinstance(obj, dict):
            _set_llm_last_error("Parsed JSON is not an object")
            _dbg("parsed JSON is not an object")
            return {"type": "object", "properties": {}}

        obj.setdefault("type", "object")
        obj.setdefault("properties", {})
        if not isinstance(obj["properties"], dict):
            obj["properties"] = {}

        _set_llm_last_error(None)
        _dbg(f"schema draft ok: {len(obj['properties'])} properties")
        return obj

    except Exception as e:
        _set_llm_last_error(f"OpenAI call failed: {e}")
        _set_llm_last_meta(None)
        _dbg(f"OpenAI call failed: {e}")
        return {"type": "object", "properties": {}}


def propose_full_schema_from_llm(doc_type: str, rows: List[Dict[str, Any]], max_fields: int = 60, samples_per_field: int = 10) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Return (new_file_name, current_active_schema, llm_draft_schema)
    The draft is a full schema (object + properties) synthesized by the LLM from data rows.
    """
    _, current_schema = load_active_schema(doc_type)  # you already have this import in this file
    draft = llm_generate_full_schema(doc_type, rows, max_fields=max_fields, samples_per_field=samples_per_field)
    draft = polish_full_schema_draft(doc_type, draft)
    print("polish_full_schema_draft doc_type =", doc_type)

    
    ts = _dt.now().strftime("%Y-%m-%d_%H%M%S")    
    new_name = f"{doc_type}.schema.llm-draft.v{ts}.json"
    return new_name, current_schema, draft



# -------- public: active schema loader (compat) --------

def load_active_schema(doc_type: str) -> Tuple[Path, Dict[str, Any]]:
    """(compat) Return (path, json) for the active schema of a given doc type."""
    name = get_active_name(doc_type)
    path = SCHEMA_DIR / name
    return path, _read_json(path)

# -------- public: propose (compat) --------

def propose_vnext_schema(
    doc_type: str,
    approved_new_fields: Any  # accepts dict OR list OR str
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    (compat) Returns (new_file_name, current_schema, vnext_schema).
    Adds 'approved_new_fields' to the active schema as properties
    (stripping any _provisional_ prefix). Accepts dict/list/str inputs.
    """
    _, current = load_active_schema(doc_type)

    # Normalize to {clean_key: property_obj}
    new_props: Dict[str, Any] = _coerce_new_fields(approved_new_fields)

    # Merge with current schema (stripping provisional and avoiding dupes)
    vnext = _merge_new_properties(current, new_props)

    ts = _dt.now().strftime("%Y-%m-%d_%H%M%S")
    new_name = f"{doc_type}.schema.v{ts}.json"
    return new_name, current, vnext


# -------- public: writer (compat) --------

def write_vnext_and_point_active(doc_type: str, new_name: str, vnext_schema: Dict[str, Any], make_active: bool) -> Path:
    dest = SCHEMA_DIR / new_name
    _write_json(dest, vnext_schema)
    if make_active:
        _set_active_name(doc_type, new_name)
    return dest

# --- ADD after write_vnext_and_point_active(...) and before preview_vnext_schema(...) ---

def migrate_active_schema(doc_type: str) -> Dict[str, Any]:
    """
    One-time migration for the ACTIVE schema of a given doc_type (e.g., 'sov', 'loss_run').
    - Loads active schema json
    - Renames any _provisional_* property keys to clean keys
    - Saves back to the same active file
    Returns a summary dict.
    """
    path, current = load_active_schema(doc_type)
    if not isinstance(current, dict):
        current = {}

    props_before, _ = _get_props_node(current)
    before_keys = set(props_before.keys()) if isinstance(props_before, dict) else set()

    _rename_provisional_keys_inplace(current)

    props_after, _ = _get_props_node(current)
    after_keys = set(props_after.keys()) if isinstance(props_after, dict) else set()

    _write_json(path, current)

    renamed = [k for k in before_keys if k.startswith(PROV_PREFIX)]
    return {
        "doc_type": doc_type,
        "file": str(path),
        "renamed_count": len(renamed),
        "before_prop_count": len(before_keys),
        "after_prop_count": len(after_keys),
    }


def migrate_all_active_schemas(doc_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run migration for multiple active schemas.
    If doc_types not provided, derive from the ACTIVE pointer; if empty, default to ['sov','loss_run'].
    """
    ptr = _get_active_pointer()
    if doc_types is None:
        doc_types = list(ptr.keys()) or ["sov", "loss_run"]

    out = {}
    for dt in doc_types:
        try:
            out[dt] = migrate_active_schema(dt)
        except Exception as e:
            out[dt] = {"error": str(e)}
    return out


# -------- new: preview & generate vNext --------

def preview_vnext_schema(doc_type: str, new_properties: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build an in-memory vNext by merging `new_properties` with the active schema.
    Strips _provisional_ prefixes and returns (merged_schema_json, diff_dict).
    """
    _, active = load_active_schema(doc_type)
    merged = _merge_new_properties(active, new_properties or {})
    return merged, _compute_diff(active, merged)

def generate_vnext_schema(
    doc_type: str,
    new_properties: Optional[Dict[str, Any]] = None,
    make_active: bool = True,
    filename: Optional[str] = None,
) -> Tuple[str, Path]:
    """
    Write a timestamped (or custom-named) vNext file into SCHEMA_DIR and (optionally) set it active.
    Strips _provisional_ from new field names before writing.
    Also removes those newly added fields from the pending queue.
    Returns (filename, path).
    """
    # Merge the provided props into the ACTIVE schema (with stripping)
    _, active = load_active_schema(doc_type)
    merged = _merge_new_properties(active, new_properties or {})

    # Decide filename
    vnext_name = filename or f"{doc_type}.schema.v{time.strftime('%Y%m%d_%H%M%S')}.json"
    vnext_path = SCHEMA_DIR / vnext_name

    # Write schema file
    _write_json(vnext_path, merged)

    # Optionally switch active pointer
    if make_active:
        _set_active_name(doc_type, vnext_name)

    # Best-effort: remove these fields from the pending new_fields queue
    try:
        from core.schemas.proposals_store import remove_new_fields
        field_names = list((new_properties or {}).keys())
        # Normalize in case UI passed provisional names
        field_names = [ _strip_provisional(n) for n in field_names ]
        if field_names:
            removed = remove_new_fields(doc_type, field_names)
            print(f"[schema-builder] removed {removed} queued new_field(s) for {doc_type}")
    except Exception as e:
        print(f"[schema-builder] warning: could not remove queued new_fields: {e}")

    return vnext_name, vnext_path

# -------- small helper to build a property from UI inputs --------

def build_property(
    field_name: str,
    typ: str = "string",
    description: Optional[str] = None,
    examples: Optional[List[Any]] = None,
    enum: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a JSON Schema property object (no provisional in field_name here)."""
    clean = _strip_provisional(field_name or "")
    prop: Dict[str, Any] = {"type": typ}
    if description:
        prop["description"] = description
    if examples:
        prop["examples"] = examples
    if enum:
        prop["enum"] = enum
    return {clean: prop}

# --- ADD AT END OF FILE ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate ACTIVE schema(s): rename _provisional_* properties to clean keys.")
    parser.add_argument("--doc", action="append", dest="docs",
                        help="Doc type(s) to migrate (e.g., --doc sov --doc loss_run). If omitted, all active schemas are migrated.")
    args = parser.parse_args()
    result = migrate_all_active_schemas(args.docs)
    print(json.dumps(result, indent=2, ensure_ascii=False))

