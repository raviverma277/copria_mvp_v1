# core/extraction/field_mapping.py

from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json, re, os
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from copy import deepcopy
import difflib

# Project utilities
from core.utils.llm_status import record_llm_call_start, record_llm_call_end
from core.utils.state import get_state
from core.schemas.active import active_keys, load_active_schema, active_titles

# Optional deps
try:
    import numpy as np
except Exception:
    np = None
try:
    import pandas as pd
except Exception:
    pd = None

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------

SOV_NUMERIC_FIELDS = {
    "tiv_building", "tiv_content", "tiv_bi",
    "bi_indemnity_period_months", "stories", "year_built"
}
LOSS_NUMERIC_FIELDS = {
    "gross_paid", "gross_outstanding", "net_paid", "net_outstanding"
}

_CROSSWALK_PATH = Path("core/schemas/crosswalk.json")
PROV_PREFIX = "_provisional_"

def _strip_provisional(key: str) -> str:
    """Return base name if key starts with _provisional_; else return key."""
    return key[len(PROV_PREFIX):] if isinstance(key, str) and key.startswith(PROV_PREFIX) else key

def _to_jsonable(x):
    # pandas.Timestamp
    if pd is not None and isinstance(x, getattr(pd, "Timestamp", ())):
        return x.isoformat()
    # built-in date/datetime
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    # numpy scalars
    if np is not None:
        if isinstance(x, getattr(np, "integer", ())):
            return int(x)
        if isinstance(x, getattr(np, "floating", ())):
            return float(x)
        if isinstance(x, getattr(np, "bool_", ())):
            return bool(x)
    # Decimal
    if isinstance(x, Decimal):
        try:
            return float(x)
        except Exception:
            return str(x)
    return x

def _json_simplify(obj):
    if isinstance(obj, dict):
        return {k: _json_simplify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_simplify(v) for v in obj]
    return _to_jsonable(obj)

def _norm_tokens(s: str) -> list[str]:
    """Tokenize to lowercase alnum chunks."""
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip().split()

def _norm_join(s: str) -> str:
    """Normalize to a single machine key form (tokens joined by underscores)."""
    return "_".join(_norm_tokens(s))

def _guess_candidates(header: str, allowed_fields: list[str], n: int = 3) -> list[str]:
    """Light heuristic to suggest likely schema keys for a header."""
    h = (header or "").strip().lower().replace("-", " ").replace("_", " ")
    ALIASES = {
        "build type": ["construction"],
        "construction type": ["construction"],
        "bi sum": ["tiv_bi"],
        "business interruption": ["tiv_bi"],
        "paid gross amount": ["gross_paid"],
        "gross paid": ["gross_paid"],
    }
    if h in ALIASES:
        return [a for a in ALIASES[h] if a in allowed_fields]

    scores = difflib.get_close_matches(h, allowed_fields, n=n, cutoff=0.0)
    if "build" in h or "construction" in h:
        if "construction" in allowed_fields and "construction" not in scores:
            scores.insert(0, "construction")
    if "bi" in h or "business interruption" in h:
        if "tiv_bi" in allowed_fields and "tiv_bi" not in scores:
            scores.insert(0, "tiv_bi")
    if "paid" in h and "gross" in h:
        if "gross_paid" in allowed_fields and "gross_paid" not in scores:
            scores.insert(0, "gross_paid")
    out = []
    for s in scores:
        if s in allowed_fields and s not in out:
            out.append(s)
    return out[:n]

# -----------------------------------------------------------------------------
# Crosswalk I/O
# -----------------------------------------------------------------------------

def _read_crosswalk() -> dict:
    try:
        raw = json.loads(_CROSSWALK_PATH.read_text(encoding="utf-8"))
        # normalize keys once so lookups are consistent everywhere
        return {_norm_join(k): v for k, v in raw.items()}
    except Exception:
        return {}
def _write_crosswalk(data: dict) -> None:
    """
    Write the crosswalk mapping to the default crosswalk path.
    Keeps keys/values as provided (caller should pass normalized keys if desired).
    """
    _CROSSWALK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CROSSWALK_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def _read_crosswalk_raw(path: Path) -> dict:
    """
    Read crosswalk.json exactly as-is (no key normalization).
    Used solely for one-time migrations.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_crosswalk_raw(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def migrate_crosswalk_inplace(crosswalk_path: Optional[str] = None) -> dict:
    """
    One-time migration for a *flat* crosswalk:
    - Rewrites targets that start with `_provisional_` to their clean base.
    - Does NOT modify keys (source headers); only the mapped targets.
    Returns a summary dict.
    """
    path = Path(crosswalk_path) if crosswalk_path else _CROSSWALK_PATH
    raw = _read_crosswalk_raw(path)
    if not isinstance(raw, dict):
        raw = {}

    changed = 0
    total = 0
    new_map = {}

    for k, v in raw.items():
        total += 1
        tgt = v
        try:
            if isinstance(tgt, str):
                clean = _strip_provisional(tgt)
                if clean != tgt:
                    changed += 1
                new_map[k] = clean
            else:
                # preserve non-string values as-is
                new_map[k] = tgt
        except Exception:
            new_map[k] = tgt

    _write_crosswalk_raw(path, new_map)
    return {"updated": changed, "total": total, "path": str(path)}


# -----------------------------------------------------------------------------
# Basic header mapping (crosswalk-only) — used by some legacy paths
# -----------------------------------------------------------------------------

def resolve_headers_basic(headers: List[str]) -> Dict[str, str]:
    """
    Crosswalk-only mapping: { original_header -> canonical_field | original_header }.
    NOTE: This does not consult the active schema; primary pipeline should use
    resolve_headers_with_llm (below), which contains parsing-safety logic.
    """
    cw = _read_crosswalk()
    mapping: Dict[str, str] = {}
    for h in headers:
        key = _norm_join(h)
        mapping[h] = cw.get(key, h)
    return mapping

# -----------------------------------------------------------------------------
# Numeric coercions
# -----------------------------------------------------------------------------

_amount_pat = re.compile(r"[^\d\-\.\(\)]")  # strip currency symbols, commas, spaces, letters

def _coerce_amount(val: Any) -> Any:
    """
    Convert '£5,000,000' -> 5000000, '(1,200)' -> -1200, '1.2e3' -> 1200.
    Return original if not parseable.
    """
    if val is None:
        return None
    if isinstance(val, (int, float, Decimal)):
        return val
    s = str(val).strip()
    if s == "":
        return None
    neg = s.startswith("(") and s.endswith(")")
    s2 = _amount_pat.sub("", s)
    if s2 in {"", "-", ".", "-.", ".-"}:
        return None
    try:
        d = Decimal(s2)
        if neg:
            d = -d
        return int(d) if d == d.to_integral_value() else float(d)
    except (InvalidOperation, ValueError):
        return val

def _coerce_row_numbers(row: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
    """Coerce known numeric fields for SOV / Loss Run."""
    fields = SOV_NUMERIC_FIELDS if doc_type == "sov" else (LOSS_NUMERIC_FIELDS if doc_type == "loss_run" else set())
    if not fields:
        return row
    out = dict(row)
    for k in list(row.keys()):
        if k in fields:
            out[k] = _coerce_amount(row[k])
    return out

# -----------------------------------------------------------------------------
# Legacy normalize (keeps unknowns as-is) — primary flow uses resolver below
# -----------------------------------------------------------------------------

def normalize_rows(headers: List[str], rows: List[Dict[str, Any]], doc_type: str) -> List[Dict[str, Any]]:
    """
    Apply crosswalk mapping and coerce numeric fields based on doc_type.
    Unknown headers are kept as-is.
    """
    m = resolve_headers_basic(headers)
    out: List[Dict[str, Any]] = []
    for r in rows:
        norm = {}
        for h, v in r.items():
            tgt = m.get(h, h)
            norm[tgt] = v
        norm = _coerce_row_numbers(norm, doc_type)
        out.append(norm)
    return out

# -----------------------------------------------------------------------------
# OpenAI client plumbing (shared with classifier style)
# -----------------------------------------------------------------------------

def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        return None, None
    try:
        from openai import OpenAI  # >= 1.x
        return OpenAI(api_key=api_key), "responses_v1"
    except Exception:
        try:
            import openai
            openai.api_key = api_key
            return openai, "chat_legacy"
        except Exception:
            return None, None

def _client_caps(client):
    """
    Detect which API surface the installed SDK supports.
    Returns one of: "chat_v1", "responses_v1", "legacy_chat", or "none".
    """
    try:
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            return "chat_v1"
        if hasattr(client, "responses"):
            return "responses_v1"
    except Exception:
        pass
    try:
        import openai  # noqa
        if hasattr(client, "ChatCompletion"):
            return "legacy_chat"
    except Exception:
        pass
    return "none"

# -----------------------------------------------------------------------------
# LLM proposal generation for unknown headers
# -----------------------------------------------------------------------------

def _llm_map_headers(
    doc_type: str,
    unknown_headers: List[str],
    sample_rows: List[Dict[str, Any]],
    allowed_fields: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Ask the LLM to propose mappings ONLY for 'unknown_headers'.
    Returns a proposal object:
      { "doc_type": ..., "mappings":[...], "new_fields":[...] }
    - Keeps raw / pre-filter copies for the JSON debug tab.
    - Validates target_field against ACTIVE schema keys ∪ allowed_fields.
    - Converts low-confidence mappings and invalid targets into new_fields.
    - Ensures any still-unmapped unknown headers appear in new_fields.
    """
    if not unknown_headers:
        return None

    client, _mode = _get_openai_client()
    if not client:
        print("[schema-proposal] OPENAI client not available; skipping LLM proposal")
        return None

    # Build allow-list from ACTIVE schema; union with caller's list for the prompt
    active_field_list = active_keys(doc_type)  # active schema keys
    prompt_allow_list = sorted(set(active_field_list) | set(allowed_fields))
    allow_text = ", ".join(prompt_allow_list)

    # Per-header candidate hints to steer the model
    hints = {h: _guess_candidates(h, prompt_allow_list) for h in unknown_headers}

    # Prompt
    system = (
        "You map spreadsheet column headers to a canonical insurance schema.\n"
        f"Document type: {doc_type}.\n"
        "Rules:\n"
        "- Map ONLY the headers I provide (do NOT add others).\n"
        "- For EACH header, choose EXACTLY ONE target_field from the allowed machine keys list below.\n"
        "- Use machine keys only (not human labels). If you truly cannot map a header, omit it from 'mappings' and do NOT guess.\n"
        "- If a header cannot be mapped, you may propose it as a new field in 'new_fields' with a machine-like field_name and short description.\n"
        "- Return STRICT JSON only:\n"
        "{ \"doc_type\": \"sov|loss_run|questionnaire\","
        "  \"mappings\": [ {\"source_header\":\"...\",\"target_field\":\"<MACHINE_KEY>\",\"confidence\":0-1,\"rationale\":\"...\"} ],"
        "  \"new_fields\": [ {\"field_name\":\"...\",\"description\":\"...\",\"example\":\"...\"} ] }"
    )

    fewshot = (
        "Examples:\n"
        "- Header: 'Build Type' → target_field: 'construction'\n"
        "- Header: 'BI Sum' → target_field: 'tiv_bi'\n"
        "- Header: 'Paid Gross Amount' → target_field: 'gross_paid'\n"
    )

    safe_samples = _json_simplify(sample_rows[:3])
    sample_text = json.dumps(safe_samples, ensure_ascii=False, default=str)

    user = (
        "Allowed machine keys:\n"
        f"{allow_text}\n\n"
        f"{fewshot}\n"
        "Map ONLY these headers (candidate hints in parentheses):\n"
        + "\n".join([f"- {h} (candidates: {hints[h]})" for h in unknown_headers]) +
        "\n\n"
        f"Sample rows (first 3):\n{sample_text}\n\n"
        "Return only the JSON as specified above."
    )

    model_chat = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    model_legacy = os.environ.get("OPENAI_MODEL", "gpt-4-0613")
    record_llm_call_start(model_chat)

    try:
        caps = _client_caps(client)
        if caps == "chat_v1":
            resp = client.chat.completions.create(
                model=model_chat,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": user}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
        elif caps == "responses_v1":
            resp = client.responses.create(
                model=model_chat,
                input=[{"role": "system", "content": system},
                       {"role": "user",   "content": user}],
                temperature=0
            )
            try:
                content = resp.output_text
            except Exception:
                content = getattr(resp, "content", [{"text": {"value": ""}}])[0]["text"]["value"]
        else:
            resp = client.ChatCompletion.create(
                model=model_legacy,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": user}],
                temperature=0
            )
            content = resp["choices"][0]["message"]["content"]

        record_llm_call_end(success=True)
        print("[schema-proposal] LLM proposal received")

        # ---- DEBUG: save raw LLM content (pre-parse) ----
        try:
            state = get_state()
            state.setdefault("proposals_raw", {}).setdefault(doc_type, []).append(content)
        except Exception:
            pass

        # Parse
        data = json.loads(content)

        # ---- DEBUG: save TRUE pre-filter snapshot ----
        try:
            state = get_state()
            state.setdefault("proposals_parsed", {}).setdefault(doc_type, []).append(deepcopy(data))
        except Exception:
            pass

        # Shape defaults
        if not isinstance(data, dict):
            return None
        data.setdefault("doc_type", doc_type)
        data.setdefault("mappings", [])
        data.setdefault("new_fields", [])

        # ---------- VALIDATION / REPAIR ----------
        # Active schema keys for validation
        active_path, _ = load_active_schema(doc_type)
        active_set = {_norm_join(k) for k in active_field_list}
        passed_set = {_norm_join(a) for a in allowed_fields}
        allowed_union = active_set | passed_set
        print(f"[schema-proposal] allowed keys ({doc_type} @ {active_path.name}): "
              f"{len(allowed_union)} keys; sample: {sorted(list(allowed_union))[:10]}")

        unknown_norm = {_norm_join(h) for h in unknown_headers}

        # 1) Accept only mappings whose SRC is in the unknown set AND TGT is allowed
        ACCEPT_MIN = 0.75
        kept_mappings: List[Dict[str, Any]] = []
        downgraded_new: List[Dict[str, Any]] = []
        invalid_to_new: List[Dict[str, Any]] = []

        for m in list(data.get("mappings", [])):
            src_raw = (m.get("source_header") or "").strip()
            tgt_raw = (m.get("target_field") or "").strip()
            src = _norm_join(src_raw)
            tgt = _norm_join(tgt_raw)

            # clamp confidence
            try:
                conf = float(m.get("confidence", 0.7))
            except Exception:
                conf = 0.7
            conf = max(0.0, min(1.0, conf))

            if (src in unknown_norm) and (tgt in allowed_union):
                if conf >= ACCEPT_MIN:
                    m["confidence"] = conf
                    kept_mappings.append(m)
                else:
                    downgraded_new.append({
                        "field_name": _norm_join(src_raw) or _norm_join(tgt_raw) or "unknown_field",
                        "description": f"Suggested new field (low-confidence mapping {conf:.2f}). {m.get('rationale','')}".strip(),
                        "example": None
                    })
            else:
                invalid_to_new.append({
                    "field_name": _norm_join(src_raw) or _norm_join(tgt_raw) or "unknown_field",
                    "description": f"Suggested new field (invalid mapping). {m.get('rationale','')}".strip(),
                    "example": None
                })

        if downgraded_new:
            print(f"[schema-proposal] downgraded {len(downgraded_new)} low-confidence mapping(s) to new_fields")
        if invalid_to_new:
            print(f"[schema-proposal] moved {len(invalid_to_new)} invalid mapping(s) to new_fields")

        data["mappings"] = kept_mappings
        data["new_fields"].extend(downgraded_new)
        data["new_fields"].extend(invalid_to_new)

        # 2) If some unknown headers still weren’t mentioned at all, add them as new_fields
        mentioned_srcs = {_norm_join(m.get("source_header")) for m in kept_mappings}
        for h in unknown_headers:
            if _norm_join(h) not in mentioned_srcs:
                fn = _norm_join(h)
                if fn and fn not in {_norm_join(nf.get("field_name")) for nf in data["new_fields"]}:
                    example = None
                    try:
                        if sample_rows and isinstance(sample_rows[0], dict) and h in sample_rows[0]:
                            example = _to_jsonable(sample_rows[0][h])
                    except Exception:
                        pass
                    data["new_fields"].append({
                        "field_name": fn,
                        "description": "Header not in active schema; propose new field.",
                        "example": example
                    })

        return data

    except Exception as e:
        record_llm_call_end(success=False)
        print(f"[schema-proposal] LLM proposal failed: {e}")
        return None


# -----------------------------------------------------------------------------
# Primary resolver used by the pipeline (parsing safety added here)
# -----------------------------------------------------------------------------

def resolve_headers_with_llm(doc_type: str, headers, rows):
    """
    Resolve headers using crosswalk + active schema + (optionally) LLM for unknowns.
    Unknown headers are carried through into normalized rows as provisional columns:
      _provisional_<normalized_header>
    Returns:
      {
        "normalized_rows": [...],     # includes mapped keys + provisional_* keys
        "source_rows": [...],         # original row dicts
        "provisional_fields": [...],  # list of provisional field names included
        "proposals": {...}|None       # LLM proposals for unknown headers
      }
    """
    headers = headers or []
    rows = rows or []

    xwalk = _read_crosswalk()                # { normalized_header -> schema_key (maybe provisional) }
    active_keys_list = active_keys(doc_type) # active schema keys (canonical)
    titles_map = active_titles(doc_type)     # key -> title
    title_to_key = {_norm_join(v): k for k, v in titles_map.items() if v}

    # Fast maps for lookups
    active_norm_to_canon = {_norm_join(k): k for k in active_keys_list}

    # 1) Build known_map from headers
    known_map: Dict[str, str] = {}
    for h in headers:
        nh = _norm_join(h)
        target = None

        # (a) explicit crosswalk
        if nh in xwalk:
            tgt = xwalk[nh]
            base = _strip_provisional(tgt)
            # PARSING-SAFETY: if crosswalk still points to provisional but the base is now in the active schema,
            # resolve to the base key so normalized rows use the promoted field.
            target = base if base in active_keys_list else tgt

        # (b) header is already a machine key (by key or normalized key)
        elif h in active_keys_list or nh in active_norm_to_canon:
            target = active_norm_to_canon.get(nh, h)

        # (c) header matches a schema "title"
        elif nh in title_to_key:
            target = title_to_key[nh]

        if target:
            known_map[h] = target

    # 2) Unknown headers
    unknown_headers = [h for h in headers if h not in known_map]

    # 3) Ask LLM ONLY for unknowns (if any)
    proposals = None
    if unknown_headers:
        proposals = _llm_map_headers(
            doc_type=doc_type,
            unknown_headers=unknown_headers,
            sample_rows=rows,
            allowed_fields=active_keys_list,  # pass active keys to the model
        )

        # Optional: auto-apply only very high-confidence mappings immediately
        if proposals and proposals.get("mappings"):
            for m in proposals["mappings"]:
                try:
                    src = (m.get("source_header") or "").strip()
                    tgt = (m.get("target_field") or "").strip()
                    conf = float(m.get("confidence", 0.0))
                    if src and tgt and conf >= 0.90:
                        known_map[src] = tgt
                        if src in unknown_headers:
                            unknown_headers.remove(src)
                except Exception:
                    pass

    # 4) Build normalized rows:
    #    - Put mapped fields under their canonical key
    #    - Carry unknown headers through as provisional columns
    provisional_fields: List[str] = []
    normalized_rows: List[Dict[str, Any]] = []

    prov_name_for = {h: f"{PROV_PREFIX}{_norm_join(h)}" for h in unknown_headers}
    provisional_fields = list(prov_name_for.values())

    for r in rows:
        out: Dict[str, Any] = {}
        # mapped keys
        for h, v in r.items():
            if h in known_map:
                out[known_map[h]] = v
        # provisional (unknown) keys
        for h in unknown_headers:
            if h in r:
                out[prov_name_for[h]] = r[h]

        # Coerce known numeric fields (SOV/LOSS lists) without touching provisional
        out = _coerce_row_numbers(out, doc_type)

        normalized_rows.append(out)

    return {
        "normalized_rows": normalized_rows,
        "source_rows": rows,
        "provisional_fields": provisional_fields,
        "proposals": proposals
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate crosswalk.json targets from _provisional_* → clean base.")
    parser.add_argument("--crosswalk", default=str(_CROSSWALK_PATH), help="Path to crosswalk.json (default: core/schemas/crosswalk.json)")
    args = parser.parse_args()
    summary = migrate_crosswalk_inplace(args.crosswalk)
    print(json.dumps(summary, indent=2))
