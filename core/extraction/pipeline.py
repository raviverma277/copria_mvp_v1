# core/extraction/pipeline.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

from ..parsing.dispatch import parse_files
from .field_mapping import resolve_headers_with_llm
from ..validation.json_schema_validator import validate_rows
from .normalizers import normalize_sov_rows, normalize_loss_rows

# NEW: profiler & baseline suggestion converter
from core.schemas.profiler import (
    profile_rows,
    profile_to_suggestions,
)  # ✅ correct source
from core.schemas.active import (
    load_active_schema,
    active_keys,
)  # ✅ to fetch active schema dicts
from core.extraction.field_mapping import _strip_provisional, _read_crosswalk

# --- Risk reasoning & justifications ---
from core.risk.rules import rules_from_bundle
from core.risk.reasoner import justify_with_llm, dedupe_and_cap
from core.utils.events import publish
from core.risk.redflags import evaluate_red_flags, _load_rules
from core.risk.evidence import (
    pack_sov_snippets,
    pack_loss_snippets,
    pack_notes_snippets,
    attach_topk_evidence_to_items,
    build_llm_context,
)
from core.risk.miner import mine_additional_risks
from core.config import get_config
from core.utils.events import publish
from core.risk.miner import _uid_from_item  # reuse the helper
from core.risk.pipeline import run_risk_pipeline
from core.config import QUICK_RISKS_ENABLED
from dataclasses import is_dataclass, asdict



import hashlib, re
from typing import Dict, Any, List

# --- Pricing imports ---
from core.pricing.pricing_contracts import SubmissionPricingInput, PricingResult, PricingRange
from core.pricing.pricing_benchmark import PriceBenchmark
from core.pricing.pricing_engine import RulesPricer
from core.pricing.strategy_factory import get_pricer
# Optional LLM refinement (USD):
# from pricing.llm_pricer import LLMPricerBlend

import streamlit as st
import os


# --- Samples helper for LLM enrichment ---
from typing import Any, Dict, List


def collect_field_samples(
    rows: List[Dict[str, Any]], fields: List[str], limit_per_field: int = 30
) -> Dict[str, List[Any]]:
    """
    Collect up to N non-null examples per candidate field from normalized rows.
    Returns: { field: [samples...] }   (missing/empty lists are omitted)
    """
    targets = {f: [] for f in fields}
    for row in rows:
        for f in fields:
            if len(targets[f]) >= limit_per_field:
                continue
            if f in row and row[f] is not None:
                targets[f].append(row[f])
    return {k: v for k, v in targets.items() if v}


def _email_envs_from_submission_bundle(sb_dict: dict) -> list[dict]:
    """
    Turn SubmissionBundle.attachments (with role='email') into results['email_envelopes'].
    Maps keys to what the Email tab expects:
      - from_addr -> from
      - sent_at   -> sent_date
    Also stitches child attachments by parent_id with filename/content_type.
    """
    out = []
    atts = (sb_dict or {}).get("attachments") or []
    if not isinstance(atts, list):
        return out

    # index attachments by id so we can find children quickly
    by_id = {a.get("id"): a for a in atts if isinstance(a, dict)}

    # find all email envelope attachments
    email_atts = [
        a
        for a in atts
        if (isinstance(a, dict) and str(a.get("role", "")).lower() == "email")
    ]

    for e in email_atts:
        env = (
            (e.get("email_envelope") or {})
            if isinstance(e.get("email_envelope"), dict)
            else {}
        )
        rec = {
            "from": env.get("from") or env.get("from_addr"),
            "to": env.get("to") or [],
            "cc": env.get("cc") or [],
            "subject": env.get("subject"),
            "sent_date": env.get("sent_at") or env.get("date") or env.get("sent_date"),
            "body_text": env.get("body_text")
            or "",  # not provided in sample; keep empty
            "attachments": [],
        }

        # stitch children (attachments that point to this email by parent_id)
        eid = e.get("id")
        if eid:
            children = [a for a in atts if a.get("parent_id") == eid]
            for c in children:
                rec["attachments"].append(
                    {
                        "filename": c.get("name"),
                        "content_type": c.get("mime_type"),
                        "role": c.get("role"),
                    }
                )
        out.append(rec)
    return out

# ==== QRA: helpers (safe to add) ====

# helper: coerce dataclass/object into a shallow dict
def _coerce_to_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    if obj is None:
        return {}
    # dataclass
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            pass
    # custom object with to_dict()
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            d = to_dict()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    # generic object
    if hasattr(obj, "__dict__"):
        try:
            # shallow copy; also coerce a few obvious nested attrs if they’re dataclasses
            d = dict(obj.__dict__)
            for k, v in list(d.items()):
                if is_dataclass(v):
                    d[k] = asdict(v)
                elif hasattr(v, "__dict__") and not isinstance(v, (dict, list, str, int, float, bool)):
                    d[k] = dict(v.__dict__)
            return d
        except Exception:
            pass
    return {}

def _get_key_or_attr(container, key, default=None):
    """Get dict[key] or getattr(container, key) with a default."""
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip().lower())
    return re.sub(r"-+", "-", s).strip("-")

def _stamp_uid(risk: Dict[str, Any]) -> Dict[str, Any]:
    code = risk.get("code", "UNK")
    locs = "|".join(sorted(risk.get("locations", [])))
    title = _slugify(risk.get("title", ""))
    basis = f"{code}|{locs}|{title}"
    risk["uid"] = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
    return risk

# Field alias helper to be resilient to schema drift
FIELD_ALIASES = {
    "location_id": ["location_id", "locationId", "id"],
    "sprinklers": ["sprinklers", "fire_sprinklers", "is_sprinklered", "sprinklered"],
    "fire_alarm": ["fire_alarm", "is_fire_alarm_present", "alarm_system"],
    "roof_age_years": ["roof_age_years", "roofAgeYears", "roof_age", "roofAge"],
}

def _get(record: Dict[str, Any], key: str, default=None):
    for k in FIELD_ALIASES.get(key, [key]):
        if k in record and record[k] not in (None, "", "Unknown"):
            return record[k]
        pk = f"_provisional_{k}"
        if pk in record and record[pk] not in (None, "", "Unknown"):
            return record[pk]
    return default

def _as_bool(x) -> bool:
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return x != 0
    if isinstance(x, str): return x.strip().lower() in {"y","yes","true","1","present"}
    return False

# ====  quick_risks_from_bundle  ====
def quick_risks_from_bundle(bundle) -> List[Dict[str, Any]]:
    """
    Lightweight, deterministic red flags over the SubmissionBundle (contract-first).
    Supports legacy shapes, too.
    Looks at:
      - No sprinklers / no fire alarm
      - Roof age > 20
      - Flood zone high (if present) / wildfire high (if present)
      - Prior large claim (> $100k incurred or paid)
    """
    if not QUICK_RISKS_ENABLED:
        return []

    quick: List[Dict[str, Any]] = []

    # Normalize bundle to a dict first, but we’ll also allow attr fallbacks
    sb = _coerce_to_dict(bundle) or bundle or {}

    # 1) Pull "locations" from multiple possible shapes
    locations = []

     # Contract-first (recommended): sb["sov"]["records"] OR object.sov.records
    sov = _get_key_or_attr(sb, "sov")
    if sov is None and not isinstance(sb, dict):  # if sb is still an object
        sov = _get_key_or_attr(bundle, "sov")

    sov_dict = _coerce_to_dict(sov)
    sov_records = _get_key_or_attr(sov_dict, "records") or _get_key_or_attr(sov, "records") or []
    if isinstance(sov_records, list) and sov_records:
        locations = sov_records
    else:
        # Legacy fallbacks:
        # - sb["locations"]
        # - sb["normalized"]["locations"]
        locations = _get_key_or_attr(sb, "locations") \
            or _get_key_or_attr(_get_key_or_attr(sb, "normalized", {}), "locations") \
            or []

    # 2) Deterministic per-location checks
    for loc in locations or []:
        loc_id = _get(loc, "location_id", default="UNKNOWN-LOC")

        # No sprinklers
        sprinklers = _get(loc, "sprinklers")
        if sprinklers is not None and not _as_bool(sprinklers):
            quick.append(_stamp_uid({
                "code": "FIRE-NO-SPRINKLER",
                "title": "No sprinklers",
                "severity": "high",
                "locations": [loc_id],
                "tags": ["quick"],
                "rationale": "Location indicates no automatic sprinkler protection.",
                "evidence_refs": [{"loc_id": loc_id, "field": "sprinklers", "value": sprinklers}],
            }))

        # No fire alarm
        fire_alarm = _get(loc, "fire_alarm")
        if fire_alarm is not None and not _as_bool(fire_alarm):
            quick.append(_stamp_uid({
                "code": "FIRE-NO-ALARM",
                "title": "No fire alarm",
                "severity": "medium",
                "locations": [loc_id],
                "tags": ["quick"],
                "rationale": "No fire/alarm system present.",
                "evidence_refs": [{"loc_id": loc_id, "field": "fire_alarm", "value": fire_alarm}],
            }))

        # Roof age > 20
        roof_age = _get(loc, "roof_age_years")
        try:
            if roof_age is not None and float(roof_age) > 20:
                quick.append(_stamp_uid({
                    "code": "ROOF-AGED",
                    "title": f"Roof age {int(float(roof_age))} yrs",
                    "severity": "medium",
                    "locations": [loc_id],
                    "tags": ["quick"],
                    "rationale": "Roof age exceeds 20 years.",
                    "evidence_refs": [{"loc_id": loc_id, "field": "roof_age_years", "value": roof_age}],
                }))
        except Exception:
            pass

        # Optional: Flood & Wildfire quick checks (only if obvious fields are present)
        flood = _get(loc, "flood_zone") or _get(loc, "flood_risk")
        if isinstance(flood, str) and flood.strip().upper() in {"AE", "A", "V", "VE", "HIGH"}:
            quick.append(_stamp_uid({
                "code": "FLOOD-HIGH",
                "title": "High flood zone",
                "severity": "high",
                "locations": [loc_id],
                "tags": ["quick"],
                "rationale": f"Flood zone '{str(flood)}' indicates elevated flood risk.",
                "evidence_refs": [{"loc_id": loc_id, "field": "flood_zone", "value": flood}],
            }))

        wild = _get(loc, "wildfire_risk") or _get(loc, "wildfire_score")
        try:
            # accept categorical or numeric (>= 4/5 or >= 70/100)
            if isinstance(wild, str) and wild.strip().lower() in {"high", "very high"}:
                wild_hit = True
            elif wild is not None and float(wild) >= 4:   # 1..5 scale
                wild_hit = True
            elif wild is not None and float(wild) >= 70:  # 0..100 scale
                wild_hit = True
            else:
                wild_hit = False
        except Exception:
            wild_hit = False

        if wild_hit:
            quick.append(_stamp_uid({
                "code": "WILDFIRE-HIGH",
                "title": "High wildfire risk",
                "severity": "high",
                "locations": [loc_id],
                "tags": ["quick"],
                "rationale": "Wildfire risk score/category indicates elevated risk.",
                "evidence_refs": [{"loc_id": loc_id, "field": "wildfire_risk", "value": wild}],
            }))

    # 3) Prior large claims (scan loss_run.records if present) — resilient to dicts & dataclasses
    # Reuse 'sb' if the function already created it; otherwise derive it now
    try:
        sb  # defined earlier in the function when reading SOV
    except NameError:
        sb = _coerce_to_dict(bundle) or bundle or {}

    loss = _get_key_or_attr(sb, "loss_run")
    if loss is None and not isinstance(sb, dict):
        # If sb is still an object, also try attribute access on the original bundle
        loss = _get_key_or_attr(bundle, "loss_run")

    loss_dict = _coerce_to_dict(loss)
    loss_records = _get_key_or_attr(loss_dict, "records") or _get_key_or_attr(loss, "records") or []

    # Normalize each row to a dict so .get(...) works even if rows are dataclass instances
    norm_loss_records = []
    for row in (loss_records or []):
        norm_loss_records.append(row if isinstance(row, dict) else _coerce_to_dict(row))

    THRESH = 100_000  # tune as desired
    large_claims = []
    for lr in norm_loss_records:
        # be tolerant of keys
        amt = None
        for k in ["incurred", "paid", "claim_amount", "loss_amount"]:
            v = lr.get(k) or lr.get(f"_provisional_{k}")
            try:
                if v not in (None, "", "Unknown"):
                    amt = float(str(v).replace(",", "").replace("$", ""))
                    break
            except Exception:
                pass
        if amt is not None and amt >= THRESH:
            # try to extract a location id if present
            loc_id = _get(lr, "location_id", default=None) or _get(lr, "loc_id", default=None)
            large_claims.append((amt, loc_id))

    if large_claims:
        # aggregate into one quick item; include the worst example in rationale
        top = max(large_claims, key=lambda x: x[0])
        locs = sorted({lc[1] for lc in large_claims if lc[1]}) or []
        quick.append(_stamp_uid({
            "code": "PRIOR-LARGE-CLAIMS",
            "title": "Prior large claim(s) detected",
            "severity": "high",
            "locations": locs,
            "tags": ["quick"],
            "rationale": f"At least one claim ≥ ${THRESH:,.0f} (max observed ≈ ${top[0]:,.0f}).",
            "evidence_refs": [{"loc_id": lc[1], "field": "incurred/paid", "value": lc[0]} for lc in large_claims[:3]],
        }))


    return quick



def merge_quick_into_full(full_risks: List[Dict[str, Any]], quick_risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_uid = {r.get("uid"): r for r in full_risks if r.get("uid")}
    out = list(full_risks)
    for q in quick_risks:
        if "uid" not in q: q = _stamp_uid(q)
        uid = q["uid"]
        if uid in by_uid:
            r = by_uid[uid]
            r["tags"] = sorted(set((r.get("tags") or [])) | {"quick"})
            if not r.get("rationale"): r["rationale"] = q.get("rationale")
            if not r.get("evidence_refs"): r["evidence_refs"] = q.get("evidence_refs")
        else:
            out.append(q)
    return out
# ====================================


# ---------- Pricing helpers (derived metrics + serialization) ----------
def _sum_tiv(rows: list[dict]) -> float:
    """
    Sum Total Insured Value from normalized SOV rows.
    Prefers 'total_tiv' if present, else falls back to tiv_building + tiv_contents.
    """
    total = 0.0
    for r in rows or []:
        try:
            if r.get("total_tiv") not in (None, ""):
                total += float(r.get("total_tiv") or 0)
            else:
                tb = float(r.get("tiv_building") or 0)
                tc = float(r.get("tiv_contents") or 0)
                total += (tb + tc)
        except Exception:
            pass
    return float(total)

def _approx_cope_score(rows: list[dict]) -> float:
    """
    Very simple, explainable COPE proxy from normalized SOV fields.
    You likely already compute a COPE score elsewhere; if so, replace this with that value.
    """
    if not rows:
        return 70.0
    # Start from 70 and nudge
    score = 70.0
    n = min(len(rows), 200)
    has_sprinklers = 0
    has_alarm = 0
    roof_age_sum = 0.0
    flood_hits = 0
    wildfire_hits = 0

    for r in rows[:n]:
        if r.get("sprinklered") is True: has_sprinklers += 1
        if str(r.get("fire_alarm")).strip().lower() in {"true", "yes", "y", "1"}:
            has_alarm += 1
        try:
            roof_age_sum += float(r.get("roof_age_years") or 0)
        except Exception:
            pass
        if str(r.get("flood_zone")).strip().lower() not in {"", "none", "no", "false"}:
            flood_hits += 1
        if str(r.get("wildfire_risk")).strip().lower() not in {"", "none", "no", "false"}:
            wildfire_hits += 1

    # Normalize by sample
    p_spr = has_sprinklers / n
    p_alarm = has_alarm / n
    avg_roof = (roof_age_sum / n) if n else 0.0
    p_flood = flood_hits / n
    p_wild = wildfire_hits / n

    score += 10.0 * p_spr
    score += 5.0 * p_alarm
    score -= 0.2 * max(0.0, avg_roof - 10.0)  # older roofs reduce
    score -= 8.0 * p_flood
    score -= 6.0 * p_wild
    return max(0.0, min(100.0, score))

def _cope_index_from_breakdown(cope_obj: dict) -> float:
    """
    Convert the Risk tab's COPE 'points' breakdown (e.g., {'construction':2,...})
    into a 0..100 COPE index where 100 = best risk.
    Assumes each dimension is ~0..10 points; adjust if your scale changes.
    """
    bd = (cope_obj or {}).get("breakdown") or {}
    if not bd:
        return 70.0  # neutral default
    dims = max(1, len(bd))
    max_points = 10.0 * dims
    points = sum(float(v or 0) for v in bd.values())
    idx = 100.0 - (points / max_points) * 100.0  # invert: lower points => higher index
    return float(max(0.0, min(100.0, round(idx, 2))))



def _loss_ratio_proxy(loss_rows: list[dict], tiv_total: float) -> float:
    """
    Proxy loss ratio when premium history is unavailable:
    sum(incurred) / TIV (clamped 0..1). Replace with true premium-based LR when available.
    """
    incurred = 0.0
    for r in loss_rows or []:
        try:
            incurred += float(r.get("incurred") or 0)
        except Exception:
            pass
    if tiv_total <= 0:
        return 0.0
    lr = incurred / tiv_total
    return float(max(0.0, min(1.0, lr)))

def _pricing_result_to_dict(pr: PricingResult) -> dict:
    return {
        "submission_id": pr.submission_id,
        "percentiles": [
            {
                "metric": p.metric,
                "value": p.value,
                "percentile": p.pctl,
                "median": p.median,
                "iqr": p.iqr,
            } for p in (pr.percentiles or [])
        ],
        "pricing_range": {
            "premium_min": pr.pricing_range.premium_min,
            "premium_median": pr.pricing_range.premium_median,
            "premium_max": pr.pricing_range.premium_max,
            "currency": pr.pricing_range.currency,
        },
        "confidence": pr.confidence,
        "confidence_label": pr.confidence_label,
        "reason_codes": pr.reason_codes,
        "llm_explainer": pr.llm_explainer,
    }
# ----------------------------------------------------------------------


def run_extraction_pipeline(
    files: Optional[List[Any]] = None,
    parsed_bundle: Optional[Dict[str, Any]] = None,
    submission_bundle: Optional[Dict[str, Any]] = None,  # <-- NEW
) -> Dict[str, Any]:
    """
    If parsed_bundle is provided (from Classification), reuse it.
    Else, if submission_bundle is provided (e.g., Cytora connector), coerce it into the
    classified 'bundle' shape this pipeline expects.
    Otherwise parse files fresh. Returns a minimal structure that the UI expects.
    """

    def _coerce_submission_bundle_to_parsed(sb: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a SubmissionBundle-like dict into the 'classified bundle' shape used downstream:
          {
            "submission": [],
            "sov": [ { "sheets": [ { "headers": [...], "rows": [...] } ] } ],
            "loss_run": [ { "sheets": [ { "headers": [...], "rows": [...] } ] } ],
            "questionnaire": [],
            "email_body": [],
            "other": []
          }
        Currently supports sb.get("sov") and sb.get("loss_run") when of form {"records":[{...}, ...]}.
        """
        out: Dict[str, Any] = {
            "submission": [],
            "sov": [],
            "loss_run": [],
            "questionnaire": [],
            "email_body": [],
            "other": [],
        }

        try:
            sov = (sb or {}).get("sov")
            if isinstance(sov, dict) and isinstance(sov.get("records"), list):
                rows = [r for r in sov["records"] if isinstance(r, dict)]
                # derive headers from union of keys across rows
                headers = sorted({k for r in rows for k in r.keys()})
                out["sov"] = [
                    {
                        "sheets": [
                            {"name": "Cytora SOV", "headers": headers, "rows": rows}
                        ]
                    }
                ]

            loss = (sb or {}).get("loss_run")
            if isinstance(loss, dict) and isinstance(loss.get("records"), list):
                rows = [r for r in loss["records"] if isinstance(r, dict)]
                headers = sorted({k for r in rows for k in r.keys()})
                out["loss_run"] = [
                    {
                        "sheets": [
                            {"name": "Cytora Loss", "headers": headers, "rows": rows}
                        ]
                    }
                ]

            # Optional note passthrough: treat freeform notes as questionnaire context
            notes = (sb or {}).get("notes")
            if isinstance(notes, str) and notes.strip():
                out["questionnaire"] = [{"filename": "notes", "text": notes.strip()}]

        except Exception:
            # Fail soft: leave out sections we couldn't coerce
            pass

        return out

    # --- Decide source of 'bundle' ---
    if parsed_bundle is not None:
        bundle = parsed_bundle
    elif submission_bundle is not None:
        # Accept both a dataclass .to_dict() shape and plain dicts
        sb = submission_bundle
        # If it's a dataclass-like object, try to get a dict view
        if hasattr(sb, "to_dict") and callable(getattr(sb, "to_dict")):
            try:
                sb = sb.to_dict()
            except Exception:
                sb = dict(getattr(sb, "__dict__", {}) or {})
        bundle = _coerce_submission_bundle_to_parsed(sb if isinstance(sb, dict) else {})
        # Fallback to empty default if nothing could be coerced
        if not any(
            bundle.get(k) for k in ("sov", "loss_run", "questionnaire", "email_body")
        ):
            bundle = {
                "submission": [],
                "sov": [],
                "loss_run": [],
                "questionnaire": [],
                "email_body": [],
                "other": [],
            }
    else:
        bundle = (
            parse_files(files)
            if files
            else {
                "submission": [],
                "sov": [],
                "loss_run": [],
                "questionnaire": [],
                "email_body": [],
                "other": [],
            }
        )

    # --- Pull email envelopes from SubmissionBundle.attachments (if present) ---
    email_envs_from_sb: List[Dict[str, Any]] = []
    if submission_bundle is not None:
        sb_obj = submission_bundle
        # dataclass-style object -> dict
        if hasattr(sb_obj, "to_dict") and callable(getattr(sb_obj, "to_dict")):
            try:
                sb_dict = sb_obj.to_dict()
            except Exception:
                sb_dict = dict(getattr(sb_obj, "__dict__", {}) or {})
        elif isinstance(sb_obj, dict):
            sb_dict = sb_obj
        else:
            sb_dict = {}

        email_envs_from_sb = _email_envs_from_submission_bundle(sb_dict)

    # --- Submission (placeholder) ---
    submission_core = {"insured_name": "ACME LTD", "currency": "GBP"}

    # --- Helpers / imports ---
    def _norm_join(s: str) -> str:       

        return "_".join(
            re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip().split()
        )

    def _active_set(dt: str) -> set[str]:
        try:
            return set(active_keys(dt))
        except Exception:
            return set()

    # --- Boolean coercion + sprinkler repair -------------------------------------
    def _coerce_bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
        return None  # unknown stays None

    def _get_sprinkler_value(row: dict):
        # honor common aliases you may see in wild data
        for k in ("sprinklered", "is_sprinklered", "sprinklers", "sprinkler"):
            if k in row:
                return _coerce_bool(row.get(k))
        return None

    def _repair_sprinklered(normalized_rows, source_rows):
        """
        Ensure normalized_rows[i]['sprinklered'] is a proper boolean when the source row had it.
        We never assume 'unknown' -> False; only set when we can coerce to True/False.
        """
        out = []
        for i, norm in enumerate(normalized_rows):
            norm = dict(norm)  # shallow copy
            val = norm.get("sprinklered", None)
            # Coerce if present as a string like "True"/"False"
            coerced = _coerce_bool(val)
            if coerced is not None:
                norm["sprinklered"] = coerced
            else:
                # Fall back to source row (same order, you already preserved rows)
                src = source_rows[i] if i < len(source_rows) else {}
                src_val = _get_sprinkler_value(src)
                if src_val is not None:
                    norm["sprinklered"] = src_val
            out.append(norm)
        return out

    def _enrich_sov_from_source(normalized_rows, source_rows):
        """
        If normalization dropped/renamed fields, refill from source.
        Also compute total_tiv if missing (building + contents).
        """
        KEYS = [
            "fire_alarm",
            "hazardous_materials",
            "flood_zone",
            "wildfire_risk",
            "earthquake_exposure",
            "roof_age_years",
            "number_of_stories",
            "tiv_building",
            "tiv_contents",
            "total_tiv",
            "location_id",
            "address",
        ]
        out = []
        for i, norm in enumerate(normalized_rows):
            norm = dict(norm)
            src = source_rows[i] if i < len(source_rows) else {}
            # bring over values that are missing/None in the normalized row
            for k in KEYS:
                if norm.get(k) in (None, "") and (k in src):
                    norm[k] = src[k]
            # compute total_tiv if not present
            if norm.get("total_tiv") in (None, ""):
                try:
                    tb = float(norm.get("tiv_building") or 0)
                    tc = float(norm.get("tiv_contents") or 0)
                    norm["total_tiv"] = tb + tc
                except Exception:
                    pass
            out.append(norm)
        return out

    def _coerce_num(v):
        try:
            if v is None or v == "":
                return None
            return float(v)
        except Exception:
            return None

    def _enrich_loss_from_source(normalized_rows, source_rows):
        """
        Ensure key loss columns survive normalization and are numeric where appropriate.
        """
        KEYS = [
            "policy_year",
            "claim_number",
            "loss_date",
            "cause",
            "location_id",
            "status",
            "paid",
            "open_reserve",
            "incurred",
        ]
        out = []
        for i, norm in enumerate(normalized_rows):
            norm = dict(norm)
            src = source_rows[i] if i < len(source_rows) else {}
            for k in KEYS:
                if norm.get(k) in (None, "") and (k in src):
                    norm[k] = src[k]
            # numeric coercion
            for k in ("paid", "open_reserve", "incurred"):
                if k in norm:
                    norm[k] = _coerce_num(norm.get(k))
            out.append(norm)
        return out

    # -----------------------------------------------------------------------------

    # --- SOV with LLM proposals ---
    sov_rows: List[Dict[str, Any]] = []
    sov_source_rows: List[Dict[str, Any]] = []
    sov_proposals: List[Dict[str, Any]] = []

    for excel in bundle.get("sov", []):
        for sh in excel.get("sheets", []):
            headers, rows = sh.get("headers", []), sh.get("rows", [])
            res = resolve_headers_with_llm("sov", headers, rows)

            # ensure preview has source rows
            sov_source_rows.extend(rows[:500])

            # if LLM proposed mappings, make sure those source headers exist on rows (for preview)
            if res.get("proposals") and res["proposals"].get("mappings"):
                unknown_headers = [
                    m["source_header"]
                    for m in res["proposals"]["mappings"]
                    if m.get("source_header")
                ]
                if unknown_headers:
                    for row in rows:
                        for h in unknown_headers:
                            row.setdefault(h, None)

            sov_rows.extend(res.get("normalized_rows", [])[:500])
            if res.get("proposals"):
                sov_proposals.append(res["proposals"])

    # --- Loss Run with LLM proposals ---
    loss_rows: List[Dict[str, Any]] = []
    loss_source_rows: List[Dict[str, Any]] = []
    loss_proposals: List[Dict[str, Any]] = []

    for excel in bundle.get("loss_run", []):
        for sh in excel.get("sheets", []):
            headers, rows = sh.get("headers", []), sh.get("rows", [])
            res = resolve_headers_with_llm("loss_run", headers, rows)

            loss_source_rows.extend(rows[:2000])

            if res.get("proposals") and res["proposals"].get("mappings"):
                unknown_headers = [
                    m["source_header"]
                    for m in res["proposals"]["mappings"]
                    if m.get("source_header")
                ]
                if unknown_headers:
                    for row in rows:
                        for h in unknown_headers:
                            row.setdefault(h, None)

            loss_rows.extend(res.get("normalized_rows", [])[:2000])
            if res.get("proposals"):
                loss_proposals.append(res["proposals"])

    # --- Normalize for UI/validation ---
    sov_rows = normalize_sov_rows(sov_rows)
    loss_rows = normalize_loss_rows(loss_rows)

    # Ensure 'sprinklered' survives as True/False if present in source
    sov_rows = _repair_sprinklered(sov_rows, sov_source_rows)

    # Ensure other red-flag inputs survive normalization
    sov_rows = _enrich_sov_from_source(sov_rows, sov_source_rows)

    # enrich LOSS so amounts and fields persist from the source
    loss_rows = _enrich_loss_from_source(loss_rows, loss_source_rows)

    # --- Build evidence snippets (SOV, Loss, Notes) ---
    notes_text = None
    try:
        # from submission bundle (if present) or any notes you already collect
        if submission_bundle is not None:
            sb = (
                submission_bundle.to_dict()
                if hasattr(submission_bundle, "to_dict")
                else submission_bundle
            )
            notes_text = (sb or {}).get("notes")
    except Exception:
        notes_text = None

    sov_snips = pack_sov_snippets(sov_rows, top_k=10)
    loss_snips = pack_loss_snippets(loss_rows, top_k=10)
    notes_snips = pack_notes_snippets(notes_text, top_k=6)
    llm_ctx = build_llm_context(sov_snips, loss_snips, notes_snips)

    # --- Risk items (baseline rules + short LLM justifications) ---
    # Build minimal bundle shape the rules expect
    bundle_for_rules = {
        "sov": [{"sheets": [{"rows": sov_rows}]}],
        "loss_run": [{"sheets": [{"rows": loss_rows}]}],
    }

    risk_items: List[Any] = []
    try:
        # 1) Deterministic rules
        base_items = rules_from_bundle(bundle_for_rules)
        print(f"[RISK] base rules items: {len(base_items)}")

        # 2) JSON red-flag rules (also deterministic)
        try:
            loc_to_addr = {
                str(r.get("location_id")): r.get("address")
                for r in sov_rows
                if r.get("location_id")
            }
            rf_rules = _load_rules()
            rf_items = evaluate_red_flags(rf_rules, sov_rows, loss_rows, loc_to_addr)
            print(f"[RISK] red-flag items: {len(rf_items)}")
        except Exception as e:
            print("[RISK] red-flag evaluation failed:", repr(e))
            rf_items = []

        risk_items = base_items + rf_items

    except Exception as e:
        # If even deterministic rules failed, surface it but don't crash the whole run
        print("[RISK] deterministic stage failed:", repr(e))
        risk_items = []

    # 3) Concise LLM justification (optional, fail-soft)
    try:
        before = len(risk_items)
        risk_items = justify_with_llm(risk_items)  # adds short notes only
        print(f"[RISK] justify_with_llm ok (kept {before} items)")
    except Exception as e:
        print("[RISK] justify_with_llm failed, keeping deterministic items:", repr(e))

    # 4) De-dupe & cap (always safe)
    try:
        risk_items = dedupe_and_cap(risk_items)
        print(f"[RISK] after dedupe: {len(risk_items)}")
    except Exception as e:
        print("[RISK] dedupe failed, leaving list as-is:", repr(e))

    # 5) Make it JSON-serializable for the UI
    risk_items_payload = [
        (
            ri.model_dump()
            if hasattr(ri, "model_dump")
            else (ri.dict() if hasattr(ri, "dict") else dict(ri))
        )
        for ri in (risk_items or [])
    ]

    # 6) Optional: LLM risk miner (behind UI/flag). Failure here must not nuke items.
    try:
        cfg = get_config()
        use_llm = st.session_state.get("use_llm_miner", cfg.get("use_llm_miner", False))
    except Exception:
        use_llm = False

    if use_llm:
        try:
            print("[LLM-MINER] enabled")
            existing_codes = {(ri.get("code") or "") for ri in risk_items_payload}
            existing_titles = {(ri.get("title") or "") for ri in risk_items_payload}
            llm_ctx = build_llm_context(
                sov_snips, loss_snips, notes_snips
            )  # already built above in your file
            mined = mine_additional_risks(
                llm_ctx, existing_codes, existing_titles, model=cfg["llm_miner_model"]
            )
            print(f"[LLM-MINER] parsed items: {len(mined)}")
            # Safety net: ensure uid exists on every mined item
            for m in mined:
                if not m.get("uid"):
                    m["uid"] = _uid_from_item(m)

            # Merge but do NOT drop base items; keep miner tags visible
            risk_items_payload.extend(mined)
            # prefer item with a uid; merge tags and evidence; dedupe by (code,title)
            by_key = {}
            for x in risk_items_payload:
                code = x.get("code") or ""
                title = x.get("title") or ""
                key = (code, title)
                prev = by_key.get(key)

                if not prev:
                    by_key[key] = x
                    continue

                # Choose a winner: prefer the one that has a uid
                this_has_uid = bool(x.get("uid"))
                prev_has_uid = bool(prev.get("uid"))

                if this_has_uid and not prev_has_uid:
                    winner, other = x, prev
                else:
                    winner, other = prev, x

                # Merge tags (unique, order-preserving)
                tags = list(
                    dict.fromkeys(
                        (winner.get("tags") or []) + (other.get("tags") or [])
                    )
                )
                if tags:
                    winner["tags"] = tags

                # Merge evidence (unique by (source, locator); soft-cap to 4)
                ev = (winner.get("evidence") or []) + (other.get("evidence") or [])
                ev_seen, ev_uniq = set(), []
                for e in ev:
                    t = (e.get("source"), e.get("locator"))
                    if t in ev_seen:
                        continue
                    ev_seen.add(t)
                    ev_uniq.append(e)
                winner["evidence"] = ev_uniq[:4]

                # Keep any llm_notes already present on the winner; if only the other has it, copy over
                if not winner.get("llm_notes") and other.get("llm_notes"):
                    winner["llm_notes"] = other["llm_notes"]

                by_key[key] = winner

            risk_items_payload = list(by_key.values())
            print(
                f"[LLM-MINER] after merge/dedupe (uid-preferred): {len(risk_items_payload)}"
            )

        except Exception as e:
            print("[LLM-MINER] failed, continuing with deterministic items:", repr(e))

    # 7) Attach top-k evidence (never let this kill the list)
    try:
        risk_items_payload = attach_topk_evidence_to_items(
            risk_items_payload, sov_rows, loss_rows, notes_text=notes_text, per_item_k=2
        )
    except Exception as e:
        print("[RISK] attach_topk_evidence_to_items failed:", repr(e))

    # Safety net: restore mined uids if any step dropped them inadvertently
    try:
        for it in risk_items_payload:
            if (it.get("tags") and "llm-mined" in it["tags"]) and not it.get("uid"):
                it["uid"] = _uid_from_item(it)  # imported from core.risk.miner
    except Exception as e:
        print("[RISK] uid safety-net failed:", repr(e))

    # ---- Emit events with finalized risk items (includes UIDs) ----
    try:
        # Use the run_id already in session (app sets this when you click Process)
        run_id_for_event = st.session_state.get("run_id")

        # Aggregate event with all UIDs (handy for agents/subscribers)
        publish(
            "RiskItemsReady",
            {
                "run_id": run_id_for_event,
                "count": len(risk_items_payload),
                "uids": [ri.get("uid") for ri in risk_items_payload if ri.get("uid")],
                "source": "pipeline",
            },
        )

        # (Optional) Per-item creation events (more verbose; great for debugging/streaming)
        for ri in risk_items_payload:
            try:
                publish(
                    "RiskItemCreated",
                    {
                        "run_id": run_id_for_event,
                        "uid": ri.get("uid"),
                        "code": ri.get("code"),
                        "title": ri.get("title"),
                        "severity": ri.get("severity"),
                        "anchors": [
                            (e.get("source"), e.get("locator"))
                            for e in (ri.get("evidence") or [])
                        ],
                        "tags": ri.get("tags") or [],
                        "source": "pipeline",
                    },
                )
            except Exception as inner_e:
                print("[events] RiskItemCreated emit failed:", repr(inner_e))

    except Exception as e:
        print("[events] RiskItemsReady emit failed:", repr(e))

    # ---------------- Filters ----------------
    def _filter_profile_not_in_active(
        profile: Dict[str, Any], dt: str
    ) -> Dict[str, Any]:
        """Return profile limited to not-in-ACTIVE keys (compare on clean base).
        Handles both shapes:
          A) {"properties": { field: stats }}
          B) { field: stats }
        """
        if not isinstance(profile, dict):
            return {}
        act_clean = {_strip_provisional(a) for a in _active_set(dt)}

        props = profile.get("properties")
        if isinstance(props, dict):
            filtered = {
                k: v for k, v in props.items() if _strip_provisional(k) not in act_clean
            }
            out = dict(profile)
            out["properties"] = filtered
            return out

        filtered = {
            k: v for k, v in profile.items() if _strip_provisional(k) not in act_clean
        }
        return filtered

    def _filter_suggestions_not_in_active(suggestions: Any, dt: str) -> Any:
        """Coerce suggestions and drop items present in ACTIVE (compare on clean base)."""
        act_clean = {_strip_provisional(a) for a in _active_set(dt)}

        def _keep(name: str) -> bool:
            return _strip_provisional(name) not in act_clean

        if not suggestions:
            return {"suggestions": []}
        if isinstance(suggestions, list):
            return {"suggestions": [s for s in suggestions if _keep(s)]}
        if isinstance(suggestions, dict):
            if "suggestions" in suggestions and isinstance(
                suggestions["suggestions"], list
            ):
                return {
                    "suggestions": [s for s in suggestions["suggestions"] if _keep(s)]
                }
            if "properties" in suggestions and isinstance(
                suggestions["properties"], dict
            ):
                return {
                    "properties": {
                        k: v for k, v in suggestions["properties"].items() if _keep(k)
                    }
                }
            return {"suggestions": [k for k in suggestions.keys() if _keep(k)]}
        return {"suggestions": []}

    def _reduce_and_filter_proposals(
        doc_type: str, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Keep a single 'latest' proposal and filter its contents so that:
          - new_fields: only fields whose CLEAN base is NOT in ACTIVE
          - mappings:   KEEP when target is a valid ACTIVE key;
                        DROP only if the source header is already present in crosswalk (duplicate).
        """
        if not items:
            return []

        latest = items[-1] or {}

        act = {_strip_provisional(k) for k in (active_keys(doc_type) or [])}
        cw = _read_crosswalk() or {}  # already normalized keys
        cw_keys = set(cw.keys())  # compare with normalized source header

        # ---- filter new_fields (not already active)
        nfs = []
        for nf in latest.get("new_fields") or []:
            if isinstance(nf, dict):
                name = _strip_provisional(str(nf.get("field_name", "")).strip())
            else:
                name = _strip_provisional(str(nf).strip())
            if name and name not in act:
                nfs.append(nf)
        latest["new_fields"] = nfs

        # ---- filter mappings
        mps = []
        seen_src = set()
        for mp in latest.get("mappings") or []:
            src_raw = str(mp.get("source_header", "")).strip()
            tgt_raw = str(mp.get("target_field", "")).strip()
            if not src_raw or not tgt_raw:
                continue

            src_norm = _norm_join(src_raw)
            tgt_clean = _strip_provisional(tgt_raw)

            # target must be a valid ACTIVE key for this doc_type
            if tgt_clean not in act:
                continue

            # drop if this source header is already mapped in crosswalk (duplicate)
            if src_norm in cw_keys:
                continue

            # de-dupe within a single proposal payload
            if src_norm in seen_src:
                continue
            seen_src.add(src_norm)

            # keep (this will surface in the Schema Review tab)
            mps.append(
                {
                    "source_header": src_raw,
                    "target_field": tgt_clean,
                    "confidence": float(mp.get("confidence", 0) or 0),
                    "rationale": mp.get("rationale", ""),
                }
            )

        latest["mappings"] = mps
        return [latest]

    # ========== Dynamic Schema Profiling + Suggestions (filtered by ACTIVE) ==========
    schema_profile: Dict[str, Dict[str, Any]] = {}
    schema_profile_full: Dict[str, Dict[str, Any]] = {}
    schema_suggestions: Dict[str, Dict[str, Any]] = {}

    if sov_rows:
        prof_sov = profile_rows(sov_rows)  # full profile
        schema_profile_full["sov"] = prof_sov
        schema_profile["sov"] = _filter_profile_not_in_active(prof_sov, "sov")
        sov_active_schema = load_active_schema("sov")
        if isinstance(sov_active_schema, (list, tuple)):
            sov_active_schema = sov_active_schema[0]
        sug_sov = profile_to_suggestions(prof_sov, sov_active_schema)
        schema_suggestions["sov"] = _filter_suggestions_not_in_active(sug_sov, "sov")

    if loss_rows:
        prof_loss = profile_rows(loss_rows)
        schema_profile_full["loss_run"] = prof_loss
        schema_profile["loss_run"] = _filter_profile_not_in_active(
            prof_loss, "loss_run"
        )
        loss_active_schema = load_active_schema("loss_run")
        if isinstance(loss_active_schema, (list, tuple)):
            loss_active_schema = loss_active_schema[0]
        sug_loss = profile_to_suggestions(prof_loss, loss_active_schema)
        schema_suggestions["loss_run"] = _filter_suggestions_not_in_active(
            sug_loss, "loss_run"
        )
    # ======================================================================

    # --- Reduce & filter LLM proposals to one latest, trimmed by ACTIVE/crosswalk ---
    sov_proposals = _reduce_and_filter_proposals("sov", sov_proposals)
    loss_proposals = _reduce_and_filter_proposals("loss_run", loss_proposals)

    # --- Email envelopes ---
    email_envelopes: List[Dict[str, Any]] = []
    for em in bundle.get("email_body", []):
        hdrs = em.get("headers", {}) or {}
        email_envelopes.append(
            {
                "from": hdrs.get("from", ""),
                "to": hdrs.get("to", []),
                "cc": hdrs.get("cc", []),
                "sent_date": hdrs.get("date", ""),
                "subject": hdrs.get("subject", ""),
                "body_text": (em.get("body_text") or "")[:8000],
                "attachments": [
                    {
                        "filename": a.get("filename"),
                        "content_type": a.get("content_type"),
                    }
                    for a in em.get("attachments", [])
                ],
            }
        )

    # Merge envelopes derived from SubmissionBundle.attachments (role='email')
    if email_envs_from_sb:
        email_envelopes.extend(email_envs_from_sb)

    # --- Questionnaire context ---
    questionnaire_context: List[Dict[str, Any]] = []
    for q in bundle.get("questionnaire", []):
        text = (q.get("text") or "")[:8000]
        if text:
            questionnaire_context.append(
                {"source": q.get("filename") or "questionnaire", "excerpt": text}
            )

    # --- Validate normalized rows against ACTIVE schema ---
    sov_validation = validate_rows("sov", sov_rows)
    loss_run_validation = validate_rows("loss_run", loss_rows)

    # ===================== Pricing Enrichment (inserted here) =====================
    try:
        # 1) Derive submission-level metrics from normalized data
        total_tiv = _sum_tiv(sov_rows)
        # Prefer COPE from the Risk pipeline (normalized to 0–100); fallback to heuristic if unavailable
        try:
            _risk_bundle_min = {
                "sov": {"records": sov_rows or []},
                "loss_run": {"records": loss_rows or []},
            }
            risk_cope_obj, _ = run_risk_pipeline(_risk_bundle_min)
            cope_score = _cope_index_from_breakdown(risk_cope_obj)
            _cope_source = "risk_pipeline"
        except Exception as _e:
            print("[PRICING] Risk COPE unavailable, using heuristic COPE:", repr(_e))
            cope_score = _approx_cope_score(sov_rows)
            _cope_source = "heuristic"
        risk_count_total = len(risk_items_payload or [])
        loss_ratio = _loss_ratio_proxy(loss_rows, total_tiv)

        # 2) Build pricing input
        submission_id = st.session_state.get("run_id") or "unknown"
        p_in = SubmissionPricingInput(
            submission_id=submission_id,
            tiv=total_tiv,
            cope_score=cope_score,
            risk_count=risk_count_total,
            loss_ratio=loss_ratio,
            # optionally pass occupancy/construction/protection/exposure if you have them
        )

        # 3) Benchmark + strategy-picked pricing
        bm = PriceBenchmark(csv_path="data/benchmarks/pricing_benchmark.csv")

        # Pick strategy (Streamlit control can set st.session_state['pricing_strategy'])
        strategy = (
            st.session_state.get("pricing_strategy")
            or os.getenv("PRICING_STRATEGY", "rules")
        )

        try:
            pricer = get_pricer(strategy)
        except Exception as _e:
            print(f"[PRICING] unknown strategy '{strategy}', defaulting to 'rules'")
            pricer = get_pricer("rules")

        p_res = pricer.price(p_in, bm)


        # 4) Currency normalization (USD). You asked to keep the LLM pricer in USD as well.
        p_res.pricing_range = PricingRange(
            premium_min=p_res.pricing_range.premium_min,
            premium_median=p_res.pricing_range.premium_median,
            premium_max=p_res.pricing_range.premium_max,
            currency="USD",
        )

        # 5) Optional LLM refinement (kept off by default; enable via config/flag)
        # try:
        #     cfg = get_config()
        #     if cfg.get("use_llm_pricing_blend", False):
        #         llm_client = cfg.get("llm_client")  # your wrapper
        #         if llm_client:
        #             from pricing.llm_pricer import LLMPricerBlend
        #             p_res = LLMPricerBlend(llm_client, weight_llm=0.35).blend(p_in, bm, p_res)
        # except Exception as e:
        #     print("[PRICING] LLM blend skipped:", repr(e))

        pricing_payload = _pricing_result_to_dict(p_res)
        pricing_payload["strategy"] = strategy
        pricing_payload["cope_source"] = _cope_source 
    except Exception as e:
        print("[PRICING] enrichment failed:", repr(e))
        pricing_payload = None
    # ==============================================================================


    
    # --- Assemble result object expected by the UI ---
    return {
        "submission_core": submission_core,
        "sov": sov_rows,
        "loss_runs": loss_rows,
        "sov_source_rows": sov_source_rows,  # <— for preview
        "loss_run_source_rows": loss_source_rows,  # <— for preview
        "proposals": {
            "sov": sov_proposals,  # reduced + filtered
            "loss_run": loss_proposals,  # reduced + filtered
        },
        "email_envelopes": email_envelopes,
        "questionnaire_context": questionnaire_context,
        "sov_validation": sov_validation,
        "loss_run_validation": loss_run_validation,
        "schema_profile_full": schema_profile_full,  # all fields (diagnostics)
        "schema_profile": schema_profile,  # only not-in-ACTIVE (UI)
        "schema_suggestions": schema_suggestions,  # only not-in-ACTIVE (UI)
        "risk_items": risk_items_payload,
        "evidence_snippets": {
            "sov": sov_snips,
            "loss": loss_snips,
            "notes": notes_snips,
        },
        "llm_context": build_llm_context(sov_snips, loss_snips, notes_snips),
        "pricing": pricing_payload,   # <— NEW: Risk & Pricing tab will read this
        "debug_bundle_counts": {k: len(v) for k, v in bundle.items()},
    }
