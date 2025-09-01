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
        import re

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
        "debug_bundle_counts": {k: len(v) for k, v in bundle.items()},
    }
