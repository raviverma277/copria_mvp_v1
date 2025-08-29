import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import json, re

from core.config import get_config
from core.utils.state import get_state
from core.parsing.dispatch import parse_files
from core.extraction.pipeline import run_extraction_pipeline
from core.risk.pipeline import run_risk_pipeline
from core.pricing.engine import price_submission
from core.schemas.active import get_active_name, active_keys
from core.schemas.active import active_titles
from core.extraction.field_mapping import _strip_provisional, PROV_PREFIX
from core.schemas.schema_builder import (
    build_property,
    preview_vnext_schema,
    generate_vnext_schema,
)

import pandas as pd 


# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# ---- Ingestion router & contract (ADD) ----
import uuid, os
from core.extraction.ingest_router import select_source, from_local, from_cytora, get_bundle_auto
from core.schemas.contracts import IngestSource
import uuid


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}

def _unmapped_headers(bundle: dict, bucket: str) -> list[str]:
    """
    Look at the raw classified bundle and return headers that are not
    in core/schemas/crosswalk.json for the given bucket ('sov' or 'loss_run').
    """
    cw_path = Path("core/schemas/crosswalk.json")
    cw = json.loads(cw_path.read_text()) if cw_path.exists() else {}
    nm = set()

    for excel in (bundle or {}).get(bucket, []):
        for sh in excel.get("sheets", []) or []:
            for h in (sh.get("headers") or []):
                key = re.sub(r"[^a-z0-9]+", " ", str(h).lower()).strip()
                if key not in cw:
                    nm.add(str(h))
    return sorted(nm)

def _unmapped_questionnaire_keys(bundle: dict) -> list[str]:
    """
    Scan questionnaire text for 'Key: Value' or 'Key - Value' patterns and report
    keys not present in core/schemas/questionnaire_crosswalk.json.
    """
    qx_path = Path("core/schemas/questionnaire_crosswalk.json")
    qx = _load_json(qx_path)
    seen = set()
    pat = re.compile(r"^\s*([A-Za-z][\w /&%\-]{2,60})\s*[:\-–]\s*\S+", re.MULTILINE)
    for q in (bundle or {}).get("questionnaire", []):
        text = q.get("text", "") or ""
        for m in pat.finditer(text):
            key = m.group(1)
            if _norm(key) not in qx:
                seen.add(key.strip())
    return sorted(seen)

def _unmapped_email_fields(results: dict) -> list[str]:
    """
    Compare parsed email envelope keys against email_envelope.schema.json properties.
    Returns extra/missing keys as 'extra:<key>' or 'missing:<key>'.
    """
    schema_path = Path("core/schemas/json/email_envelope.schema.json")
    schema = _load_json(schema_path)
    allowed = set((schema.get("properties") or {}).keys())
    issues = set()

    for env in (results or {}).get("email_envelopes", []):
        present = set([k for k, v in env.items() if v not in (None, "", [], {})])
        # flag extras that aren't in the schema (ignoring 'attachments' which we add locally)
        for k in present - allowed - {"attachments"}:
            issues.add(f"extra:{k}")
        # flag important missing fields
        for req in ["from", "to", "subject", "sent_date"]:
            if req in allowed and (env.get(req) in (None, "", [], {})):
                issues.add(f"missing:{req}")

    return sorted(issues)

def _fmt_addr_list(v):
    if isinstance(v, list):
        return ", ".join(v)
    if isinstance(v, str):
        return v
    return ""

def combine_label(key: str, title_map: dict) -> str:
    title = title_map.get(key)
    return f"{title}\n({key})" if title and title != key else key

def _apply_proposals_preview(source_rows, normalized_rows, proposal_obj):
    """
    Merge proposed mappings into the current normalized rows JUST for display.
    Does not save to crosswalk.json.
    """
    if not proposal_obj:
        return normalized_rows

    mappings = (proposal_obj or {}).get("mappings", [])
    if not mappings:
        return normalized_rows

    map_src_to_tgt = {m["source_header"]: m["target_field"] for m in mappings}

    out = []
    # We zip by index; if lengths differ we fall back to reading keys by presence.
    for idx in range(max(len(source_rows), len(normalized_rows))):
        raw = source_rows[idx] if idx < len(source_rows) else {}
        norm = dict(normalized_rows[idx]) if idx < len(normalized_rows) else {}

        for src, tgt in map_src_to_tgt.items():
            if tgt not in norm and src in raw:
                norm[tgt] = raw[src]
        out.append(norm)
    return out

# ---------- ACTIVE schema helpers ----------
def _active_keys_map() -> dict[str, set[str]]:
    return {
        "SOV": set(active_keys("sov")),
        "LOSS_RUN": set(active_keys("loss_run")),
    }

def _clean_list(values):    
    out = []
    seen = set()
    for v in values or []:
        c = _strip_provisional(str(v).strip())
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _filter_new_fields(doc_kind: str, candidates: list[str]) -> list[str]:
    ak = _active_keys_map()
    base = _clean_list(candidates)
    return [c for c in base if c not in ak.get(doc_kind, set())]


st.set_page_config(page_title="CoPRIA", layout="wide")
st.title("CoPRIA – Commercial Property Risk Intelligence Agent")

def _bridge_streamlit_secrets_to_env(keys=("OPENAI_API_KEY", "USE_LLM_MINER", "LLM_MINER_MODEL")):
    try:
        # Accessing st.secrets can raise FileNotFoundError if no secrets.toml exists.
        secrets = st.secrets
        for key in keys:
            val = secrets.get(key, None)
            if val is not None and val != "":
                os.environ[key] = str(val)
    except FileNotFoundError:
        # No secrets.toml in local dev – that's fine; .env (dotenv) will cover it.
        pass
    except Exception as e:
        # Don't block the app; just log for visibility.
        print("[secrets->env] Skipped bridging Streamlit secrets:", repr(e))

_bridge_streamlit_secrets_to_env()

# ---- Safe wrapper so missing secrets.toml never shows in the UI ----
def _safe_llm_status():
    try:
        # lazy import so st.secrets is only touched here
        from core.utils import llm_status as _llm
        return _llm.get_llm_status()
    except FileNotFoundError:
        return {
            "api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "last_call": None,
            "last_duration": None,
            "last_success": None,
        }
    except Exception as e:
        print("[llm_status] suppressed:", repr(e))
        return {
            "api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "last_call": None,
            "last_duration": None,
            "last_success": None,
        }




state = get_state()




# ---------------- Sidebar ---------------- #

with st.sidebar:
    st.markdown("### Active Schemas")
    st.write({
        "sov": get_active_name("sov"),
        "loss_run": get_active_name("loss_run"),
        "questionnaire": get_active_name("questionnaire"),
    })
    st.caption(f"SOV keys: {len(active_keys('sov'))} • Loss keys: {len(active_keys('loss_run'))}")

st.markdown("---")  # divider

# ---- Unified Controls (center) ----
from core.config import get_config
from core.schemas.contracts import IngestSource
import os

_cfg = get_config()
src = select_source()

with st.container(border=True):
    st.markdown("### Controls")

    col1, col2, col3 = st.columns([0.45, 0.35, 0.20])

    # Ingest Source selector
    with col1:
        st.markdown("**Ingest Source**")
        ui_src = st.selectbox(
            "Override ingest source",
            options=[IngestSource.LOCAL.value, IngestSource.CYTORA.value, IngestSource.AUTO.value],
            index=[IngestSource.LOCAL, IngestSource.CYTORA, IngestSource.AUTO].index(src),
            label_visibility="collapsed",
            help="Default comes from ENV (INGEST_SOURCE). This overrides it for this session."
        )
        if ui_src != src.value:
            os.environ["INGEST_SOURCE"] = ui_src
            src = select_source()
        st.caption(f"Using: `{src.value}`")

    # LLM Risk Miner toggle (moved from sidebar)
    with col2:
        st.markdown("**Risk Miner**")
        use_llm_ui = st.checkbox(
            "Use LLM Risk Miner",
            value=_cfg.get("use_llm_miner", False),
            help="Enable to let the LLM propose additional risks using the evidence snippets"
        )
        # session flag the pipeline reads
        st.session_state["use_llm_miner"] = use_llm_ui
        st.caption(f"LLM Miner model: {_cfg.get('llm_miner_model')}")

    # (Optional) API key/quick status pill
    with col3:
        info = _safe_llm_status()
        ok = info.get("api_key_set", False)
        st.markdown("**LLM Key**")
        st.markdown(
            f"<div style='padding:8px 10px;border-radius:999px;"
            f"background:{'#16a34a' if ok else '#ef4444'};color:white;width:max-content;'>"
            f"{'Set' if ok else 'Missing'}</div>",
            unsafe_allow_html=True
        )


# load defaults from config.yaml/env
_cfg = get_config()

with st.sidebar:
    st.header("Upload")
    files = st.file_uploader(
        "Upload submission pack (PDF/XLSX/EML/MSG)",
        accept_multiple_files=True,
        type=["pdf", "xlsx", "xls", "docx", "eml", "msg"],
    )

    colA, colB = st.columns(2)
    with colA:
        classify_btn = st.button("Classify")
    with colB:
        process_btn = st.button("Process")

    if classify_btn:
        if files:
            with st.spinner("Classifying..."):
                state["raw_bundle"] = parse_files(files)
            st.success("Classification complete.")
        else:
            st.warning("Please upload at least one file.")

    if process_btn:
        # --- EARLY CYTORA FAST-PATH: bypass local preconditions/warnings ---
        source = select_source()
        if source == IngestSource.CYTORA:
            # Clear any stale LLM state (keep your existing clears)
            for k in ("last_llm_call", "llm_full_draft", "llm_full_make_active_persist"):
                state.pop(k, None)
                st.session_state.pop(k, None)

            run_id = str(uuid.uuid4())
            submission_bundle = from_cytora(run_id=run_id)  # reads sample JSON (or your configured path)
            st.sidebar.caption(f"Run: `{run_id[:8]}`")

            with st.spinner("Extracting from Cytora sample…"):
                state["results"] = run_extraction_pipeline(
                    parsed_bundle=None,
                    submission_bundle=submission_bundle
                )
            st.success("Extraction complete.")

            # Persist results for the next render and re-run the app to draw tabs
            st.session_state["results"] = state.get("results")
            try:
                st.rerun()               # Streamlit ≥ 1.32
            except Exception:
                st.experimental_rerun()  # fallback for older versions


        # --- LOCAL / AUTO path (your existing behavior) ---
        # Reuse classification if present; else parse fresh once
        bundle = state.get("raw_bundle")
        if bundle is None and files:
            with st.spinner("Classifying..."):
                bundle = parse_files(files)
                state["raw_bundle"] = bundle

        if bundle is None:
            # Only warn on LOCAL/AUTO when no inputs are present
            st.warning("Please upload files or click Classify first.")
        else:
            # --- CLEAR any stale "Full Schema LLM" info before re-running pipeline ---
            state.pop("last_llm_call", None)
            st.session_state.pop("last_llm_call", None)

            # NEW: also clear the persisted LLM full-draft so the 'Resume last LLM draft' panel disappears
            state.pop("llm_full_draft", None)
            st.session_state.pop("llm_full_draft", None)  # defensive, in case you ever mirror it

            # (Optional) clear the 'Make ACTIVE' checkbox state for the draft panel
            st.session_state.pop("llm_full_make_active_persist", None)

            # ---- BEGIN: NEW ingestion wiring (kept as you added) ----
            run_id = str(uuid.uuid4())
            source = select_source()

            # Prepare args that describe your current local-parsed state
            # (your parse_files(...) already produced "bundle" in raw/classified shape)
            local_args = {
                # Pass through the normalized tables if your pipeline already puts them here later;
                # for step-1, just leave these None and rely on parsers already used by pipeline.
                "parsed_sov": None,
                "parsed_loss": None,
                # Put any freeform text you already have (optional)
                "notes": None,
                # Persisted file metadata list if you have it (optional now); you can leave []
                "attachments": [],
            }

            # Decide which ingestion path to use
            if source == IngestSource.LOCAL:
                submission_bundle = from_local(**local_args, run_id=run_id)
            elif source == IngestSource.CYTORA:
                # If the user switched to Cytora after uploads/classify, allow Cytora too.
                submission_bundle = from_cytora(run_id=run_id)
            else:
                submission_bundle = get_bundle_auto(local_args, run_id=run_id)

            # Make the run metadata visible
            st.sidebar.caption(f"Run: `{run_id[:8]}`")

            # ---- END: NEW ingestion wiring ----

            with st.spinner("Extracting..."):
                # Keep your legacy parsed flow, and also pass the new contract (optional in pipeline)
                state["results"] = run_extraction_pipeline(
                    parsed_bundle=bundle,
                    submission_bundle=submission_bundle
                )
            st.success("Extraction complete.")



    
    st.sidebar.markdown("### 🔍 LLM Status")
    llm_info = _safe_llm_status()

    if llm_info["api_key_set"]:
        st.sidebar.success("API key set")
    else:
        st.sidebar.error("No API key")

    st.sidebar.write(f"**Model:** {llm_info['model']}")
    st.sidebar.write(f"**Last Call:** {llm_info['last_call'] or '—'}")
    st.sidebar.write(f"**Duration:** {llm_info['last_duration'] or '—'}s")
    if llm_info["last_success"] is True:
        st.sidebar.success("Last call succeeded")
    elif llm_info["last_success"] is False:
        st.sidebar.error("Last call failed")
    else:
        st.sidebar.info("No calls yet")
    
    # --- Full-schema (Section F) LLM diagnostics: action, time, duration, model, tokens ---
    try:
        from core.schemas.schema_builder import get_llm_last_meta, get_llm_last_error
    except Exception:
        def get_llm_last_meta(): return None
        def get_llm_last_error(): return None

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Full-schema draft (LLM)**")

    # Persisted payload from the Section F button handler
    last_full = state.get("last_llm_call")  # set in Section F when you call propose_full_schema_from_llm
    meta = get_llm_last_meta() or {}
    err  = get_llm_last_error()

    if not last_full:
        st.sidebar.caption("No full-schema LLM calls yet.")
    else:
        # Basic call facts
        st.sidebar.write(f"**Action:** {last_full.get('action','—')}")
        st.sidebar.write(f"**Bucket:** {last_full.get('bucket','—')}")
        st.sidebar.write(f"**When:** {last_full.get('ts','—')}")

        # Duration formatting (prefer ms if present; fall back to seconds if you ever store it)
        dur_ms = last_full.get("duration_ms")
        if isinstance(dur_ms, (int, float)):
            st.sidebar.write(f"**Duration:** {int(dur_ms)} ms")
        else:
            st.sidebar.write(f"**Duration:** {last_full.get('last_duration','—')}s")

        # Model & token usage (if the provider returned it)
        model = last_full.get("model") or meta.get("model")
        if model:
            st.sidebar.write(f"**Model:** {model}")

        usage = last_full.get("usage") or meta.get("usage") or {}
        if isinstance(usage, dict) and any(k in usage for k in ("prompt_tokens","completion_tokens","total_tokens")):
            st.sidebar.write("**Tokens:**")
            st.sidebar.json({
                "prompt": usage.get("prompt_tokens"),
                "completion": usage.get("completion_tokens"),
                "total": usage.get("total_tokens"),
            }, expanded=False)

        # Success/error indicator
        if err:
            st.sidebar.error(f"Error: {err}")
        else:
            st.sidebar.success("Last full-schema call succeeded")


# --------------- Tabs ---------------- #
tabs = st.tabs([
    "Classification",     # 0
    "Submission",         # 1
    "SOV",                # 2
    "Loss Runs",          # 3
    "Email",              # 4   <-- new
    "Questionnaire",      # 5   <-- new
    "Risk & Pricing",     # 6
    "Schema Review",      # 7
    "JSON"                # 8
])


# --- Tab 0: Classification (routing debug) --- #
with tabs[0]:
    st.subheader("Document Classification")
    bundle = state.get("raw_bundle")
    if not bundle:
        st.info("Upload files and click **Classify** (or just click **Process** to classify+extract in one go).")
    else:
        # colored pills
        def pill(label: str, count: int):
            bg = "#16a34a" if count > 0 else "#9ca3af"
            st.markdown(
                f"""
                <div style="
                    display:flex;align-items:center;justify-content:center;
                    background:{bg};color:white;
                    padding:10px 12px;border-radius:999px;
                    font-weight:600;min-width:140px;">
                    {label}: {count}
                </div>
                """,
                unsafe_allow_html=True,
            )

        buckets = ["submission", "sov", "loss_run", "questionnaire", "email_body", "other"]
        cols = st.columns(len(buckets))
        for i, b in enumerate(buckets):
            with cols[i]:
                pill(b.capitalize(), len(bundle.get(b, [])))

        st.write("")
        if st.checkbox("Show raw classification JSON"):
            st.json(bundle)

        # per-bucket details
        for b in buckets:
            with st.expander(f"{b.upper()} – {len(bundle.get(b, []))} item(s)", expanded=False):
                for idx, item in enumerate(bundle.get(b, []), 1):
                    name = item.get("filename", "(no name)")
                    meta = item.get("meta", {})
                    st.markdown(f"**[{idx}] {name}**")
                    if meta:
                        st.caption(f"meta: {meta}")

                    page_tags = item.get("page_tags", [])
                    if page_tags:
                        tag_str = ", ".join(
                            [f"p{t.get('page')}:{t.get('type')}({float(t.get('confidence',0)):.2f})" for t in page_tags[:12]]
                        )
                        if len(page_tags) > 12:
                            tag_str += " ..."
                        st.write("page_tags:", tag_str)

                    if "text" in item and item["text"]:
                        st.text_area("text preview", item["text"][:1500], height=120, key=f"txtprev_{b}_{idx}")
                    elif "sheets" in item:
                        for sh in item["sheets"][:2]:
                            st.write(f"Sheet: {sh.get('name')}")
                            headers = sh.get("headers") or []
                            st.write("Headers:", ", ".join(str(h) for h in headers))
                    elif "body_text" in item:
                        st.text_area("email preview", item["body_text"][:1500], height=120, key=f"emailprev_{idx}")

# If we have extraction results, render the other tabs
results = state.get("results")

# --- Tab 1: Submission --- #
with tabs[1]:
    st.subheader("Submission Summary (Lloyd's Fields)")
    if results:
        sub = results.get("submission_core", {})
        st.json(sub)        
    else:
        st.info("Click **Process** to extract submission fields.")

# --- Tab 2: SOV --- #
with tabs[2]:
    st.subheader("Statement of Values (SOV)")

    results = state.get("results") or {}
    sov_rows = results.get("sov") or []
    sov_source = (results or {}).get("sov_source_rows", [])
    sov_props = ((results or {}).get("proposals", {}) or {}).get("sov", [])
    latest_sov_prop = sov_props[-1] if sov_props else None

    # Allow a quick preview using the latest proposal (not persisted)
    preview = st.checkbox("Preview with proposed mappings (not saved)", value=False, key="sov_preview")
    rows_to_show = sov_rows
    if preview and sov_source and latest_sov_prop:
        # This helper is your existing function that overlays proposal mappings on the fly
        rows_to_show = _apply_proposals_preview(sov_source, sov_rows, latest_sov_prop)

    if not rows_to_show:
        st.info("No SOV rows found yet. Click **Process**.")
    else:
        import pandas as pd
        df_sov = pd.DataFrame(rows_to_show)

        # ---- FIX: remove duplicate column labels to avoid PyArrow/Streamlit crash ----
        df_sov = df_sov.loc[:, ~df_sov.columns.duplicated()]

        # Show only ACTIVE schema fields by default
        schema_keys = set(active_keys("sov"))
        schema_cols = [c for c in df_sov.columns if c in schema_keys]

        # Provisional/unknown columns we carried through from normalization (prefixed)
        provisional_cols = [
            c for c in df_sov.columns
            if str(c).startswith("_provisional_") and c not in schema_cols
        ]
        hidden_count = len(provisional_cols)

        # Toggle to reveal provisional/unknown columns (for debugging/profiling only)
        show_prov = st.checkbox(
            f"Show provisional/unknown fields ({hidden_count} hidden)",
            value=False,
            key="sov_show_prov"
        )

        # Build the display list and ensure no duplicates make it through
        view_cols = schema_cols + (provisional_cols if show_prov else [])
        view_cols = list(dict.fromkeys(view_cols))  # order-preserving de-dup

        if not view_cols:
            st.info("No active-schema fields to display yet.")
        else:
            # Pretty labels for schema fields only (keep provisional columns clearly machine-named)
            sov_titles = active_titles("sov") or {}  # dict: key -> title
            rename_map = {k: combine_label(k, sov_titles) for k in schema_cols}

            dfv = df_sov[view_cols].copy()
            dfv.rename(columns=rename_map, inplace=True)

            st.dataframe(dfv, use_container_width=True)

            # Legend (schema key -> display title) for the columns we actually showed
            with st.expander("Column legend (schema key → display title)"):
                legend = {k: sov_titles.get(k, k) for k in sorted(schema_cols)}
                st.json(legend)

    # Schema validation results (if present)
    sov_issues = (results or {}).get("sov_validation", [])
    if sov_issues:
        st.warning(f"Schema validation: {len(sov_issues)} issue(s) found for SOV")
        with st.expander("View SOV schema validation details", expanded=False):
            st.json(sov_issues)
    else:
        if sov_rows:
            st.success("SOV rows conform to the active schema.")



# --- Tab 3: Loss Runs --- #
with tabs[3]:
    st.subheader("Loss Runs")

    results = state.get("results") or {}
    loss_rows = results.get("loss_runs") or []
    loss_source = (results or {}).get("loss_run_source_rows", [])
    loss_props = ((results or {}).get("proposals", {}) or {}).get("loss_run", [])
    latest_loss_prop = loss_props[-1] if loss_props else None

    # Allow a quick preview using the latest proposal (not persisted)
    preview_loss = st.checkbox("Preview with proposed mappings (not saved)", value=False, key="loss_preview")
    rows_to_show_loss = loss_rows
    if preview_loss and loss_source and latest_loss_prop:
        # Your existing helper to overlay proposal mappings on the fly
        rows_to_show_loss = _apply_proposals_preview(loss_source, loss_rows, latest_loss_prop)

    if not rows_to_show_loss:
        st.info("No Loss Run rows found yet. Click **Process**.")
    else:
        import pandas as pd
        df_loss = pd.DataFrame(rows_to_show_loss)

        # ---- FIX: remove duplicate column labels to avoid PyArrow/Streamlit crash ----
        df_loss = df_loss.loc[:, ~df_loss.columns.duplicated()]

        # Show only ACTIVE schema fields by default
        schema_keys = set(active_keys("loss_run"))
        schema_cols = [c for c in df_loss.columns if c in schema_keys]

        # Provisional/unknown columns we carried through from normalization (prefixed)
        provisional_cols = [
            c for c in df_loss.columns
            if str(c).startswith("_provisional_") and c not in schema_cols
        ]
        hidden_count = len(provisional_cols)

        # Toggle to reveal provisional/unknown columns (for debugging/profiling only)
        show_prov = st.checkbox(
            f"Show provisional/unknown fields ({hidden_count} hidden)",
            value=False,
            key="loss_show_prov"
        )

        # Build the display list and ensure no duplicates make it through
        view_cols = schema_cols + (provisional_cols if show_prov else [])
        view_cols = list(dict.fromkeys(view_cols))  # order-preserving de-dup

        if not view_cols:
            st.info("No active-schema fields to display yet.")
        else:
            # Pretty labels for schema fields only (keep provisional columns clearly machine-named)
            loss_titles = active_titles("loss_run") or {}  # dict: key -> title
            rename_map = {k: combine_label(k, loss_titles) for k in schema_cols}

            dfv = df_loss[view_cols].copy()
            dfv.rename(columns=rename_map, inplace=True)

            st.dataframe(dfv, use_container_width=True)

            # Legend (schema key -> display title) for the columns we actually showed
            with st.expander("Column legend (schema key → display title)"):
                legend = {k: loss_titles.get(k, k) for k in sorted(schema_cols)}
                st.json(legend)

    # Schema validation results (if present)
    loss_issues = (results or {}).get("loss_run_validation", [])
    if loss_issues:
        st.warning(f"Schema validation: {len(loss_issues)} issue(s) found for Loss Runs")
        with st.expander("View Loss Run schema validation details", expanded=False):
            st.json(loss_issues)
    else:
        if loss_rows:
            st.success("Loss Run rows conform to the active schema.")



# --- Tab 4: Email --- #
with tabs[4]:
    st.subheader("Email Envelopes")
    results = state.get("results")
    if results and results.get("email_envelopes"):
        for i, env in enumerate(results["email_envelopes"], 1):
            st.markdown(f"**[{i}] From:** {env.get('from','')}")
            st.write(f"**To:** {_fmt_addr_list(env.get('to'))}")
            if env.get("cc"):
                st.write(f"**Cc:** {_fmt_addr_list(env.get('cc'))}")
            st.write(f"**Subject:** {env.get('subject','')}")
            st.text_area("Body preview", env.get("body_text",""), height=120, key=f"email_body_{i}")
            if env.get("attachments"):
                st.caption("Attachments:")
                for att in env["attachments"]:
                    st.write(f"- {att.get('filename')} ({att.get('content_type')})")
        # Schema Discovery aide: show email envelope issues (extras/missing)
        email_issues = _unmapped_email_fields(state.get("results", {}))
        if email_issues:
            st.caption("Email envelope fields needing attention (vs schema):")
            st.write(email_issues)

    else:
        st.info("No email envelopes parsed yet. Upload an .eml/.msg and click Process.")

# --- Tab 5: Questionnaire --- #
with tabs[5]:
    st.subheader("Questionnaire (context)")
    results = state.get("results")
    if results and results.get("questionnaire_context"):
        for i, qc in enumerate(results["questionnaire_context"], 1):
            st.markdown(f"**[{i}] Source:** {qc.get('source','')}")
            st.text_area("Excerpt", qc.get("excerpt",""), height=140, key=f"qcx_{i}")
        # Schema Discovery aide: show unmapped questionnaire keys found in text
        q_unmapped = _unmapped_questionnaire_keys(state.get("raw_bundle", {}))
        if q_unmapped:
            st.caption("Unmapped questionnaire keys (consider adding to core/schemas/questionnaire_crosswalk.json):")
            st.write(q_unmapped[:60])  # cap the list for readability

    else:
        st.info("No questionnaire pages detected yet.")

# --- Tab 6: Risk & Pricing ---
with tabs[6]:
    st.subheader("Risk Items")
    risk_items = (results or {}).get("risk_items", [])

    if not results:
        st.info("Run **Process** to compute risk & pricing.")
    else:
        # ---- Render Risk Items (from pipeline) ----
        if not risk_items:
            st.caption("No risk items detected yet.")
        else:
            st.caption(f"{len(risk_items)} item(s) found")
            # Optional quick filter by severity (keeps backwards-compat if field missing)
            severities = sorted({(ri.get("severity") or "unknown") for ri in risk_items})
            sel = st.multiselect("Filter by severity", options=severities, default=severities, label_visibility="collapsed")
            to_show = [ri for ri in risk_items if (ri.get("severity") or "unknown") in sel]

            for ri in to_show:
                with st.container(border=True):
                    title = ri.get("title") or ri.get("code") or "Risk"
                    sev = (ri.get("severity") or "unknown").upper()
                    conf = ri.get("confidence", 0.0)
                    tags = [t.lower() for t in (ri.get("tags") or [])]
                    if "llm-mined" in tags:
                        if len(tags) > 1:
                            # means it was merged with other tags (rule-based + LLM)
                            st.caption("🔎 LLM-proposed (merged)")
                        else:
                            st.caption("🔎 LLM-proposed")
                    st.markdown(f"**{title}**  ·  _severity_: **{sev}**  ·  _confidence_: {conf:.2f}")
                    if tags:
                        st.caption("Tags: " + ", ".join(tags))

                    # LLM justification (short)
                    if ri.get("llm_notes"):
                        st.write(ri["llm_notes"])
                    else:
                        # fallback to rationale
                        rationale = ri.get("rationale")
                        if rationale:
                            st.write(rationale)

                    # Evidence (rows/snippets/locators)
                    ev = ri.get("evidence") or []
                    primary = [e for e in ev if (e.get("role","primary") == "primary")]
                    context = [e for e in ev if e.get("role") == "context"]

                    if primary:
                        st.markdown("_Evidence_")
                        for e in primary:
                            st.code(f"{e.get('source')} | {e.get('locator')} → {e.get('snippet')}")
                            if e.get("source_anchor"):
                                st.caption(f"anchor: {e['source_anchor']}")

                    if context:
                        with st.expander(f"More context ({len(context)})"):
                            for e in context:
                                st.code(f"{e.get('source')} | {e.get('locator')} → {e.get('snippet')}")
                                if e.get("source_anchor"):
                                    st.caption(f"anchor: {e['source_anchor']}")


        st.divider()

        # ---- Existing COPE score & nuanced concerns (unchanged) ----
        st.subheader("COPE Score & Nuanced Concerns")
        cope, concerns = run_risk_pipeline(results)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("COPE Score", cope.get("score", "—"))
            st.json(cope.get("breakdown", {}))
        with col2:
            st.write("Nuanced Concerns")
            for c in concerns.get("concerns", []):
                st.markdown(f"- **[{c.get('severity')}] {c.get('title')}**")
                st.caption(c.get("rationale", ""))

        st.subheader("Indicative Pricing")
        price = price_submission(results, cope, concerns)
        st.json(price)


# --- Tab 7: Schema Review --- #
with tabs[7]:
    import json, difflib, re
    from copy import deepcopy
    from pathlib import Path
    import streamlit as st

    # Helpers we already rely on elsewhere
    from core.schemas.active import (
        load_active_schema, get_active_name, set_active_name,
        active_keys, active_titles
    )
    from core.schemas.schema_builder import (
        propose_vnext_schema, write_vnext_and_point_active, generate_vnext_schema
    )
    # Crosswalk IO (these exist in your project)
    from core.extraction.field_mapping import _read_crosswalk, _write_crosswalk

    results = state.get("results") or {}
    proposals = (results.get("proposals") or {})
    sov_props = proposals.get("sov") or []
    loss_props = proposals.get("loss_run") or []

    st.subheader("Schema Review")

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    PROV_PREFIX = "_provisional_"

    def _strip_provisional(key: str) -> str:
        return key[len(PROV_PREFIX):] if isinstance(key, str) and key.startswith(PROV_PREFIX) else key

    def _norm_src_header(s: str) -> str:
        """Local normalizer for crosswalk keys (lowercase + collapse non-alnum)."""
        s = (s or "").lower().strip()
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        return s

    def _load_props_for(doc_type: str):
        if doc_type == "sov":
            return sov_props
        if doc_type == "loss_run":
            return loss_props
        return []

    # Session buckets for approvals (mappings) & queued new fields
    state.setdefault("approved_maps", {"sov": [], "loss_run": []})
    state.setdefault("approved_new_fields", {"sov": {}, "loss_run": {}})

    # ---------------------------------------------------------------------
    # Section A: Review LLM proposals and approve mappings
    # ---------------------------------------------------------------------
    st.markdown("### A) Review & approve LLM proposals (header → machine key)")

    if not sov_props and not loss_props:
        st.info("No LLM proposals available. Process a pack with unknown headers to see suggestions.")
    else:
        colA, colB = st.columns(2)

        with colA:
            st.markdown("**SOV proposals**")
            if not sov_props:
                st.caption("No SOV proposals.")
            else:
                # show the latest proposal first
                latest = sov_props[-1]
                mappings = latest.get("mappings") or []
                if not mappings:
                    st.caption("No mapping suggestions in the latest SOV proposal.")
                else:
                    with st.form("approve_sov_maps"):
                        sel = []
                        for i, m in enumerate(mappings, 1):
                            src = m.get("source_header", "")
                            tgt = m.get("target_field", "")
                            conf = float(m.get("confidence", 0))
                            rat = m.get("rationale", "")
                            chk = st.checkbox(
                                f"[{conf:.2f}] {src} → {tgt}",
                                value=False,
                                key=f"sov_map_{i}"
                            )
                            if chk:
                                sel.append({
                                    "doc_type": "sov",
                                    "source_header": src,
                                    "target_field": tgt,
                                    "confidence": conf,
                                    "rationale": rat,
                                })
                        if st.form_submit_button("Approve selected SOV mappings"):
                            # Append to session; actual persistence happens in Section C
                            state["approved_maps"]["sov"].extend(sel)
                            st.success(f"Queued {len(sel)} SOV mapping(s) for Apply.")

        with colB:
            st.markdown("**Loss Run proposals**")
            if not loss_props:
                st.caption("No Loss Run proposals.")
            else:
                latest = loss_props[-1]
                mappings = latest.get("mappings") or []
                if not mappings:
                    st.caption("No mapping suggestions in the latest Loss Run proposal.")
                else:
                    with st.form("approve_loss_maps"):
                        sel = []
                        for i, m in enumerate(mappings, 1):
                            src = m.get("source_header", "")
                            tgt = m.get("target_field", "")
                            conf = float(m.get("confidence", 0))
                            rat = m.get("rationale", "")
                            chk = st.checkbox(
                                f"[{conf:.2f}] {src} → {tgt}",
                                value=False,
                                key=f"loss_map_{i}"
                            )
                            if chk:
                                sel.append({
                                    "doc_type": "loss_run",
                                    "source_header": src,
                                    "target_field": tgt,
                                    "confidence": conf,
                                    "rationale": rat,
                                })
                        if st.form_submit_button("Approve selected Loss Run mappings"):
                            state["approved_maps"]["loss_run"].extend(sel)
                            st.success(f"Queued {len(sel)} Loss Run mapping(s) for Apply.")

    # ---------------------------------------------------------------------
    # Section B: Queue NEW fields for vNext (Profiler + LLM)
    # ---------------------------------------------------------------------
    st.markdown("### B) Queue NEW fields for vNext (Profiler + LLM)")

    # Gather not-in-ACTIVE suggestions from the extraction results (already filtered up-stream)
    prof_sug = ((results or {}).get("schema_suggestions") or {})

    def _listify_suggestions_block(x):
        if not x:
            return []
        if isinstance(x, list):
            return [str(v) for v in x]
        if isinstance(x, dict):
            if "suggestions" in x and isinstance(x["suggestions"], list):
                return [str(v) for v in x["suggestions"]]
            if "properties" in x and isinstance(x["properties"], dict):
                return [str(k) for k in x["properties"].keys()]
            return [str(k) for k in x.keys()]
        return []

    prof_sov_list  = _listify_suggestions_block(prof_sug.get("sov"))
    prof_loss_list = _listify_suggestions_block(prof_sug.get("loss_run"))

    # LLM "new_fields" (from the latest proposals lists built above in this tab)
    def _latest_new_fields(props_list):
        if not props_list:
            return []
        latest = props_list[-1]
        return [nf.get("field_name") for nf in (latest.get("new_fields") or []) if nf.get("field_name")]

    llm_sov_new  = _latest_new_fields(sov_props)
    llm_loss_new = _latest_new_fields(loss_props)

    # Filter LLM new_fields against ACTIVE so only not-in-ACTIVE remain
    llm_sov_new  = _filter_new_fields("SOV", llm_sov_new)
    llm_loss_new = _filter_new_fields("LOSS_RUN", llm_loss_new)

    # ---- Robust de-dup logic (UI level)
    # Group by CLEAN base name; if any provisional form exists, display that one.
    # When approving, ALWAYS save the CLEAN base (no prefix).
    def _prepare_candidates(*lists):
        grouped = {}
        for name in [str(v).strip() for L in lists for v in (L or [])]:
            if not name:
                continue
            base = _strip_provisional(name)
            current = grouped.get(base)
            if current is None:
                grouped[base] = name
            else:
                if name.startswith(PROV_PREFIX) and not current.startswith(PROV_PREFIX):
                    grouped[base] = name
        return [(disp, base) for base, disp in grouped.items()]

    sov_candidates  = _prepare_candidates(prof_sov_list,  llm_sov_new)
    loss_candidates = _prepare_candidates(prof_loss_list, llm_loss_new)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**SOV – approve new fields**")
        if not sov_candidates:
            st.caption("No SOV new fields from Profiler/LLM.")
        else:
            with st.form("approve_sov_newfields"):
                adds = {}
                for display_name, clean_base in sorted(sov_candidates, key=lambda x: x[1]):
                    chk = st.checkbox(f"Queue: {display_name}", value=False, key=f"sov_nf_{display_name}")
                    if chk:
                        adds[clean_base] = {"type": "string", "title": clean_base.replace('_',' ').title()}
                if st.form_submit_button("Queue selected SOV new fields"):
                    state["approved_new_fields"]["sov"].update(adds)
                    st.success(f"Queued {len(adds)} SOV field(s) for vNext (stored clean).")

    with col2:
        st.markdown("**Loss Run – approve new fields**")
        if not loss_candidates:
            st.caption("No Loss Run new fields from Profiler/LLM.")
        else:
            with st.form("approve_loss_newfields"):
                adds = {}
                for display_name, clean_base in sorted(loss_candidates, key=lambda x: x[1]):
                    chk = st.checkbox(f"Queue: {display_name}", value=False, key=f"loss_nf_{display_name}")
                    if chk:
                        adds[clean_base] = {"type": "string", "title": clean_base.replace('_',' ').title()}
                if st.form_submit_button("Queue selected Loss Run new fields"):
                    state["approved_new_fields"]["loss_run"].update(adds)
                    st.success(f"Queued {len(adds)} Loss Run field(s) for vNext (stored clean).")


    # ---------------------------------------------------------------------
    # Section C: Enrich queued field properties with LLM (optional)
    # ---------------------------------------------------------------------
    st.markdown("### C) Enrich queued field properties with LLM (optional)")

    from core.extraction.pipeline import collect_field_samples
    from core.schemas.schema_builder import enrich_properties_with_llm

    # Prefer single source of truth for the provisional prefix
    try:
        from core.extraction.field_mapping import PROVISIONAL_PREFIX as PROV_PREFIX
    except Exception:
        PROV_PREFIX = "_provisional_"

    def _sanitize_samples(vals):
        """Trim strings, coerce common boolean-like strings to booleans, dedupe while preserving order."""
        out = []
        for v in (vals or []):
            if isinstance(v, str):
                s = v.strip()
                low = s.lower()
                if low in ("true", "false", "yes", "no", "y", "n", "1", "0"):
                    out.append(low in ("true", "yes", "y", "1"))
                else:
                    out.append(s)
            else:
                out.append(v)
        # stable de-dupe (by repr to keep booleans distinct from strings)
        seen = set()
        dedup = []
        for x in out:
            k = repr(x)
            if k not in seen:
                seen.add(k)
                dedup.append(x)
        return dedup[:30]

    for bucket in ("sov", "loss_run"):
        queued = state["approved_new_fields"].get(bucket) or {}
        if not queued:
            st.caption(f"No queued fields for **{bucket}**.")
            continue

        # Collect value samples from the normalized rows (already in results)
        rows = (results or {}).get("sov" if bucket == "sov" else "loss_runs") or []

        # Collect samples for both clean and provisional keys
        fields = list(queued.keys())
        fields_to_collect = fields + [f"{PROV_PREFIX}{f}" for f in fields]
        samples_raw = collect_field_samples(rows, fields_to_collect, limit_per_field=30)

        # Collapse to clean keys: prefer clean samples; else use provisional, then sanitize
        samples = {}
        for f in fields:
            raw = samples_raw.get(f) or samples_raw.get(f"{PROV_PREFIX}{f}") or []
            samples[f] = _sanitize_samples(raw)

        colA, colB = st.columns([0.7, 0.3])
        with colA:
            with st.expander(f"Queued fields for {bucket} (click to view)", expanded=False):
                st.json(queued)
            with st.expander(f"Sample preview for {bucket} (sanitized)", expanded=False):
                # show only the first few examples per field so you can confirm input quality
                st.json({f: samples.get(f, [])[:10] for f in fields})
            st.caption(f"Fields: {len(fields)} • With samples (any): {sum(1 for f in fields if samples.get(f))}")

        with colB:
            if st.button(f"LLM-enrich {bucket} queued fields", key=f"llm_enrich_{bucket}"):
                enriched = enrich_properties_with_llm(bucket, queued, samples)
                # Update in-place so the Generate step uses richer properties
                state["approved_new_fields"][bucket] = enriched
                st.success(f"Enriched {len(enriched)} field(s) for **{bucket}**.")
                with st.expander(f"Preview enriched properties for {bucket}", expanded=False):
                    st.json(enriched)




    # ---------------------------------------------------------------------
    # Section D: APPLY approved mappings → crosswalk.json (FIX HERE)
    # ---------------------------------------------------------------------
    st.markdown("### D) Apply approved mappings to crosswalk.json")

    if st.button("Apply approved mappings"):
        xw = _read_crosswalk()  # dict: normalized source header -> target field

        # Build a flat list of approvals from both buckets
        combined = (state["approved_maps"].get("sov") or []) + (state["approved_maps"].get("loss_run") or [])

        # Normalize and strip provisional before persisting
        applied = 0
        skipped = []
        for m in combined:
            raw_src = (m.get("source_header") or "").strip()
            raw_tgt = (m.get("target_field") or "").strip()
            if not raw_src or not raw_tgt:
                continue

            src_norm = _norm_src_header(raw_src)
            tgt_clean = _strip_provisional(raw_tgt)

            # Optional: sanity check against active schema keys
            dt = m.get("doc_type") or ""
            if dt in ("sov", "loss_run"):
                if tgt_clean not in set(active_keys(dt)):
                    # surface to user but still allow writing if you prefer
                    skipped.append(f"[{dt}] {raw_src} → {raw_tgt} (clean='{tgt_clean}') not in active schema")
                    # If you want to hard-block, `continue` here.
                    # continue

            xw[src_norm] = tgt_clean
            applied += 1

        _write_crosswalk(xw)

        # Clear queue after apply
        state["approved_maps"] = {"sov": [], "loss_run": []}

        if applied:
            st.success(f"Applied {applied} mapping(s) to crosswalk.json (provisional prefix stripped).")
        if skipped:
            with st.expander("Some mappings were not recognized in active schema (FYI)", expanded=False):
                st.write("\n".join(skipped))

    # ---------------------------------------------------------------------
    # Section E: Generate & activate vNext schema (optional) — unified controls
    # ---------------------------------------------------------------------
    st.markdown("### E) Generate & activate vNext schema (optional)")

    # One selector for bucket, one "Make ACTIVE" checkbox, one generate button
    bucket = st.selectbox(
        "Choose schema bucket",
        options=("sov", "loss_run"),
        index=0,
        key="gen_bucket_select",
        help="Select which schema to generate the vNext for."
    )
    queued = state["approved_new_fields"].get(bucket) or {}

    colX, colY, colZ = st.columns([0.5, 0.25, 0.25])
    with colX:
        st.caption(f"Queued new fields for **{bucket}**: {len(queued)}")
    with colY:
        make_active_np = st.checkbox(
            "Make ACTIVE",
            value=False,  # <-- do not pre-select
            key="make_active_np_unified",
            help="If checked, the active pointer will be updated to the newly generated schema."
        )
    with colZ:
        gen_disabled = (len(queued) == 0)
        if st.button(
            "Generate vNext schema",
            key="gen_noprev_unified",
            disabled=gen_disabled
        ):
            new_name, cur, vnext = propose_vnext_schema(bucket, queued)
            dest = write_vnext_and_point_active(bucket, new_name, vnext, make_active=make_active_np)
            state["approved_new_fields"][bucket] = {}
            st.success(
                f"Wrote {dest.name} and "
                f"{'updated' if make_active_np else 'did not change'} active pointer for **{bucket}**."
            )
            st.info("Click **Process** to validate against the new schema.")

    # Optional preview (uses the same single Make ACTIVE checkbox above)
    if st.checkbox("Preview vNext schema diff", value=False, key="diff_unified"):
        new_name, cur, vnext = propose_vnext_schema(bucket, queued)
        cur_txt = json.dumps(cur, indent=2, ensure_ascii=False).splitlines(keepends=True)
        nxt_txt = json.dumps(vnext, indent=2, ensure_ascii=False).splitlines(keepends=True)
        diff = difflib.unified_diff(cur_txt, nxt_txt, fromfile="current", tofile=new_name, n=2)
        st.code("".join(diff) or "(no changes)")

        # use the same single generate button (no duplicate “make active” checkbox here)
        if st.button("Generate vNext schema (from preview)", key="gen_preview_unified", disabled=(len(queued) == 0)):
            dest = write_vnext_and_point_active(bucket, new_name, vnext, make_active=make_active_np)
            state["approved_new_fields"][bucket] = {}
            st.success(
                f"Wrote {dest.name} and "
                f"{'updated' if make_active_np else 'did not change'} active pointer for **{bucket}**."
            )
            st.info("Click **Process** to validate against the new schema.")

    # ---------------------------------------------------------------------
    # Section F: EXPERIMENTAL — Draft a full schema with LLM (from data)
    # ---------------------------------------------------------------------
    import time
    from datetime import datetime as _dt
    st.markdown("### F) Experimental: Draft a full schema with LLM (from data)")

    from core.schemas.schema_builder import propose_full_schema_from_llm, write_vnext_and_point_active
    # Optional diagnostics if you've added them in schema_builder.py:
    try:
        from core.schemas.schema_builder import get_llm_last_error, get_llm_last_meta
    except Exception:
        def get_llm_last_error(): return None
        def get_llm_last_meta(): return None

    # --- Controls to create a new draft ---
    col1, col2, col3 = st.columns([0.35, 0.35, 0.30])
    with col1:
        bucket_full = st.selectbox(
            "Choose bucket to draft",
            options=("sov", "loss_run"),
            index=0,
            key="llm_full_bucket",
            help="The LLM will analyze current normalized rows and propose a full JSON Schema."
        )
    with col2:
        max_fields = st.slider("Max fields to include", 10, 120, 60, 5, key="llm_full_max_fields")
    with col3:
        samples_per = st.slider("Samples per field", 3, 30, 10, 1, key="llm_full_samples_per")

    rows_for_bucket = (results or {}).get("sov" if bucket_full == "sov" else "loss_runs") or []

    # Small diagnostics panel
    with st.expander("LLM diagnostics (click to view)", expanded=False):
        import os
        st.write({
            "OPENAI_API_KEY set": bool(os.environ.get("OPENAI_API_KEY")),
            "OPENAI_MODEL": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "rows_available": len(rows_for_bucket),
            "last_llm_error": get_llm_last_error(),
        })

    # --- Create a new draft ---
    btn_disabled = (len(rows_for_bucket) == 0)
    if st.button("Draft full schema with LLM", key="btn_llm_full_draft", disabled=btn_disabled):
        t0 = time.perf_counter()
        started_iso = _dt.now().isoformat(timespec="seconds")

        new_name, cur_active, draft_schema = propose_full_schema_from_llm(
            bucket_full, rows_for_bucket, max_fields=max_fields, samples_per_field=samples_per
        )
        duration_ms = round((time.perf_counter() - t0) * 1000)

        # Persist the draft so it survives re-runs
        state["llm_full_draft"] = {
            "bucket": bucket_full,
            "file_name": new_name,
            "current": cur_active,
            "draft": draft_schema,
        }

        # Pull diagnostics from schema_builder (model/usage + any error)
        try:
            from core.schemas.schema_builder import get_llm_last_error, get_llm_last_meta
            last_err = get_llm_last_error()
            last_meta = get_llm_last_meta() or {}
        except Exception:
            last_err = None
            last_meta = {}

        # Record “Last Call” for the sidebar (write to both stores)
        payload = {
            "ts": started_iso,
            "action": f"full_schema_draft:{bucket_full}",
            "bucket": bucket_full,
            "duration_ms": duration_ms,
            "model": last_meta.get("model"),
            "usage": last_meta.get("usage"),
            "fields": list((draft_schema or {}).get("properties", {}).keys())[:10],
            "error": last_err,
        }
        state["last_llm_call"] = payload
        st.session_state["last_llm_call"] = payload  # <- ensure sidebar can read immediately

        # Show outcome in the main pane
        if last_err:
            st.warning(f"LLM reported: {last_err}")
        else:
            st.success(f"Draft ready: {new_name}")

        # IMPORTANT: force a rerun so the sidebar (rendered above) picks up the new state now
        try:
            st.rerun()   # Streamlit >=1.32
        except Exception:
            st.experimental_rerun()  # older Streamlit fallback


    # --- Resume last draft (persistent preview + generate) ---
    draft_state = state.get("llm_full_draft")
    if draft_state:
        st.markdown("#### Resume last LLM draft")
        bkt   = draft_state.get("bucket")
        fname = draft_state.get("file_name")
        cur   = draft_state.get("current") or {"type": "object", "properties": {}}
        draft = draft_state.get("draft")   or {"type": "object", "properties": {}}

        colA, colB = st.columns([0.6, 0.4])
        with colA:
            st.caption(f"Bucket: **{bkt}** • File: **{fname}**")
            with st.expander("Preview draft schema JSON", expanded=False):
                st.code(json.dumps(draft, indent=2, ensure_ascii=False), language="json")
            # Diff vs ACTIVE
            try:
                cur_txt   = json.dumps(cur,   indent=2, ensure_ascii=False).splitlines(keepends=True)
                draft_txt = json.dumps(draft, indent=2, ensure_ascii=False).splitlines(keepends=True)
                import difflib
                diff = difflib.unified_diff(cur_txt, draft_txt, fromfile="ACTIVE", tofile=fname, n=2)
                st.code("".join(diff) or "(no changes)")
            except Exception:
                st.info("Diff unavailable.")
        with colB:
            make_active = st.checkbox("Make ACTIVE", value=False, key="llm_full_make_active_persist")
            # Guard if draft is empty
            disabled_save = not isinstance(draft, dict) or not (draft.get("properties") or {})
            if st.button("Save draft as vNext", key="btn_llm_full_write_persist", disabled=disabled_save):
                dest = write_vnext_and_point_active(bkt, fname, draft, make_active=make_active)
                st.success(
                    f"Wrote {dest.name} and "
                    f"{'updated' if make_active else 'did not change'} active pointer for **{bkt}**."
                )
                st.info("Click **Process** to validate against the new schema.")
    else:
        st.caption("No LLM draft in session yet. Use the control above to create one.")





    # ---------------------------------------------------------------------
    # Section G: Quick status
    # ---------------------------------------------------------------------
    with st.expander("Current queues / debug", expanded=False):
        st.write("**Approved mappings (pending apply):**")
        st.json(state.get("approved_maps"))
        st.write("**Queued new fields for vNext:**")
        st.json(state.get("approved_new_fields"))
        st.write("**Active schema files:**")
        st.json({
            "sov": get_active_name("sov"),
            "loss_run": get_active_name("loss_run")
        })


# ===== JSON TAB HELPERS (add these just above the JSON tab) =====
from typing import List, Dict

def _listify_suggestions(raw) -> List[str]:
    """
    Try to coerce 'suggestions' shaped in a few common ways into a flat list[str].
    Accepts: list[str], {'suggestions': [...]}, {'properties': {...}} etc.
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        if "suggestions" in raw and isinstance(raw["suggestions"], list):
            return [str(x) for x in raw["suggestions"]]
        if "properties" in raw and isinstance(raw["properties"], dict):
            return [str(k) for k in raw["properties"].keys()]
        # fall back to keys if dict is already a {field:meta}
        return [str(k) for k in raw.keys()]
    return []

def _extract_profiler_suggestions(results: dict) -> Dict[str, List[str]]:
    """
    Returns {'SOV': [...], 'LOSS_RUN': [...]} of profiler suggestions.
    Pulls from results['schema_suggestions'] if present; otherwise empty lists.
    """
    out = {"SOV": [], "LOSS_RUN": []}
    if not results:
        return out
    sug = (results or {}).get("schema_suggestions", {}) or {}
    out["SOV"] = _listify_suggestions(sug.get("sov"))
    out["LOSS_RUN"] = _listify_suggestions(sug.get("loss_run"))
    return out

def _extract_llm_field_suggestions(results: dict) -> Dict[str, List[str]]:
    """
    Returns {'SOV': [...], 'LOSS_RUN': [...]} of LLM-suggested field names.
    Sources:
      1) results['proposals'][bucket][-1]['new_fields'][*]['field_name']
      2) results['schema_inference']['_llm'][bucket]['properties'].keys()
    """
    out = {"SOV": [], "LOSS_RUN": []}
    if not results:
        return out

    # 1) From LLM mapping proposals: new_fields array (if present)
    props = (results.get("proposals") or {})
    for bucket, key in (("sov", "SOV"), ("loss_run", "LOSS_RUN")):
        lst = props.get(bucket) or []
        if lst:
            latest = lst[-1]
            new_fields = (latest or {}).get("new_fields") or []
            for nf in new_fields:
                name = (nf or {}).get("field_name")
                if name:
                    out[key].append(str(name))

    # 2) From schema_inference._llm properties
    inf = (results or {}).get("schema_inference", {})
    llm = (inf or {}).get("_llm", {})
    for bucket, key in (("sov", "SOV"), ("loss_run", "LOSS_RUN")):
        props2 = ((llm.get(bucket) or {}).get("properties") or {})
        if isinstance(props2, dict):
            out[key].extend([str(k) for k in props2.keys()])

    # de-dupe while preserving order
    def _uniq(seq): 
        s, outl = set(), []
        for x in seq:
            if x not in s:
                s.add(x); outl.append(x)
        return outl

    out["SOV"] = _uniq(out["SOV"])
    out["LOSS_RUN"] = _uniq(out["LOSS_RUN"])
    return out

def render_json_tab_filtered_suggestions(results: dict):
    """
    Pretty renderer for: Schema Profiling & Suggestions (de-duped, not in ACTIVE)
    - Shows compact badges & counts
    - Collapsible tables per bucket (SOV / Loss Run)
    - Optional full JSON view kept in expanders
    """
    import streamlit as st
    import pandas as pd

    # --- Gather Profiler + LLM suggestions (already filtered elsewhere) ---
    prof = _extract_profiler_suggestions(results)
    llm  = _extract_llm_field_suggestions(results)

    sov_prof_new = _filter_new_fields("SOV", prof["SOV"])
    lr_prof_new  = _filter_new_fields("LOSS_RUN", prof["LOSS_RUN"])
    sov_llm_new  = _filter_new_fields("SOV", llm["SOV"])
    lr_llm_new   = _filter_new_fields("LOSS_RUN", llm["LOSS_RUN"])

    sov_combined = _clean_list(sov_prof_new + sov_llm_new)
    lr_combined  = _clean_list(lr_prof_new + lr_llm_new)

    # --- Tiny badge helper ---
    def pill(label: str, count: int, color: str = "#0ea5e9"):
        bg = color if count > 0 else "#9ca3af"
        return f"""
        <span style="
            display:inline-block;
            padding:6px 10px;
            margin:2px 6px 2px 0;
            border-radius:999px;
            font-size:12px;
            font-weight:600;
            color:white;
            background:{bg};
            white-space:nowrap;">
            {label}: {count}
        </span>
        """

    # --- Header ---
    st.subheader("Schema Profiling & Suggestions (de-duped, not in ACTIVE)")

    # --- Summary badges row ---
    st.markdown(
        pill("SOV total", len(sov_combined), "#16a34a") +
        pill("SOV (Profiler)", len(sov_prof_new), "#0ea5e9") +
        pill("SOV (LLM)", len(sov_llm_new), "#8b5cf6") +
        "&nbsp;&nbsp;" +
        pill("Loss total", len(lr_combined), "#16a34a") +
        pill("Loss (Profiler)", len(lr_prof_new), "#0ea5e9") +
        pill("Loss (LLM)", len(lr_llm_new), "#8b5cf6"),
        unsafe_allow_html=True
    )

    # --- Build nice tables ---
    def as_table(fields: list[str], source: str, bucket: str) -> pd.DataFrame:
        return pd.DataFrame(
            [{"Field": f, "Source": source, "Bucket": bucket} for f in fields]
        )

    sov_df = pd.concat(
        [
            as_table(sov_prof_new, "Profiler", "SOV"),
            as_table(sov_llm_new,  "LLM",      "SOV"),
        ],
        ignore_index=True
    )

    lr_df = pd.concat(
        [
            as_table(lr_prof_new, "Profiler", "Loss Run"),
            as_table(lr_llm_new,  "LLM",      "Loss Run"),
        ],
        ignore_index=True
    )

    # --- Two pretty blocks side-by-side ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### SOV — New field suggestions")
        if sov_df.empty:
            st.info("No SOV suggestions (all present in ACTIVE).")
        else:
            # Compact table with automatic height
            st.dataframe(
                sov_df.sort_values(["Source", "Field"]).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Show JSON (pretty)"):
                st.json(
                    {
                        "profiler": sov_prof_new,
                        "llm": sov_llm_new,
                        "combined_unique": sov_combined,
                    }
                )

    with col2:
        st.markdown("#### Loss Run — New field suggestions")
        if lr_df.empty:
            st.info("No Loss Run suggestions (all present in ACTIVE).")
        else:
            st.dataframe(
                lr_df.sort_values(["Source", "Field"]).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Show JSON (pretty)"):
                st.json(
                    {
                        "profiler": lr_prof_new,
                        "llm": lr_llm_new,
                        "combined_unique": lr_combined,
                    }
                )


# --- Tab 8: JSON --- #
with tabs[8]:
    st.subheader("Raw Extracted JSON")
    results = state.get("results")

    # --- Debug toggles (kept) ---
    st.markdown("### 🔍 Debug")
    debug_raw = st.checkbox(
        "Show raw LLM responses (before cleaning/filtering)",
        value=False,
        help="Displays the exact text returned by the LLM for each doc_type."
    )
    debug_parsed = st.checkbox(
        "Show parsed LLM JSON (before post-filtering)",
        value=False,
        help="If available, shows the parsed object directly from the LLM output, before we filter non-allowed target_field values."
    )

    if debug_raw:
        raw_store = state.get("proposals_raw") or {}
        st.subheader("Raw LLM Responses")
        if raw_store:
            for bucket, items in raw_store.items():
                latest = items[-1:] if isinstance(items, list) else items
                with st.expander(f"RAW • {bucket} (latest only)", expanded=False):
                    if isinstance(latest, list):
                        for idx, txt in enumerate(latest, 1):
                            st.markdown(f"**{bucket} raw (latest)**")
                            st.code(txt, language="json")
                    else:
                        st.code(latest, language="json")

        else:
            st.info("No raw LLM responses stored. Run **Process** after triggering Schema Discovery.")

    if debug_parsed:
        parsed_store = state.get("proposals_parsed") or {}
        st.subheader("Parsed LLM JSON (pre-filter)")
        if parsed_store:
            for bucket, items in parsed_store.items():
                with st.expander(f"PARSED • {bucket} ({len(items)} object(s))", expanded=False):
                    for idx, obj in enumerate(items, 1):
                        st.markdown(f"**{bucket} parsed #{idx}**")
                        st.json(obj)
        else:
            st.info("No parsed pre-filter proposals stored.")

    if results:
        # Show core results exactly as used in the app
        st.subheader("Processed Results (used by UI)")
        st.json(results)

        # Show model proposals (post-filter) exactly as produced
        st.subheader("Schema Discovery – Proposals (post-filter)")
        props = (results or {}).get("proposals", {})
        if props and (props.get("sov") or props.get("loss_run")):
            st.success("Schema Discovery proposals generated via LLM.")
            if props.get("sov"):
                st.markdown("**SOV proposals**")
                for i, p in enumerate(props["sov"], 1):
                    with st.expander(f"SOV proposal #{i}"):
                        st.json(p)
            if props.get("loss_run"):
                st.markdown("**Loss Run proposals**")
                for i, p in enumerate(props["loss_run"], 1):
                    with st.expander(f"Loss proposal #{i}"):
                        st.json(p)
        else:
            st.info("No proposals (either all headers mapped via crosswalk, or no OPENAI_API_KEY set).")

        # ===== NEW: Filtered suggestions (Profiler + LLM) not-in-ACTIVE =====
        render_json_tab_filtered_suggestions(results)

        # (Optional) LLM-assisted raw property suggestions block (unchanged)
        inf = (results or {}).get("schema_inference", {})
        llm = (inf or {}).get("_llm", {})
        if llm:
            st.subheader("LLM-Assisted Suggestions (raw properties)")
            for bucket in ("sov", "loss_run"):
                props2 = (llm.get(bucket) or {}).get("properties") or {}
                if props2:
                    with st.expander(f"{bucket.upper()} – LLM proposed properties", expanded=False):
                        st.json(props2)

        st.subheader("Schema Validation (summary)")
        st.write({
            "sov_issues": len((results or {}).get("sov_validation", []) or []),
            "loss_run_issues": len((results or {}).get("loss_run_validation", []) or []),
        })
    else:
        st.info("Run **Process** to see extracted JSON & proposals.")



