# core/utils/llm_status.py
from __future__ import annotations
import os
import time
from typing import Optional

# ---- in-memory state (process-local) ----
_last_start_ts: Optional[float] = None     # epoch seconds
_last_duration_s: Optional[float] = None
_last_success: Optional[bool] = None
_last_model: Optional[str] = None

def record_llm_call_start(model_name: str):
    global _last_start_ts, _last_model, _last_success
    _last_start_ts = time.time()
    _last_model = model_name
    _last_success = None  # reset until end recorded

def record_llm_call_end(success: bool):
    global _last_duration_s, _last_success
    if _last_start_ts:
        _last_duration_s = time.time() - _last_start_ts
    _last_success = bool(success)

def get_llm_status():
    """
    Return a small status dict. Must NOT raise FileNotFoundError if .streamlit/secrets.toml is missing.
    Do NOT touch st.secrets at module import time. Only inside the function, under try/except.
    """
    api_key_set = False
    model = _last_model or os.environ.get("LLM_MINER_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"

    # Try reading Streamlit secrets only inside try/except
    try:
        import streamlit as st
        api_key_set = bool(os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", ""))
        if not os.environ.get("OPENAI_API_KEY"):
            v = st.secrets.get("OPENAI_API_KEY", "")
            if v:
                os.environ["OPENAI_API_KEY"] = str(v)
    except FileNotFoundError:
        api_key_set = bool(os.environ.get("OPENAI_API_KEY"))
    except Exception as e:
        print("[get_llm_status] suppressed:", repr(e))
        api_key_set = bool(os.environ.get("OPENAI_API_KEY"))

    # Render-friendly last call time (ISO) if we have one
    last_call_iso = None
    if _last_start_ts:
        try:
            from datetime import datetime, timezone
            last_call_iso = datetime.fromtimestamp(_last_start_ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
        except Exception:
            last_call_iso = None

    return {
        "api_key_set": api_key_set,
        "model": model,
        "last_call": last_call_iso,
        "last_duration": None if _last_duration_s is None else round(_last_duration_s, 2),
        "last_success": _last_success,
    }
