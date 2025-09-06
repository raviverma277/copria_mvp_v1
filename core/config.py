# core/config.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

# Optional: load .env in local dev
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # We'll tolerate missing yaml and just use env defaults


# --- Feature toggles / constants ---
# QRA on by default, as discussed
QUICK_RISKS_ENABLED: bool = True


def _load_yaml(path: str | Path) -> dict:
    """Best-effort YAML loader; returns {} if file missing or pyyaml not installed."""
    p = Path(path)
    if not p.exists() or yaml is None:
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except Exception:
        return {}


def _getenv_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def get_config() -> Dict[str, Any]:
    """
    Central place for app/runtime config. 
    Merges (in order): defaults <- YAML (if present) <- ENV.
    Returns only keys the app actually uses.
    """
    # Look for a YAML config if you keep one; both locations are supported.
    cfg = {}
    for candidate in ("core/config.yaml", "config.yaml"):
        cfg.update(_load_yaml(candidate))

    # Defaults (keep these in sync with your app’s expectations)
    defaults = {
        "use_llm_miner": False,
        "llm_miner_model": "gpt-4o-mini",
        "pricing_strategy": "rules",
        "ingest_source": "LOCAL",  # LOCAL | CYTORA | AUTO
    }

    # Merge YAML over defaults
    merged = {**defaults, **cfg}

    # ENV overrides
    merged["use_llm_miner"]   = _getenv_bool("USE_LLM_MINER", merged["use_llm_miner"])
    merged["llm_miner_model"] = os.getenv("LLM_MINER_MODEL", merged["llm_miner_model"])
    merged["pricing_strategy"] = os.getenv("PRICING_STRATEGY", merged["pricing_strategy"])
    merged["ingest_source"]    = os.getenv("INGEST_SOURCE", merged["ingest_source"])

    return merged
