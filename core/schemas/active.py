# core/schemas/active.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

SCHEMA_DIR = Path("core/schemas/json")
ACTIVE_PTR = Path("core/schemas/active_schema.json")


# ------------ I/O helpers ------------
def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _norm_doc_type(doc_type: str) -> str:
    return (doc_type or "").strip().lower()


# ------------ Active pointer accessors ------------
def get_active_map() -> Dict[str, str]:
    """
    Returns the full mapping of {doc_type: active_schema_file_name}.
    Falls back to sensible defaults if pointer file is missing/empty.
    """
    default_map = {
        "sov": "sov.schema.json",
        "loss_run": "loss_run.schema.json",
        "questionnaire": "questionnaire.schema.json",
    }
    ptr = _read_json(ACTIVE_PTR)
    if not isinstance(ptr, dict):
        ptr = {}
    # keep any custom entries, but ensure defaults exist
    out = {
        **default_map,
        **{_norm_doc_type(k): v for k, v in ptr.items() if isinstance(v, str)},
    }
    return out


def set_active_map(mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Overwrite the active pointer file with the provided mapping.
    Keys are normalized to lowercase; values are written as-is.
    """
    clean = {_norm_doc_type(k): str(v) for k, v in (mapping or {}).items()}
    _write_json(ACTIVE_PTR, clean)
    return clean


def get_active_name(doc_type: str) -> str:
    """Return the active schema file name for a doc_type."""
    doc = _norm_doc_type(doc_type)
    return get_active_map().get(doc, "")


def set_active_name(doc_type: str, file_name: str) -> str:
    """
    Set the active schema file name for a single doc_type.
    Creates/updates ACTIVE_PTR. Returns the value written.
    """
    doc = _norm_doc_type(doc_type)
    ptr = get_active_map()
    ptr[doc] = str(file_name)
    _write_json(ACTIVE_PTR, ptr)
    return ptr[doc]


# ------------ Active schema loading ------------
def load_active_schema(doc_type: str) -> Tuple[Path, dict]:
    """Return (path, schema_json) for the currently active schema."""
    name = get_active_name(doc_type)
    path = SCHEMA_DIR / name
    return path, _read_json(path)


# ------------ Introspection helpers ------------
def _schema_keys_from_dict(obj) -> set[str]:
    """Recursively collect JSON Schema property names from common shapes."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        props = obj.get("properties")
        if isinstance(props, dict):
            keys.update([str(k) for k in props.keys()])
        for k in ("items", "$defs", "definitions", "allOf", "anyOf", "oneOf"):
            v = obj.get(k)
            if isinstance(v, dict):
                keys.update(_schema_keys_from_dict(v))
            elif isinstance(v, list):
                for el in v:
                    keys.update(_schema_keys_from_dict(el))
        for v in obj.values():
            if isinstance(v, (dict, list)):
                keys.update(_schema_keys_from_dict(v))
    elif isinstance(obj, list):
        for el in obj:
            keys.update(_schema_keys_from_dict(el))
    return keys


def active_keys(doc_type: str) -> List[str]:
    """All property keys from the active schema."""
    _, js = load_active_schema(doc_type)
    return sorted(_schema_keys_from_dict(js))


def active_titles(doc_type: str) -> Dict[str, str]:
    """Map of key -> title (if present) for nicer UI labels."""
    _, js = load_active_schema(doc_type)
    titles: Dict[str, str] = {}

    def _walk(o):
        if isinstance(o, dict):
            props = o.get("properties")
            if isinstance(props, dict):
                for k, meta in props.items():
                    t = str(meta.get("title", "")).strip()
                    if t:
                        titles[str(k)] = t
            for k in ("items", "$defs", "definitions", "allOf", "anyOf", "oneOf"):
                v = o.get(k)
                if isinstance(v, dict):
                    _walk(v)
                elif isinstance(v, list):
                    for el in v:
                        _walk(el)
            for v in o.values():
                if isinstance(v, (dict, list)):
                    _walk(v)
        elif isinstance(o, list):
            for el in o:
                _walk(el)

    _walk(js)
    return titles
