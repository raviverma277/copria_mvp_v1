# core/schemas/proposals_store.py
from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import json

CROSSWALK = Path("core/schemas/crosswalk.json")
PENDING   = Path("core/schemas/pending/new_fields.json")
PENDING.parent.mkdir(parents=True, exist_ok=True)

def _read_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def apply_crosswalk_mappings(approved: List[Dict[str, Any]]) -> int:
    """
    Append/overwrite entries in crosswalk.json.
    approved: [{source_header, target_field}, ...]
    - source_header is lowercased before saving.
    - target_field is saved AS-IS (no normalization).
    Returns count of entries changed.
    """
    cw = _read_json(CROSSWALK)
    changed = 0
    for m in approved:
        src = (m.get("source_header") or "").strip().lower()
        tgt = (m.get("target_field") or "").strip()
        if not src or not tgt:
            continue
        if cw.get(src) != tgt:
            cw[src] = tgt
            changed += 1
    if changed:
        _write_json(CROSSWALK, cw)
    return changed

def append_new_fields(doc_type: str, new_fields: List[Dict[str, Any]]) -> int:
    """
    Queue proposed new fields for later governance.
    Stored under core/schemas/pending/new_fields.json:
      { "<doc_type>": [ {field_name, description, example?}, ... ] }
    """
    cur = _read_json(PENDING)
    bucket = cur.setdefault(doc_type, [])
    existing = {x.get("field_name") for x in bucket}
    added = 0
    for f in new_fields:
        fname = (f.get("field_name") or "").strip()
        if not fname or fname in existing:
            continue
        bucket.append({
            "field_name": fname,
            "description": f.get("description", ""),
            "example": f.get("example", "")
        })
        existing.add(fname)
        added += 1
    _write_json(PENDING, cur)
    return added

import json, os
from typing import List

PENDING_DIR = os.path.join(os.path.dirname(__file__), "pending")
PENDING_FILE = os.path.join(PENDING_DIR, "new_fields.json")

def _load_pending():
    if not os.path.exists(PENDING_FILE):
        return {"sov": [], "loss_run": []}
    with open(PENDING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_pending(data: dict):
    os.makedirs(PENDING_DIR, exist_ok=True)
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def remove_new_fields(doc_type: str, field_names: List[str]) -> int:
    """Remove any queued new_fields whose field_name is in field_names."""
    data = _load_pending()
    before = len(data.get(doc_type, []))
    keep = [nf for nf in data.get(doc_type, []) if nf.get("field_name") not in set(field_names)]
    data[doc_type] = keep
    _save_pending(data)
    return before - len(keep)
