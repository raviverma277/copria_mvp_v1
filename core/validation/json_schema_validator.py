from __future__ import annotations
from pathlib import Path
import json
from jsonschema import Draft7Validator
from core.schemas.active import load_active_schema as _load_active_schema

def load_active_schema(doc_type: str) -> dict:
    _, js = _load_active_schema(doc_type)
    return js or {}

def validate_rows(doc_type: str, rows: list[dict]) -> list[dict]:
    """
    Returns a list of error dicts: {"index": i, "errors": [str, ...]}
    """
    schema = load_active_schema(doc_type)
    if not schema:
        return []

    # Resolve array-of-objects shape ⇒ use "items" as effective schema for one row
    row_schema = schema.get("items") if isinstance(schema.get("items"), dict) else schema
    validator = Draft7Validator(row_schema)

    problems = []
    for i, row in enumerate(rows or []):
        errs = [f"{'.'.join([str(p) for p in e.path]) or '$'}: {e.message}" for e in validator.iter_errors(row)]
        if errs:
            problems.append({"index": i, "errors": errs})
    return problems
