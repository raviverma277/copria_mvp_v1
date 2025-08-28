# core/parsing/docx_parser.py
from __future__ import annotations
from typing import Any, Dict

# Reuse your existing Word parser so we don't touch the rest of the pipeline
from .word_parser import parse_word as _parse_word

def parse_docx(f) -> Dict[str, Any]:
    """
    Thin wrapper so dispatch can call parse_docx().
    Uses the existing word_parser.parse_word implementation.
    """
    item: Dict[str, Any] = _parse_word(f)
    # Make sure type + filename keys exist for downstream logic/dedupe/UI
    item.setdefault("type", "docx")
    item.setdefault("meta", {})
    return item
