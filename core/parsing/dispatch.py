# core/parsing/dispatch.py
from __future__ import annotations
from typing import Any, Dict, List
from io import BytesIO
import os
from collections import defaultdict

from .pdf_parser import parse_pdf
from .excel_parser import parse_excel
from .docx_parser import parse_docx
from .email_parser import parse_email
from .classifier import DocumentClassifier


def _empty_bundle() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "submission": [],
        "sov": [],
        "loss_run": [],
        "questionnaire": [],
        "email_body": [],
        "other": [],
    }


def _route(
    bundle: Dict[str, List[Dict[str, Any]]], bucket: str, item: Dict[str, Any]
) -> None:
    if bucket not in bundle:
        bundle[bucket] = []
    bundle[bucket].append(item)


def _all_filenames(bndl: Dict[str, List[Dict[str, Any]]]) -> set[str]:
    names: set[str] = set()
    for arr in bndl.values():
        if not isinstance(arr, list):
            continue
        for it in arr:
            nm = (it.get("filename") or "").lower()
            if nm:
                names.add(nm)
    return names


def _bucket_from_page_tags(page_tags: List[Dict[str, Any]]) -> str | None:
    """
    Take page-level tags like {"page": 0, "type": "sov", "confidence": 0.92}
    and pick the dominant bucket by summing confidence per type.
    Returns a bucket if confident; otherwise None.
    """
    if not page_tags:
        return None
    scores = defaultdict(float)
    for t in page_tags:
        typ = (t.get("type") or "").lower()
        conf = float(t.get("confidence", 0) or 0)
        if typ in {"submission", "sov", "loss_run", "questionnaire"}:
            scores[typ] += conf
    if not scores:
        return None
    # pick top
    top_type, top_score = max(scores.items(), key=lambda kv: kv[1])
    # require minimum dominance to trust page tags
    total = sum(scores.values())
    if top_score >= 0.8 and top_score >= 0.5 * total:
        return top_type
    return None


def parse_files(files: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse and route a list of uploaded files (.pdf/.xlsx/.xls/.docx/.eml/.msg).
    Global de-duplication by filename is applied across the entire call so that
    if an .eml attachment and a separately uploaded file share a name, we only
    ingest one copy.
    """
    bundle = _empty_bundle()
    clf = DocumentClassifier()

    # GLOBAL de-dupe registry for this run
    seen_names: set[str] = set()

    for f in files:
        # Streamlit UploadedFile exposes .name; BytesIO we set .name upstream
        filename = os.path.basename(getattr(f, "name", "") or "upload")
        lower = filename.lower()

        # Skip if we've already processed a file with the same name
        if lower in seen_names:
            continue

        # ----- PDF -----
        if lower.endswith(".pdf"):
            item = parse_pdf(
                f
            )  # expected: {"type":"pdf","text":..., "page_tags":[...], "meta":...}
            item["filename"] = filename

            # Prefer page_tags if strong enough, else fall back to text classifier
            page_tags = item.get("page_tags", [])
            bucket = _bucket_from_page_tags(page_tags)
            if not bucket:
                result = clf.classify_text(item.get("text", ""), filename=filename)
                bucket = result.get("bucket", "other")

            _route(bundle, bucket, item)
            seen_names.add(lower)

        # ----- EXCEL -----
        elif lower.endswith((".xlsx", ".xls")):
            item = parse_excel(
                f
            )  # expected: {"type":"excel","sheets":[...], "meta":...}
            item["filename"] = filename
            result = clf.classify_excel(item.get("sheets", []), filename=filename)
            bucket = result.get("bucket", "other")
            _route(bundle, bucket, item)
            seen_names.add(lower)

        # ----- DOCX -----
        elif lower.endswith(".docx"):
            item = parse_docx(
                f
            )  # expected: {"type":"docx","text":..., "tables":..., "meta":...}
            item["filename"] = filename
            result = clf.classify_text(item.get("text", ""), filename=filename)
            bucket = result.get("bucket", "other")
            _route(bundle, bucket, item)
            seen_names.add(lower)

        # ----- EMAIL (.eml/.msg) -----
        elif lower.endswith((".eml", ".msg")):
            email = parse_email(
                f
            )  # expected: {"type":"email","headers":...,"body_text":...,"attachments":[...]}
            email["filename"] = filename
            _route(bundle, "email_body", email)
            seen_names.add(lower)

            # Names already present (from earlier in this pass)
            existing_names = _all_filenames(bundle)

            # Recursively parse attachments; skip duplicates by global+existing name
            for att in email.get("attachments", []):
                fname = (att.get("filename") or "attachment").lower()
                payload = att.get("payload")
                if not payload:
                    continue
                if fname in seen_names or fname in existing_names:
                    continue

                bio = BytesIO(payload)
                bio.name = fname  # preserve extension for classification

                sub = parse_files([bio])  # classify/parse attachment via same logic

                # Merge sub-bundle back in using _route, de-duping as we go
                for k, arr in sub.items():
                    if not isinstance(arr, list):
                        continue
                    for it in arr:
                        if "filename" not in it:
                            it["filename"] = os.path.basename(fname)
                        nm = (it.get("filename") or "").lower()
                        if nm and (nm not in seen_names) and (nm not in existing_names):
                            _route(bundle, k, it)
                            seen_names.add(nm)
                            existing_names.add(nm)

        # ----- UNKNOWN / FALLBACK -----
        else:
            # Best-effort text read for heuristic classification
            try:
                raw = f.read()
                try:
                    f.seek(0)
                except Exception:
                    pass
                try:
                    text = raw.decode("utf-8", errors="replace")
                except Exception:
                    text = ""
                item = {"type": "unknown", "text": text, "filename": filename}
                result = clf.classify_text(text, filename=filename)
                bucket = result.get("bucket", "other")
                _route(bundle, bucket, item)
                seen_names.add(lower)
            except Exception:
                # last resort: drop into 'other'
                _route(bundle, "other", {"filename": filename})
                seen_names.add(lower)

    return bundle
