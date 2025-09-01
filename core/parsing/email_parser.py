# core/parsing/email_parser.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parseaddr


def _norm_addr(addr: Tuple[str, str]) -> str:
    """Return a simple 'email' if present, otherwise the display name."""
    name, email = addr
    email = (email or "").strip()
    name = (name or "").strip()
    return email or name


def _collect_recipients(msg, header: str) -> List[str]:
    """Parse To/Cc headers into a list of clean email strings."""
    values = msg.get_all(header, [])
    if not values:
        return []
    pairs = getaddresses(values)  # [(name, email), ...]
    out = []
    for p in pairs:
        s = _norm_addr(p)
        if s:
            out.append(s)
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for s in out:
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        uniq.append(s)
    return uniq


def _extract_body_text(msg) -> str:
    """
    Prefer text/plain parts and skip attachments. Handle common encodings.
    If multipart/alternative, first try the best-quality text/plain.
    """
    # Single-part text/plain
    if msg.get_content_type() == "text/plain" and not msg.get_filename():
        try:
            return msg.get_content() or ""
        except Exception:
            payload = msg.get_payload(decode=True) or b""
            try:
                return payload.decode(
                    msg.get_content_charset() or "utf-8", errors="replace"
                )
            except Exception:
                return payload.decode("utf-8", errors="replace")

    # Multiparts
    if msg.is_multipart():
        # search text/plain parts, ignore attachments
        candidates = []
        for part in msg.walk():
            if part.is_multipart():
                continue
            if part.get_filename():  # attachment
                continue
            if part.get_content_type() == "text/plain":
                try:
                    candidates.append(part.get_content() or "")
                except Exception:
                    payload = part.get_payload(decode=True) or b""
                    try:
                        candidates.append(
                            payload.decode(
                                part.get_content_charset() or "utf-8", errors="replace"
                            )
                        )
                    except Exception:
                        candidates.append(payload.decode("utf-8", errors="replace"))
        # choose the longest non-empty candidate
        if candidates:
            candidates.sort(key=lambda s: len(s or ""), reverse=True)
            return candidates[0] or ""

    return ""


def _extract_attachments(msg) -> List[Dict[str, Any]]:
    """
    Collect attachments: filename, content_type, payload bytes.
    Skips inline text parts.
    """
    atts: List[Dict[str, Any]] = []
    if not msg.is_multipart():
        return atts

    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        if not filename:
            # not an attachment (likely a body part)
            continue
        content_type = part.get_content_type()
        data = part.get_payload(decode=True) or b""
        atts.append(
            {
                "filename": filename,
                "content_type": content_type,
                "payload": data,
            }
        )
    return atts


def parse_email(file) -> Dict[str, Any]:
    """
    Parse .eml/.msg-like bytes (works with Streamlit UploadedFile).
    Returns:
      {
        "type": "email",
        "headers": {"from": str, "to": [str], "cc": [str], "date": str, "subject": str},
        "body_text": str,
        "attachments": [{"filename": str, "content_type": str, "payload": bytes}, ...]
      }
    """
    raw = file.read()
    # Streamlit may keep the handle open for re-reads; rewind for safety
    try:
        file.seek(0)
    except Exception:
        pass

    msg = BytesParser(policy=policy.default).parsebytes(raw)

    frm = _norm_addr(parseaddr(msg.get("from", "")))
    to_list = _collect_recipients(msg, "to")
    cc_list = _collect_recipients(msg, "cc")

    headers = {
        "from": frm,
        "to": to_list,
        "cc": cc_list,
        "date": msg.get("date", "") or "",
        "subject": msg.get("subject", "") or "",
    }

    body_text = _extract_body_text(msg)
    attachments = _extract_attachments(msg)

    return {
        "type": "email",
        "headers": headers,
        "body_text": body_text,
        "attachments": attachments,
    }
