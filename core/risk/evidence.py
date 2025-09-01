# core/risk/evidence.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import re
import math

# -------------------------------
# Helpers
# -------------------------------


def _fmt_row(row: Dict[str, Any], keys_priority: List[str]) -> str:
    """Compact, stable one-liner from a row, honoring key priority."""
    parts = []
    for k in keys_priority:
        if k not in row:
            continue
        v = row.get(k)
        if v in (None, "", [], {}):
            continue
        parts.append(f"{k}={v}")
    return (
        ", ".join(parts) if parts else ", ".join(f"{k}={row[k]}" for k in list(row)[:6])
    )


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s


def _split_paragraphs(notes: str) -> List[str]:
    if not notes:
        return []
    # split on blank lines or bullet separators
    chunks = re.split(r"(?:\n\s*\n|•|-{2,})", notes)
    out = []
    for c in chunks:
        c = _clean_text(c)
        if len(c) >= 3:
            out.append(c)
    return out


def _score_para(p: str) -> float:
    """Lightweight score: keep mid-length, penalize very short/long."""
    L = len(p)
    if L <= 30:
        return 0.1 * L
    if L >= 600:
        return 600 / (L + 1)
    # favor 120–300 chars
    center = 210
    return 1.0 / (1.0 + abs(L - center) / 210)


# -------------------------------
# Snippet packers
# -------------------------------


def pack_sov_snippets(
    sov_rows: List[Dict[str, Any]],
    keys_priority: Optional[List[str]] = None,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    Returns a list of snippet dicts like:
      {"source":"sov","locator":"sheet=1,row=3","snippet":"...", "source_anchor":{"type":"sov","sheet":1,"row":3,"location_id":"WH-A"}}
    """
    keys_priority = keys_priority or [
        "location_id",
        "address",
        "occupancy",
        "construction",
        "sprinklered",
        "fire_alarm",
        "flood_zone",
        "wildfire_risk",
        "earthquake_exposure",
        "roof_age_years",
        "number_of_stories",
        "tiv_building",
        "tiv_contents",
        "total_tiv",
    ]
    out = []
    for i, r in enumerate(sov_rows, start=1):
        snip = _fmt_row(r, keys_priority)
        out.append(
            {
                "source": "sov",
                "locator": f"sheet=1,row={i}",
                "snippet": snip,
                "source_anchor": {
                    "type": "sov",
                    "sheet": 1,
                    "row": i,
                    "location_id": r.get("location_id"),
                    "address": r.get("address"),
                },
            }
        )
        if len(out) >= top_k:
            break
    return out


def pack_loss_snippets(
    loss_rows: List[Dict[str, Any]],
    keys_priority: Optional[List[str]] = None,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    keys_priority = keys_priority or [
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
    for i, r in enumerate(loss_rows, start=1):
        snip = _fmt_row(r, keys_priority)
        out.append(
            {
                "source": "loss_run",
                "locator": f"row={i}",
                "snippet": snip,
                "source_anchor": {
                    "type": "loss_run",
                    "row": i,
                    "location_id": r.get("location_id"),
                },
            }
        )
        if len(out) >= top_k:
            break
    return out


def pack_notes_snippets(
    notes_text: Optional[str],
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Split broker/insured notes into compact paragraphs and return top-k.
    Returns items like:
      {"source":"notes","locator":"para=2","snippet":"…", "source_anchor":{"type":"notes","para":2}}
    """
    if not isinstance(notes_text, str) or not notes_text.strip():
        return []
    paras = _split_paragraphs(notes_text)
    scored = [(p, _score_para(p), idx + 1) for idx, p in enumerate(paras)]
    scored.sort(key=lambda t: t[1], reverse=True)
    out = []
    for p, _s, idx in scored[:top_k]:
        out.append(
            {
                "source": "notes",
                "locator": f"para={idx}",
                "snippet": p,
                "source_anchor": {"type": "notes", "para": idx},
            }
        )
    return out


# -------------------------------
# Attachment to risk items
# -------------------------------
def attach_topk_evidence_to_items(
    risk_items: List[Dict[str, Any]],
    sov_rows: List[Dict[str, Any]],
    loss_rows: List[Dict[str, Any]],
    notes_text: Optional[str] = None,
    per_item_k: int = 2,
) -> List[Dict[str, Any]]:
    sov_snips = pack_sov_snippets(sov_rows, top_k=10)
    loss_snips = pack_loss_snippets(loss_rows, top_k=10)
    notes_snips = pack_notes_snippets(notes_text, top_k=6)

    # Build quick loc->snip maps (unchanged)
    def _index_by_loc(snips):
        idx = {}
        for s in snips:
            a = s.get("source_anchor") or {}
            loc = a.get("location_id")
            if not loc:
                m = re.search(r"\blocation_id=([A-Za-z0-9\-\_]+)", s.get("snippet", ""))
                if m:
                    loc = m.group(1)
            if loc:
                idx[str(loc)] = s
        return idx

    sov_idx = _index_by_loc(sov_snips)
    loss_idx = _index_by_loc(loss_snips)

    def _loc_from_title(title: str) -> Optional[str]:
        if not title:
            return None
        return (
            title.split(" at ", 1)[1].split(" (")[0].strip()
            if " at " in title
            else None
        )

    enriched = []
    for it in risk_items:
        ev = it.get("evidence") or []

        # Normalize roles: default any missing to 'primary'
        for e in ev:
            e.setdefault("role", "primary")

        # Build a set of existing (source, locator) to dedupe
        have = {
            (e.get("source"), e.get("locator"))
            for e in ev
            if e.get("source") and e.get("locator")
        }

        # Compute how many extra snippets we want to add
        # (we don't count existing primary/context; per_item_k is the MIN we want in total)
        need = max(0, per_item_k - len(ev))
        if need <= 0:
            enriched.append(it)
            continue

        loc = _loc_from_title(it.get("title", ""))

        # Preferred order: location-matched sov, loss; then the rest
        ordered_sources = []
        if loc and loc in sov_idx:
            ordered_sources.append(sov_idx[loc])
        if loc and loc in loss_idx:
            ordered_sources.append(loss_idx[loc])
        ordered_sources += sov_snips + loss_snips + notes_snips

        for s in ordered_sources:
            if need <= 0:
                break
            key = (s["source"], s["locator"])
            if key in have:
                continue  # don't add context duplicate of existing primary
            ev.append(
                {
                    "source": s["source"],
                    "locator": s["locator"],
                    "snippet": s["snippet"],
                    "source_anchor": s.get("source_anchor", {}),
                    "role": "context",  # <-- mark as context
                }
            )
            have.add(key)
            need -= 1

        it["evidence"] = ev
        enriched.append(it)

    return enriched


# -------------------------------
# LLM context builder (for later)
# -------------------------------


def build_llm_context(
    sov_snips: List[Dict[str, Any]],
    loss_snips: List[Dict[str, Any]],
    notes_snips: List[Dict[str, Any]],
    max_per_section: int = 6,
    max_snip_len: Optional[int] = 220,  # NEW: default keeps it backward-compatible
) -> Dict[str, Any]:
    """
    Compact, ready-to-serialize context for prompts:
    {"sov":[{locator,snippet}], "loss":[...], "notes":[...]}

    If max_snip_len is set, snippets are trimmed to that many characters to
    reduce LLM truncation risk (the miner validator uses tolerant matching).
    """

    def _trim(sn: Any) -> str:
        s = "" if sn is None else str(sn)
        if isinstance(max_snip_len, int) and max_snip_len > 0 and len(s) > max_snip_len:
            return s[:max_snip_len]
        return s

    def _shrink(lst: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for x in lst[:max_per_section]:
            loc = x.get("locator", "")
            sn = _trim(x.get("snippet", ""))
            out.append({"locator": loc, "snippet": sn})
        return out

    return {
        "sov": _shrink(sov_snips),
        "loss": _shrink(loss_snips),
        "notes": _shrink(notes_snips),
    }
