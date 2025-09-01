# core/risk/miner.py
from __future__ import annotations
from core.utils.llm_status import record_llm_call_start, record_llm_call_end
from typing import List, Dict, Any, Set, Optional
import os, json, re, hashlib
from openai import OpenAI


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _snippet_matches(snippet: str, allowed_snippets: set[str]) -> bool:
    """
    Accept if snippet matches exactly OR is a prefix/suffix of an allowed one
    after whitespace normalization. This handles minor truncation or line breaks.
    """
    ns = _norm_ws(snippet)
    if not ns:
        return False
    for a in allowed_snippets:
        na = _norm_ws(a)
        if ns == na or ns.startswith(na) or na.startswith(ns):
            return True
    return False


def _safe_get_text(resp) -> str:
    """
    Try multiple locations to retrieve the model text, depending on SDK version.
    """
    # New Responses API shortcut
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    # Try 'output' list shape (some SDK builds)
    out = getattr(resp, "output", None)
    if isinstance(out, list):
        chunks = []
        for o in out:
            # some entries are dict-like with 'content' -> [{'type':'output_text','text':...}]
            content = o.get("content") if isinstance(o, dict) else None
            if isinstance(content, list):
                for c in content:
                    t = c.get("text") or c.get("content")
                    if isinstance(t, str):
                        chunks.append(t)
        if chunks:
            return "\n".join(chunks)

    # Try Chat Completions-like shape (fallback)
    choices = getattr(resp, "choices", None)
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if msg and isinstance(msg.get("content"), str):
            return msg["content"]

    # Final fallback: string-ify
    return str(resp)


def _extract_json_array(s: str) -> Optional[str]:
    """
    Extract the first JSON array substring from text.
    Handles ```json fences, plain arrays, and partial/truncated endings best-effort.
    Returns the JSON array string, or None if not found.
    """
    if not isinstance(s, str) or not s.strip():
        return None

    # 1) Strip fences first if present
    if "```" in s:
        # try ```json ... ```
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
        if m:
            s = m.group(1).strip()

    # 2) Find from the first '[' to the last ']' present
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1].strip()
        # quick sanity: must start with '['
        if candidate.startswith("["):
            return candidate

    # 3) Balanced scan fallback (in case there are extra brackets before the array)
    start = s.find("[")
    if start != -1:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "[":
                depth += 1
            elif s[i] == "]":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1].strip()

    return None


SYSTEM = (
    "You are a commercial property underwriting aide. "
    "Your job is to propose a few NEW, short, well-grounded risk items based ONLY on the provided snippets. "
    "Never invent facts; only infer if strongly supported by the text. "
    "Each item MUST cite exactly one provided snippet (copy snippet text EXACTLY). "
    "Return ONLY a bare JSON array with up to 3 items; no prose, no code fences."
)

# Minimal schema guard
_ALLOWED_SEVERITIES = {"low", "medium", "high", "critical"}


def _valid_item(x: dict, allowed_snippets: set[str]) -> bool:
    if not isinstance(x, dict):
        return False
    # required scalars
    code = x.get("code")
    title = x.get("title")
    sev = x.get("severity")
    if not isinstance(code, str) or not code.strip():
        return False
    if not isinstance(title, str) or not title.strip():
        return False
    if not isinstance(sev, str) or sev.lower() not in _ALLOWED_SEVERITIES:
        return False

    # evidence: allow dict or list; coerce to list here
    ev = x.get("evidence")
    if isinstance(ev, dict):
        ev = [ev]
        x["evidence"] = ev  # mutate in place so caller sees normalized shape
    if not isinstance(ev, list) or not ev:
        return False

    # at least one evidence must cite an allowed snippet
    cited_ok = False
    for e in ev:
        if not isinstance(e, dict):
            continue
        snip = e.get("snippet")
        if isinstance(snip, str) and _snippet_matches(snip, allowed_snippets):
            cited_ok = True
            break
    return cited_ok


def _collect_allowed_snippets(context: Dict[str, Any]) -> Set[str]:
    allowed = set()
    for sec in ("sov", "loss", "notes"):
        for d in context.get(sec, []) or []:
            snip = d.get("snippet")
            if isinstance(snip, str) and snip:
                allowed.add(snip)
    return allowed


def _uid_from_item(d: dict) -> str:
    """
    Compute a stable uid from (code, title, evidence anchors) to match RiskItem's logic.
    """
    anchors = []
    for e in d.get("evidence") or []:
        src = e.get("source")
        loc = e.get("locator")
        if src and loc:
            anchors.append((src, loc))
    raw = {"code": d.get("code"), "title": d.get("title"), "anchors": anchors}
    s = json.dumps(raw, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def mine_additional_risks(
    context: Dict[str, Any],
    existing_codes: Set[str],
    existing_titles: Set[str],
    model: str = os.environ.get("LLM_MINER_MODEL", "gpt-4o-mini"),
    temperature: float = 0.2,
    max_items: int = 6,
) -> List[Dict[str, Any]]:
    """
    context: {"sov":[{locator,snippet}], "loss":[...], "notes":[...]}
    Returns a list[dict] of risk items with fields:
      code, title, severity, rationale, evidence=[{source, locator, snippet}]
    """
    client = OpenAI()

    # compose prompt input
    prompt = {
        "snippets": context,
        "avoid_codes": sorted(list(existing_codes))[:40],
        "avoid_titles": list(existing_titles)[:40],
        "instructions": (
            "Propose NEW risks not already covered by avoid_codes or avoid_titles. "
            "Each risk must be grounded in the snippets and cite at least one snippet (copy exactly). "
            "Keep rationale <= 200 chars."
        ),
        "output_schema": {
            "code": "string, short identifier like LLM_FALL_HAZARD",
            "title": "string, human readable",
            "severity": "one of: low|medium|high|critical",
            "rationale": "string, <=200 chars",
            "evidence": "list of {source:'sov|loss_run|notes', locator:'...', snippet:'<exact from provided>'}",
        },
    }

    # --- Call the LLM with robust fallback (no json_schema in this SDK) ------
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[LLM-MINER] OPENAI_API_KEY not set; skipping miner.")
        return []

    client = OpenAI()  # uses env

    # Keep the context but tighten instructions to force minimal JSON
    prompt["instructions"] = (
        "Propose up to 3 NEW risks not already covered by avoid_codes or avoid_titles. "
        "Return a BARE JSON array ONLY (no prose, no code fences). "
        "Each item MUST cite exactly one of the provided snippets (copy snippet text EXACTLY). "
        "Use very short titles and rationale (<=120 chars). One evidence per item."
    )

    user_payload = json.dumps(prompt, ensure_ascii=False)
    record_llm_call_start(model)

    try:
        print("[LLM-MINER] Calling model (plain JSON)…")
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_payload},
            ],
            temperature=temperature,
            # Give it more room so the array isn't truncated mid-JSON
            max_output_tokens=900,
        )
        record_llm_call_end(True)
    except Exception as e:
        record_llm_call_end(False)
        print("[LLM-MINER] LLM call failed:", repr(e))
        return []

    text = _safe_get_text(resp).strip()
    print("[LLM-MINER] Raw text (first 400 chars):", text[:400])

    # Try to extract a JSON array out of prose/fences/truncation
    json_str = _extract_json_array(text)
    if not json_str:
        print("[LLM-MINER] No JSON array found; aborting miner.")
        return []

    try:
        arr = json.loads(json_str)
        print("[LLM-MINER] Parsed items:", len(arr))
        if not isinstance(arr, list):
            return []
    except Exception as e:
        print(
            "[LLM-MINER] Error parsing JSON array:", e, "\nPayload was:", json_str[:400]
        )
        return []

    # --- normalize common model quirks before validation ---
    for x in arr:
        # severity to lowercase
        if isinstance(x.get("severity"), str):
            x["severity"] = x["severity"].lower()

        # wrap evidence dict -> list
        if isinstance(x.get("evidence"), dict):
            x["evidence"] = [x["evidence"]]

        # ensure each evidence has a source (infer from locator if missing)
        for e in x.get("evidence") or []:
            if "source" not in e or not e["source"]:
                loc = str(e.get("locator", ""))
                if loc.startswith("row="):
                    e["source"] = "loss_run"
                elif loc.startswith("para="):
                    e["source"] = "notes"
                else:
                    e["source"] = "sov"

    # validate & filter
    allowed_snips = _collect_allowed_snippets(context)
    out: List[Dict[str, Any]] = []
    dropped = 0
    for x in arr:
        if not _valid_item(x, allowed_snips):
            dropped += 1
            continue
        if x["code"] in existing_codes or x["title"] in existing_titles:
            dropped += 1
            continue

        # normalize evidence shape & role
        for e in x.get("evidence", []):
            # default source when omitted: try to infer from locator, else 'sov'
            src = e.get("source")
            if not src:
                loc = str(e.get("locator", ""))
                if loc.startswith("row="):
                    src = "loss_run"
                elif loc.startswith("para="):
                    src = "notes"
                else:
                    src = "sov"
                e["source"] = src
            e.setdefault(
                "role", "primary"
            )  # mined item’s first evidence can be primary

        # tag mined origin and cap text lengths a bit
        x.setdefault("tags", []).append("llm-mined")
        if isinstance(x.get("rationale"), str):
            x["rationale"] = x["rationale"][:220]

        out.append(x)
        if len(out) >= max_items:
            break
    # ensure every mined item has a stable uid (code + title + evidence anchors)
    for it in out:
        if not it.get("uid"):
            anchors = []
            for e in it.get("evidence") or []:
                src = e.get("source")
                loc = e.get("locator")
                if src and loc:
                    anchors.append((src, loc))
            raw = {"code": it.get("code"), "title": it.get("title"), "anchors": anchors}
            s = json.dumps(raw, sort_keys=True, ensure_ascii=False)
            it["uid"] = hashlib.md5(s.encode("utf-8")).hexdigest()

    print(f"[LLM-MINER] Kept {len(out)} item(s), dropped {dropped}.")
    return out
