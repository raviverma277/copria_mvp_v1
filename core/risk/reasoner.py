from typing import List, Dict, Any
from .models import RiskItem
from openai import OpenAI

SYSTEM = (
    "You are a commercial property underwriting aide. "
    "You must only return concise justifications grounded in supplied evidence. "
    "No speculation, no new facts. 1-2 sentences per risk."
)


def justify_with_llm(items: List[RiskItem]) -> List[RiskItem]:
    if not items:
        return items
    client = OpenAI()  # uses env OPENAI_API_KEY
    # Batch a small prompt; keep it tiny to stay fast
    for it in items:
        # Compose a tiny evidence string
        ev = "; ".join(
            [f"{e.source}@{e.locator}: {e.snippet}" for e in it.evidence if e.snippet][
                :3
            ]
        )
        prompt = (
            f"Risk: {it.title}\n"
            f"Evidence: {ev}\n"
            f"Explain briefly why this increases risk. Keep to 1–2 sentences. "
            f"Do not invent data. Return only the explanation."
        )
        try:
            resp = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_output_tokens=90,
            )
            it.llm_notes = resp.output_text.strip()
        except Exception:
            pass
    return items


def dedupe_and_cap(items: List[RiskItem], cap: int = 12) -> List[RiskItem]:
    seen = set()
    out = []
    for it in items:
        key = (it.code, it.title)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= cap:
            break
    return out
