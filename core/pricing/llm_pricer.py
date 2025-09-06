# llm_pricer.py
from typing import Optional
from core.pricing.pricing_contracts import SubmissionPricingInput, PricingResult, PricingRange
from core.pricing.pricing_benchmark import PriceBenchmark

# IMPORTANT: All amounts are in USD for both the LLM prompt and the returned PricingRange.
LLM_PRICING_PROMPT_USD = """You are a cautious commercial property insurance pricing assistant.

All currency values in this task are in **USD**.

Given:
- Submission metrics (USD & unitized):
  - TIV (USD): {tiv:,.0f}
  - COPE score (0–100): {cope}
  - Risk count: {risk_count}
  - Loss ratio (0–1): {loss_ratio}
- Benchmark summary (medians/IQR; USD where applicable):
  - TIV median (USD): {tiv_median:,.0f}
  - COPE median: {cope_median:.1f}
  - Risk count median: {risk_median:.1f}
  - Loss ratio median: {loss_median:.2f}
  - Premium-per-TIV median: {ppt_median:.6f}
- Current rules-based indicative premium range (USD):
  - min: {rmin:,.0f}
  - median: {rmed:,.0f}
  - max: {rmax:,.0f}

Task:
1) Propose a refined *indicative* premium range in **USD** for this submission (commercial property).
2) Briefly explain the rationale referencing COPE, loss ratio, risk count, TIV relative to the benchmark.
3) Provide a self-rated confidence in [0, 1].

Return ONLY a compact JSON object with keys:
- premium_min (number, USD)
- premium_median (number, USD)
- premium_max (number, USD)
- rationale (string)
- confidence (number in [0,1])

Do not include any other text before or after the JSON.
"""

class LLMPricerBlend:
    """
    Blends an LLM's USD pricing suggestion with the RulesPricer result.
    Expect the LLM client to expose a `complete_json(prompt: str) -> dict` method.
    """
    def __init__(self, llm_client, weight_llm: float = 0.35):
        """
        :param llm_client: your LLM wrapper with `complete_json(prompt)->dict`
        :param weight_llm: 0..1 weight given to LLM outputs in the blend
        """
        self.llm = llm_client
        self.weight_llm = max(0.0, min(1.0, float(weight_llm)))

    def _build_prompt(self, s: SubmissionPricingInput, bm: PriceBenchmark, base: PricingResult) -> str:
        bench_info = {
            "tiv_median": bm.median("tiv"),
            "cope_median": bm.median("cope_score"),
            "risk_median": bm.median("risk_count"),
            "loss_median": bm.median("loss_ratio"),
            "ppt_median": bm.median("premium_per_tiv"),
        }
        return LLM_PRICING_PROMPT_USD.format(
            tiv=s.tiv,
            cope=s.cope_score,
            risk_count=s.risk_count,
            loss_ratio=s.loss_ratio,
            tiv_median=bench_info["tiv_median"],
            cope_median=bench_info["cope_median"],
            risk_median=bench_info["risk_median"],
            loss_median=bench_info["loss_median"],
            ppt_median=bench_info["ppt_median"],
            rmin=base.pricing_range.premium_min,
            rmed=base.pricing_range.premium_median,
            rmax=base.pricing_range.premium_max,
        )

    @staticmethod
    def _clamp(value: float, floor: float, ceil: float) -> float:
        try:
            v = float(value)
        except Exception:
            return floor
        if ceil < floor:
            ceil = floor
        return max(floor, min(ceil, v))

    def blend(self, s: SubmissionPricingInput, bm: PriceBenchmark, base: PricingResult) -> PricingResult:
        """
        :param s: SubmissionPricingInput (assumed USD-based TIV)
        :param bm: PriceBenchmark
        :param base: PricingResult from the rules-based engine (will be updated)
        :return: Updated PricingResult with USD currency
        """
        # Ensure base currency is USD for downstream UI consistency
        base.pricing_range = PricingRange(
            premium_min=base.pricing_range.premium_min,
            premium_median=base.pricing_range.premium_median,
            premium_max=base.pricing_range.premium_max,
            currency="USD",
        )

        prompt = self._build_prompt(s, bm, base)

        try:
            raw = self.llm.complete_json(prompt)  # -> dict with keys described in prompt

            # Guardrails based on base range to avoid extreme jumps:
            # allow the LLM range to vary between 25% of base.min and 250% of base.max
            lo_guard = 0.25 * max(1.0, base.pricing_range.premium_min)
            hi_guard = 2.50 * max(1.0, base.pricing_range.premium_max)

            l_min = self._clamp(raw.get("premium_min", base.pricing_range.premium_min), lo_guard, hi_guard)
            l_med = self._clamp(raw.get("premium_median", base.pricing_range.premium_median), lo_guard, hi_guard)
            l_max = self._clamp(raw.get("premium_max", base.pricing_range.premium_max), lo_guard, hi_guard)

            # Ensure ordering min <= median <= max
            l_med = max(min(l_med, l_max), l_min)
            l_min = min(l_min, l_med)
            l_max = max(l_max, l_med)

            rationale = str(raw.get("rationale", "")).strip()
            try:
                l_conf = float(raw.get("confidence", base.confidence))
            except Exception:
                l_conf = base.confidence
            l_conf = max(0.0, min(1.0, l_conf))

            # Blend with base using weight_llm
            w = self.weight_llm
            bw = 1.0 - w

            new_min = bw * base.pricing_range.premium_min + w * l_min
            new_med = bw * base.pricing_range.premium_median + w * l_med
            new_max = bw * base.pricing_range.premium_max + w * l_max

            # Keep ordering after blend
            new_med = max(min(new_med, new_max), new_min)
            new_min = min(new_min, new_med)
            new_max = max(new_max, new_med)

            # Confidence blend (capped)
            new_conf = min(0.98, bw * base.confidence + w * l_conf)
            new_label = "High" if new_conf >= 0.75 else "Medium" if new_conf >= 0.5 else "Low"

            # Update base result in-place
            base.pricing_range = PricingRange(new_min, new_med, new_max, currency="USD")
            base.confidence = new_conf
            base.confidence_label = new_label
            base.llm_explainer = rationale or "LLM refinement applied (USD)."

            # Reason code breadcrumb
            if "LLM refinement applied" not in base.reason_codes:
                base.reason_codes.append("LLM refinement applied (USD)")
            return base

        except Exception:
            # On any failure, keep the rules-based USD result and annotate reason
            if "LLM refinement unavailable; using rules-based pricing" not in base.reason_codes:
                base.reason_codes.append("LLM refinement unavailable; using rules-based pricing (USD)")
            base.pricing_range = PricingRange(
                premium_min=base.pricing_range.premium_min,
                premium_median=base.pricing_range.premium_median,
                premium_max=base.pricing_range.premium_max,
                currency="USD",
            )
            return base
