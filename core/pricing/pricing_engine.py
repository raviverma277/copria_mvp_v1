# pricing_engine.py
from typing import List, Protocol, Tuple
from core.pricing.pricing_contracts import SubmissionPricingInput, PricingResult, PricingRange, PercentileResult
from core.pricing.pricing_benchmark import PriceBenchmark

class Pricer(Protocol):
    def price(self, s: SubmissionPricingInput, bm: PriceBenchmark) -> PricingResult: ...

def _confidence_from_spread(similar_iqr: float, ppt_iqr: float) -> float:
    # Map smaller IQR to higher confidence; clamp 0.2–0.9 for sanity
    if ppt_iqr <= 0: return 0.5
    raw = 1.0 / (1.0 + (ppt_iqr * 8.0))
    return max(0.2, min(0.9, raw))

class RulesPricer:
    """
    Simple, transparent rules you can later replace with actuarial/external models.
    """
    def __init__(self, base_rate_per_tiv: float = 0.0055,
                 w_cope: float = -0.15, w_risk: float = 0.08, w_loss: float = 0.60):
        self.base_rate = base_rate_per_tiv
        self.w_cope = w_cope
        self.w_risk = w_risk
        self.w_loss = w_loss

    def _adj(self, s: SubmissionPricingInput, bm: PriceBenchmark) -> Tuple[float, List[str]]:
        rc: List[str] = []
        # Normalize factors around benchmark medians
        cope_m = bm.median("cope_score") or 70.0
        risk_m = bm.median("risk_count") or 8.0
        loss_m = bm.median("loss_ratio") or 0.2

        cope_adj = self.w_cope * ((s.cope_score - cope_m) / 20.0)   # ± ~0.15 for ±20 pts
        risk_adj = self.w_risk * ((s.risk_count - risk_m) / 5.0)    # ~0.08 per 5 risks
        loss_adj = self.w_loss * ((s.loss_ratio - loss_m) / 0.20)   # ~0.60 per +20% loss

        total = 1.0 + cope_adj + risk_adj + loss_adj
        # guardrails
        total = max(0.5, min(1.8, total))

        # Reasons:
        if cope_adj < -0.02: rc.append("COPE above median → discount")
        if cope_adj >  0.02: rc.append("COPE below median → surcharge")
        if risk_adj >  0.02: rc.append("High risk count → surcharge")
        if risk_adj < -0.02: rc.append("Low risk count → discount")
        if loss_adj >  0.02: rc.append("Elevated loss ratio → surcharge")
        if loss_adj < -0.02: rc.append("Benign loss history → discount")

        return total, rc

    def price(self, s: SubmissionPricingInput, bm: PriceBenchmark) -> PricingResult:
        pctl = bm.summarize(s)
        ppt_iqr = bm.iqr("premium_per_tiv")
        adj_factor, reason_codes = self._adj(s, bm)

        # Baseline (median benchmark rate) blended with configured base_rate
        bench_med_ppt = max(0.003, bm.median("premium_per_tiv"))
        base_ppt = (0.5 * bench_med_ppt) + (0.5 * self.base_rate)
        ppt = base_ppt * adj_factor

        # Turn rate into premium
        median_premium = s.tiv * ppt

        # Spread around median using uncertainty from benchmark dispersion
        # Wider IQR → wider range
        spread = max(0.1, min(0.5, 0.5 * (ppt_iqr / max(bench_med_ppt, 1e-6))))
        premium_min = median_premium * (1.0 - spread)
        premium_max = median_premium * (1.0 + spread)

        # Confidence
        conf = _confidence_from_spread(similar_iqr=bm.iqr("tiv"), ppt_iqr=ppt_iqr)
        label = "High" if conf >= 0.75 else "Medium" if conf >= 0.5 else "Low"

        return PricingResult(
            submission_id=s.submission_id,
            percentiles=pctl,
            pricing_range=PricingRange(premium_min, median_premium, premium_max, currency="INR"),
            confidence=conf,
            confidence_label=label,
            reason_codes=reason_codes,
            llm_explainer=None
        )
