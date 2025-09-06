from core.pricing.pricing_contracts import SubmissionPricingInput
from core.pricing.pricing_benchmark import PriceBenchmark
from core.pricing.pricing_engine import RulesPricer

def test_rules_pricer_runs():
    bm = PriceBenchmark("data/benchmarks/pricing_benchmark.csv")
    s = SubmissionPricingInput(
        submission_id="S1",
        tiv=3_000_000, cope_score=75, risk_count=7, loss_ratio=0.18
    )
    res = RulesPricer().price(s, bm)
    assert res.pricing_range.premium_min > 0
    assert res.pricing_range.premium_max >= res.pricing_range.premium_median
