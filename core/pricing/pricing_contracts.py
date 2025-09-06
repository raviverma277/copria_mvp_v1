# pricing_contracts.py
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class SubmissionPricingInput:
    submission_id: str
    tiv: float                    # Total Insured Value
    cope_score: float             # 0–100 (from your pipeline)
    risk_count: int               # mined + rules-based
    loss_ratio: float             # e.g., 0.35 for 35%, from loss runs
    # optional extra signals you already compute:
    occupancy: Optional[str] = None
    construction: Optional[str] = None
    protection: Optional[str] = None
    exposure: Optional[str] = None

@dataclass
class PercentileResult:
    metric: str
    value: float
    pctl: float       # 0–100
    median: float
    iqr: float

@dataclass
class PricingRange:
    premium_min: float
    premium_median: float
    premium_max: float
    currency: str = "INR"

@dataclass
class PricingResult:
    submission_id: str
    percentiles: List[PercentileResult]
    pricing_range: PricingRange
    confidence: float             # 0–1
    confidence_label: str         # "Low/Med/High"
    reason_codes: List[str]       # explanations & adjustments
    llm_explainer: Optional[str] = None
