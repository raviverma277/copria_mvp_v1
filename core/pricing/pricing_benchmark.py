# pricing_benchmark.py
import csv, math
from typing import Dict, List
import numpy as np
from core.pricing.pricing_contracts import SubmissionPricingInput, PercentileResult

class PriceBenchmark:
    def __init__(self, csv_path: str):
        self.rows = []
        with open(csv_path, newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                self.rows.append({
                    "tiv": float(row["tiv"]),
                    "cope_score": float(row["cope_score"]),
                    "risk_count": float(row["risk_count"]),
                    "loss_ratio": float(row["loss_ratio"]),
                    "premium_per_tiv": float(row["premium_per_tiv"]),
                })
        self.arr = {k: np.array([r[k] for r in self.rows], dtype=float)
                    for k in ["tiv","cope_score","risk_count","loss_ratio","premium_per_tiv"]}

    def percentile_of(self, metric: str, value: float) -> float:
        """Return percentile (0–100) of 'value' within benchmark distribution."""
        data = self.arr[metric]
        # Percentile rank: proportion <= value
        return float(100.0 * np.mean(data <= value))

    def median(self, metric: str) -> float:
        return float(np.median(self.arr[metric]))

    def iqr(self, metric: str) -> float:
        q75, q25 = np.percentile(self.arr[metric], [75, 25])
        return float(q75 - q25)

    def summarize(self, s: SubmissionPricingInput) -> List[PercentileResult]:
        items = [
            ("tiv", s.tiv),
            ("cope_score", s.cope_score),
            ("risk_count", float(s.risk_count)),
            ("loss_ratio", s.loss_ratio),
        ]
        out: List[PercentileResult] = []
        for metric, val in items:
            out.append(PercentileResult(
                metric=metric,
                value=val,
                pctl=self.percentile_of(metric, val),
                median=self.median(metric),
                iqr=self.iqr(metric)
            ))
        return out
