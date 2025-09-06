# scripts/gen_mock_benchmark.py
import csv, random

def gen_row():
    tiv = random.uniform(5e5, 1.5e7)                 # 0.5M–15M
    cope = random.uniform(40, 95)                    # COPE 40–95
    risk_count = int(random.gauss(8, 4))             # ~8 ± 4
    risk_count = max(0, risk_count)
    loss_ratio = max(0.0, min(1.0, random.gauss(0.20, 0.10)))  # 0–1
    # premium_per_tiv baseline drifts up with loss ratio and down with cope
    base = 0.0045 + 0.004 * loss_ratio - 0.000015 * (cope - 70)
    ppt = max(0.003, base + random.gauss(0, 0.0008))
    return [tiv, cope, risk_count, loss_ratio, ppt]

N = 500
with open("data/benchmarks/pricing_benchmark.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tiv", "cope_score", "risk_count", "loss_ratio", "premium_per_tiv"])
    for _ in range(N):
        w.writerow(gen_row())
