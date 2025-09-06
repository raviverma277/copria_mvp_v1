# pricing/strategy_factory.py
from core.pricing.pricing_engine import RulesPricer
# from pricing.external_api_pricer import ExternalAPIPricer  # later

def get_pricer(name: str = "rules"):
    if name == "rules":
        return RulesPricer()
    # elif name == "external":
    #     return ExternalAPIPricer(...)
    raise ValueError(f"Unknown pricing strategy: {name}")
