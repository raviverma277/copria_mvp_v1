from typing import Dict, Any, Tuple
from .cope_rules import compute_cope_score
from .nuanced import generate_nuanced_concerns


def run_risk_pipeline(
    submission_bundle: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cope = compute_cope_score(submission_bundle)
    concerns = generate_nuanced_concerns(submission_bundle, cope)
    return cope, concerns
