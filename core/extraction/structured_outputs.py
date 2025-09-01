# Placeholder for OpenAI Structured Outputs integration.
# Implement actual OpenAI calls here.
from typing import Dict, Any
from ..schemas.registry import load_active_schema


def extract_with_schema(parsed_bundle: Dict[str, Any], schema_name: str):
    schema = load_active_schema(schema_name)
    # TODO: Select relevant parsed chunks by doc_type and call OpenAI with response_format=json_schema
    # Return mock for now
    if schema_name == "submission":
        return {
            "insured_name": "ACME LTD",
            "currency": "GBP",
            "effective_date": "2025-10-01",
        }
    if schema_name == "sov":
        return [
            {
                "address": "1 High St",
                "country": "GB",
                "tiv_building": 5000000,
                "sprinklers": False,
            }
        ]
    if schema_name == "loss_run":
        return [{"date_of_loss": "2024-05-12", "cause": "Fire", "gross_paid": 900000}]
    return None
