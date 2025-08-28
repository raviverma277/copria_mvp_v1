from core.risk.rules import rules_from_bundle

def test_rules_detect_unsprinklered():
    b = {"sov":[{"sheets":[{"rows":[{"location_id":"WH-B","sprinklered":False}]}]}]}
    items = rules_from_bundle(b)
    codes = {i.code for i in items}
    assert "SPRINKLER_ABSENT" in codes
