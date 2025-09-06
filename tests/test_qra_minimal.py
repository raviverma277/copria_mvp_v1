from core.extraction.pipeline import quick_risks_from_bundle, merge_quick_into_full

def test_quick_no_sprinklers():
    b = {"locations":[{"location_id":"X","sprinklers":False}]}
    qs = quick_risks_from_bundle(b)
    assert any(r["code"]=="FIRE-NO-SPRINKLER" for r in qs)
    assert all("uid" in r for r in qs)

def test_merge_dedup_and_tag():
    q=[{"uid":"u1","code":"C","title":"T","locations":["X"],"tags":["quick"]}]
    f=[{"uid":"u1","code":"C","title":"T","locations":["X"],"tags":[],"rationale":"full"}]
    m = merge_quick_into_full(f,q)
    out = [r for r in m if r["uid"]=="u1"][0]
    assert "quick" in out["tags"] and out["rationale"]=="full"
