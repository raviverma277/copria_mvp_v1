# core/risk/redflags.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable
import json, os
from .models import RiskItem, EvidenceRef

# Where to read rules from
DEFAULT_RULES_PATH = os.environ.get("RED_FLAG_RULES_PATH", "data/red_flag_rules.json")

# Map human "Field" labels in the JSON to normalized SOV keys
FIELD_MAP = {
    "Sprinkler System (Y/N)": "sprinklered",  # bool
    "Fire Alarm (Y/N)": "fire_alarm",  # bool
    "Flood Zone (e.g., Zone X, AE)": "flood_zone",  # str
    "Wildfire Risk (Low/Moderate/High or ISO Class)": "wildfire_risk",  # str
    "Earthquake Exposure (Low/Moderate/High or ShakeMap Zone)": "earthquake_exposure",  # str
    "Roof > 20 yrs": "roof_age_years",  # numeric
    "Number of Stories": "number_of_stories",  # numeric
    "Hazardous Materials (Y/N)": "hazardous_materials",  # bool
    # "Prior Claims (Y/N) and Total Loss Amount" handled specially below
    # "Total TIV and Sprinkler System" handled specially below
    "Total TIV": "total_tiv",  # numeric (if your normalizer fills this)
}


def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _contains_any(s: str, needles: List[str]) -> bool:
    s = (s or "").lower()
    return any(n.lower() in s for n in needles)


def _load_rules(path: str = DEFAULT_RULES_PATH) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate_loss(loss_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate prior claims and totals per location_id."""
    agg: Dict[str, Dict[str, Any]] = {}
    for r in loss_rows or []:
        loc = str(r.get("location_id") or "unknown")
        a = agg.setdefault(loc, {"count": 0, "total_incurred": 0.0})
        a["count"] += 1
        try:
            a["total_incurred"] += float(r.get("incurred") or 0)
        except Exception:
            pass
    return agg


def _mk_item(
    code: str,
    title: str,
    severity: str,
    rationale: str,
    source: str,
    locator: str,
    snippet: str,
    tags: List[str],
) -> RiskItem:
    return RiskItem(
        code=code,
        title=title,
        severity=severity,
        rationale=rationale,
        evidence=[
            EvidenceRef(source=source, locator=locator, snippet=snippet, role="primary")
        ],
        tags=tags,
        rule_hits=[code],
        confidence=0.7,
    )


def evaluate_red_flags(
    rules: List[Dict[str, Any]],
    sov_rows: List[Dict[str, Any]],
    loss_rows: List[Dict[str, Any]],
    loc_to_addr: Dict[str, str] | None = None,
) -> List[RiskItem]:
    """Evaluate JSON rules against normalized rows."""
    loc_to_addr = loc_to_addr or {}
    out: List[RiskItem] = []
    loss_agg = _aggregate_loss(loss_rows)

    for i, row in enumerate(sov_rows, start=1):
        loc = str(row.get("location_id") or f"row{i}")
        addr = loc_to_addr.get(loc)
        at_where = f" at {loc}" + (f" ({addr})" if addr else "")

        for r in rules:
            field = r.get("field")
            cond = (r.get("condition") or "").strip()
            desc = r.get("description") or "Red flag"
            category = r.get("category") or "General"

            # Skip sprinkler JSON rule; baseline "SPRINKLER_ABSENT" already covers it
            if field == "Sprinkler System (Y/N)":
                continue

            # Special: Roof > 20 yrs (JSON uses == 'Yes', but our data is numeric)
            if field.startswith("Roof > 20"):
                age = row.get("roof_age_years")
                try:
                    age_val = float(age) if age is not None else None
                except Exception:
                    age_val = None
                if age_val is not None and age_val > 20:
                    out.append(
                        _mk_item(
                            code="RF_ROOF_AGE",
                            title=f"Roof older than 20 years{at_where}",
                            severity="medium",
                            rationale="Roof age exceeds 20 years.",
                            source="sov",
                            locator=f"sheet=1,row={i}",
                            snippet=str({"roof_age_years": age}),
                            tags=[category, "roof"],
                        )
                    )
                continue

            # Special: Prior Claims AND Total Loss
            if field.startswith("Prior Claims"):
                agg = loss_agg.get(loc, {"count": 0, "total_incurred": 0.0})
                prior = agg["count"] > 0
                total = agg["total_incurred"]
                # Condition in file: "Prior Claims == 'Yes' and Total Loss Amount > 100000"
                trigger = prior and (total > 100000)
                if trigger:
                    out.append(
                        _mk_item(
                            code="RF_PRIOR_CLAIMS",
                            title=f"Prior significant claims{at_where}",
                            severity="medium",
                            rationale=f"Location shows prior claims totalling ${int(total):,}.",
                            source="loss_run",
                            locator=f"location={loc}",
                            snippet=f"claims={agg['count']} total_incurred={total}",
                            tags=[category, "loss_history"],
                        )
                    )
                continue

            # Special: TIV > X AND Sprinkler == No
            if field.startswith("Total TIV") and "Sprinkler" in field:
                # We expect either a pre-computed total_tiv, or compute from building/contents
                tiv = row.get("total_tiv")
                if tiv is None:
                    try:
                        tiv = float(row.get("tiv_building") or 0) + float(
                            row.get("tiv_contents") or 0
                        )
                    except Exception:
                        tiv = 0.0
                sprinkler = _coerce_bool(row.get("sprinklered"))
                trigger = (tiv > 10_000_000) and (sprinkler is False)
                if trigger:
                    out.append(
                        _mk_item(
                            code="RF_HIGH_TIV_NO_SPRINKLER",
                            title=f"High value without sprinklers{at_where}",
                            severity="high",
                            rationale=f"Total TIV ≈ ${int(tiv):,} and sprinklers absent.",
                            source="sov",
                            locator=f"sheet=1,row={i}",
                            snippet=str(
                                {
                                    k: row.get(k)
                                    for k in [
                                        "total_tiv",
                                        "tiv_building",
                                        "tiv_contents",
                                        "sprinklered",
                                    ]
                                }
                            ),
                            tags=[category, "valuation", "fire_protection"],
                        )
                    )
                continue

            # Generic field mapping
            key = FIELD_MAP.get(field)
            if not key:
                continue  # field not mapped yet

            val = row.get(key)

            # Condition patterns from your JSON
            if cond.startswith("=="):
                # e.g., == 'No' or == 'High'
                needle = cond.split("==", 1)[1].strip().strip("'\"")
                if isinstance(val, bool):
                    matches = (val is True and needle.lower() == "yes") or (
                        val is False and needle.lower() == "no"
                    )
                else:
                    matches = str(val).lower() == needle.lower()
            elif cond.startswith("contains"):
                # e.g., contains 'AE' or 'VE' or 'A'
                inside = [
                    s.strip().strip("'\"")
                    for s in cond.replace("contains", "", 1).split("or")
                ]
                matches = _contains_any(str(val), inside)
            elif cond.startswith(">"):
                try:
                    th = float(cond[1:].strip())
                    matches = float(val or 0) > th
                except Exception:
                    matches = False
            elif " and " in cond:
                # Already handled special "Prior Claims and Total Loss" above
                matches = False
            else:
                matches = False

            if matches:
                code = "RF_" + key.upper()
                sev = "high" if "sprinkler" in key or "tiv" in key else "medium"
                out.append(
                    _mk_item(
                        code=code,
                        title=f"{desc}{at_where}",
                        severity=sev,
                        rationale=desc,
                        source="sov",
                        locator=f"sheet=1,row={i}",
                        snippet=str({key: val}),
                        tags=[category],
                    )
                )

    return out
