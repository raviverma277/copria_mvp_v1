# core/parsing/excel_parser.py
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import re

def _looks_like_real_headers(cols: List[str]) -> bool:
    """
    Heuristic: real headers tend to be short strings with letters/spaces,
    not raw values like dates/numbers. Return True if most look like headers.
    """
    if not cols:
        return False
    good = 0
    for c in cols:
        cstr = str(c).strip()
        # Has letters or spaces and not mostly digits/punctuation
        if re.search(r"[A-Za-z]", cstr) and not re.fullmatch(r"[\d\W_]+", cstr):
            good += 1
    return good >= max(2, int(0.6 * len(cols)))

def parse_excel(file) -> Dict[str, Any]:
    """
    Return a uniform structure:
      {
        "type": "excel",
        "sheets": [
          {"name": str, "headers": [str...], "rows": [ {header: value, ...}, ... ]},
          ...
        ]
      }
    This trusts pandas' header row. It ONLY falls back to using the first row
    as headers if pandas' columns don't look like headers.
    """
    xl = pd.ExcelFile(file)
    sheets = []
    for name in xl.sheet_names:
        # First, trust pandas to read with the first row as header
        df = xl.parse(name)  # header=0 by default
        # Strip BOM/whitespace, fill NAs
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
        df = df.fillna("")

        # If pandas columns don't look like headers (rare), try using first row as header
        if not _looks_like_real_headers(list(df.columns)) and len(df.index) > 0:
            alt = xl.parse(name, header=None).fillna("")
            # Use row 0 as header if those cells look header-y
            first_row = [str(x).strip() for x in list(alt.iloc[0].values)]
            if _looks_like_real_headers(first_row):
                alt.columns = first_row
                alt = alt.iloc[1:].reset_index(drop=True)
                df = alt

        headers = [str(h) for h in df.columns]
        rows = df.to_dict(orient="records")
        sheets.append({"name": name, "headers": headers, "rows": rows})

    return {"type": "excel", "sheets": sheets}
