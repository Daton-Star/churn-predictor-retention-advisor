"""Segment analysis: parses the pre-computed SQL query results
(sql/QUERY_RESULTS.md) into DataFrames for the app's "Segment analysis" tab,
so the SQL work in sql/ shows up as part of the live product instead of
sitting in a folder next to it.

No live SQL connection here: sql/QUERY_RESULTS.md is committed, real output
from sql/run_queries.py, regenerated whenever the underlying queries change
(see sql/README.md).
"""

import re

import pandas as pd

from config import SQL_QUERY_RESULTS_PATH


def _extract_table(text, header_marker):
    """Return the markdown table text immediately following the
    `## \\`header_marker\\`` heading (after its SQL code fence)."""
    pattern = rf"## `{re.escape(header_marker)}`.*?\n\n(\|.*?)(?:\n\n|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find table for {header_marker!r} in {SQL_QUERY_RESULTS_PATH}")
    return match.group(1)


def _markdown_table_to_df(table_text):
    lines = [line.strip() for line in table_text.strip().splitlines() if line.strip()]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = [[c.strip() for c in line.strip("|").split("|")] for line in lines[2:]]  # skip header + separator
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().all():
            df[col] = converted
    return df


def load_churn_by_country(path=SQL_QUERY_RESULTS_PATH):
    text = path.read_text()
    return _markdown_table_to_df(_extract_table(text, "02_churn_rate_by_country.sql"))


def load_recency_risk_tiers(path=SQL_QUERY_RESULTS_PATH):
    text = path.read_text()
    return _markdown_table_to_df(_extract_table(text, "04_recency_risk_tiers.sql"))


if __name__ == "__main__":
    print(load_churn_by_country())
    print()
    print(load_recency_risk_tiers())
