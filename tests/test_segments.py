"""Unit tests for segments.py -- parsing sql/QUERY_RESULTS.md into DataFrames
for the app's "Segment analysis" tab. Run against the real, committed
QUERY_RESULTS.md (same rationale as test_rag_assistant.py's chunking tests):
that file is checked in specifically so downstream consumers don't need the
raw dataset or a live SQL connection.
"""

import pandas as pd

import segments


def test_load_churn_by_country_columns_and_types():
    df = segments.load_churn_by_country()
    assert list(df.columns) == ["country", "num_customers", "churn_rate_pct", "total_revenue"]
    assert len(df) > 0
    assert pd.api.types.is_numeric_dtype(df["num_customers"])
    assert pd.api.types.is_numeric_dtype(df["churn_rate_pct"])
    assert pd.api.types.is_numeric_dtype(df["total_revenue"])
    assert pd.api.types.is_object_dtype(df["country"])


def test_load_churn_by_country_known_row():
    df = segments.load_churn_by_country()
    uk = df[df["country"] == "United Kingdom"].iloc[0]
    assert uk["num_customers"] == 5350
    assert uk["churn_rate_pct"] == 51.1


def test_load_recency_risk_tiers_columns_and_order():
    df = segments.load_recency_risk_tiers()
    assert list(df.columns) == ["risk_tier", "num_customers", "avg_customer_value", "total_revenue"]
    assert len(df) == 4
    # tiers are already ordered 1..4 by the SQL query itself
    assert df["risk_tier"].tolist()[0].startswith("1.")
    assert df["risk_tier"].tolist()[-1].startswith("4.")


def test_load_recency_risk_tiers_revenue_sums_to_known_total():
    df = segments.load_recency_risk_tiers()
    # cross-checked against sql/QUERY_RESULTS.md's 03_revenue_at_risk.sql totals
    assert df["total_revenue"].sum() == 11380744.0 + 2900458.0 + 1068784.0 + 2393443.0
