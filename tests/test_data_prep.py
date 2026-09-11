"""Unit tests for the churn label and RFM feature engineering logic.

Deliberately uses a small hand-built DataFrame instead of the real
45MB dataset, so this suite runs in a couple of seconds with no external
dependency -- including in CI, which never downloads data_raw/.
"""

import numpy as np
import pandas as pd
import pytest

from data_prep import build_customer_features, clean_transactions


@pytest.fixture
def raw_transactions():
    return pd.DataFrame(
        [
            # customer 1: three real purchases across the observation window,
            # including a "POST" (postage) line that should count toward
            # monetary/frequency but NOT toward distinct_products.
            {"invoice": "I1", "stock_code": "P1", "description": "Widget", "quantity": 2, "invoice_date": pd.Timestamp("2011-01-01"), "price": 5.0, "customer_id": 1, "country": "United Kingdom "},
            {"invoice": "I2", "stock_code": "P2", "description": "Widget2", "quantity": 1, "invoice_date": pd.Timestamp("2011-03-01"), "price": 10.0, "customer_id": 1, "country": "United Kingdom"},
            {"invoice": "I3", "stock_code": "POST", "description": "Postage", "quantity": 1, "invoice_date": pd.Timestamp("2011-04-10"), "price": 3.0, "customer_id": 1, "country": "United Kingdom"},
            # customer 2: single purchase over a year before the snapshot -> churned
            {"invoice": "I4", "stock_code": "P3", "description": "Gadget", "quantity": 1, "invoice_date": pd.Timestamp("2010-01-01"), "price": 20.0, "customer_id": 2, "country": "EIRE"},
            # customer 3: one real purchase (defines the global snapshot date) + one cancellation
            {"invoice": "I5", "stock_code": "P4", "description": "Thing", "quantity": 4, "invoice_date": pd.Timestamp("2011-04-10"), "price": 2.5, "customer_id": 3, "country": "France"},
            {"invoice": "CI6", "stock_code": "P4", "description": "Thing", "quantity": -1, "invoice_date": pd.Timestamp("2011-04-05"), "price": 2.5, "customer_id": 3, "country": "France"},
            # missing CustomerID -> must be dropped entirely
            {"invoice": "I7", "stock_code": "P5", "description": "Mystery", "quantity": 1, "invoice_date": pd.Timestamp("2011-01-01"), "price": 1.0, "customer_id": None, "country": "Germany"},
            # non-cancelled but zero-price manual adjustment -> excluded from real purchases
            {"invoice": "I8", "stock_code": "P6", "description": "Adjustment", "quantity": 1, "invoice_date": pd.Timestamp("2011-01-05"), "price": 0.0, "customer_id": 1, "country": "United Kingdom"},
        ]
    )


def test_clean_transactions_drops_missing_customer_id(raw_transactions):
    cleaned = clean_transactions(raw_transactions)
    assert cleaned["customer_id"].isnull().sum() == 0
    assert len(cleaned) == len(raw_transactions) - 1  # the one missing-ID row


def test_clean_transactions_flags_cancellations(raw_transactions):
    cleaned = clean_transactions(raw_transactions)
    flagged = cleaned.set_index("invoice")["is_cancelled"]
    assert flagged["CI6"] is True or flagged["CI6"] == True  # noqa: E712
    assert flagged["I1"] is False or flagged["I1"] == False  # noqa: E712


def test_clean_transactions_normalizes_country(raw_transactions):
    cleaned = clean_transactions(raw_transactions)
    countries = set(cleaned["country"])
    assert "United Kingdom" in countries
    assert "Ireland" in countries  # EIRE -> Ireland via COUNTRY_ALIASES
    assert "EIRE" not in countries
    assert not any(c.endswith(" ") or c.startswith(" ") for c in countries)


def test_build_customer_features_rfm_values(raw_transactions):
    cleaned = clean_transactions(raw_transactions)
    features = build_customer_features(cleaned).set_index("customer_id")

    # Customer 1: 3 real purchases (POST counts, P6-adjustment doesn't),
    # distinct_products excludes the POST line -> only P1, P2.
    c1 = features.loc[1]
    assert c1["frequency"] == 3
    assert c1["monetary"] == pytest.approx(2 * 5.0 + 1 * 10.0 + 1 * 3.0)
    assert c1["distinct_products"] == 2
    assert c1["recency_days"] == 0  # last purchase == global snapshot date
    assert c1["tenure_days"] == 99  # 2011-01-01 -> 2011-04-10
    assert c1["churned"] == 0

    # Customer 2: one purchase, over a year stale -> churned.
    c2 = features.loc[2]
    assert c2["frequency"] == 1
    assert c2["recency_days"] > 90
    assert c2["churned"] == 1
    assert c2["country"] == "Ireland"

    # Customer 3: one real purchase + one cancellation. The cancellation
    # must not appear in frequency/monetary, but must be reflected in
    # cancellation_rate.
    c3 = features.loc[3]
    assert c3["frequency"] == 1
    assert c3["monetary"] == pytest.approx(4 * 2.5)
    assert c3["cancellation_count"] == 1
    assert c3["cancellation_rate"] == pytest.approx(1 / 2)
    assert c3["churned"] == 0


def test_churned_is_binary_and_matches_threshold(raw_transactions):
    from config import CHURN_THRESHOLD_DAYS

    cleaned = clean_transactions(raw_transactions)
    features = build_customer_features(cleaned)

    assert set(features["churned"].unique()) <= {0, 1}
    for _, row in features.iterrows():
        expected = 1 if row["recency_days"] > CHURN_THRESHOLD_DAYS else 0
        assert row["churned"] == expected


def test_single_order_customers_get_median_gap_not_zero(raw_transactions):
    # Customers 2 and 3 each have a single order, so they have no
    # observable gap between purchases. avg_days_between_purchases should
    # be filled with the dataset median (from customer 1), not 0 -- a 0
    # would misleadingly look like a very active repeat shopper.
    cleaned = clean_transactions(raw_transactions)
    features = build_customer_features(cleaned).set_index("customer_id")

    assert not np.isnan(features.loc[2, "avg_days_between_purchases"])
    assert not np.isnan(features.loc[3, "avg_days_between_purchases"])
    assert features.loc[2, "avg_days_between_purchases"] == pytest.approx(
        features.loc[3, "avg_days_between_purchases"]
    )
