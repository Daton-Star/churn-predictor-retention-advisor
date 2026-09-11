"""Unit tests for get_top_factors() -- the per-customer SHAP ranking used
by both explain.py's CLI output and app.py's Streamlit table.

Uses hand-built SHAP values rather than a real fitted model, since what's
under test is the ranking/lookup logic, not SHAP itself.
"""

import pandas as pd
import pytest

from config import FEATURE_COLUMNS
from explain import get_top_factors


@pytest.fixture
def synthetic_shap_data():
    ids = pd.Series([101, 102])
    # Two customers, one row of SHAP values each, one column per feature.
    shap_values = {col: [0.0, 0.0] for col in FEATURE_COLUMNS}
    # Customer 101: frequency is the dominant (negative -> safer) factor.
    shap_values["frequency"][0] = -0.40
    shap_values["monetary"][0] = 0.10
    shap_values["tenure_days"][0] = 0.02
    # Customer 102: cancellation_rate is the dominant (positive -> riskier) factor.
    shap_values["cancellation_rate"][1] = 0.35
    shap_values["frequency"][1] = -0.05

    shap_df = pd.DataFrame(shap_values)
    X = pd.DataFrame({col: [1.0, 2.0] for col in FEATURE_COLUMNS})
    return shap_df, X, ids


def test_top_factors_sorted_by_absolute_impact(synthetic_shap_data):
    shap_df, X, ids = synthetic_shap_data
    top = get_top_factors(101, shap_df, X, ids, top_n=3)

    assert list(top["feature"])[0] == "frequency"  # largest |shap_value| = 0.40
    assert top.iloc[0]["direction"] == "decreases churn risk"
    # Results must be sorted strictly by descending absolute impact.
    abs_impacts = top["shap_value"].abs().tolist()
    assert abs_impacts == sorted(abs_impacts, reverse=True)


def test_top_factors_direction_labels(synthetic_shap_data):
    shap_df, X, ids = synthetic_shap_data
    top = get_top_factors(102, shap_df, X, ids, top_n=2)

    assert top.iloc[0]["feature"] == "cancellation_rate"
    assert top.iloc[0]["direction"] == "increases churn risk"


def test_top_factors_respects_top_n(synthetic_shap_data):
    shap_df, X, ids = synthetic_shap_data
    top = get_top_factors(101, shap_df, X, ids, top_n=2)
    assert len(top) == 2


def test_unknown_customer_id_raises(synthetic_shap_data):
    shap_df, X, ids = synthetic_shap_data
    with pytest.raises(KeyError):
        get_top_factors(999, shap_df, X, ids, top_n=3)
