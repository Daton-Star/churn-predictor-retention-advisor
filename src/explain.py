"""Stage 3: per-customer explanations for why the model thinks they'll churn.

Run as a script: `python src/explain.py` to precompute and cache SHAP values
for the whole test set. Also imported by app.py to look up factors for one
customer on demand.

WHY SHAP (and specifically TreeExplainer) instead of global feature
importance
-----------------------------------------------------------------------
Random Forest's built-in feature_importances_ answers "which features does
the model rely on overall" -- useful for a slide, useless for a retention
rep who needs to know why *this* customer is flagged. SHAP decomposes each
individual prediction into additive per-feature contributions relative to
the model's average output, so "why is customer 17850 at risk" gets a real
answer ("low frequency contributed +0.18 to their churn probability")
instead of a shrug. TreeExplainer specifically is used (rather than the
much slower model-agnostic KernelExplainer) because it computes *exact*
Shapley values for tree ensembles in polynomial time by walking the trees
directly -- there's no approximation trade-off to justify for a model type
SHAP has a fast exact algorithm for.
"""

import joblib
import numpy as np
import pandas as pd
import shap

from config import FEATURE_COLUMNS, ID_COLUMN, MODEL_PATH, SHAP_VALUES_PATH, TEST_FEATURES_PATH


def load_model_bundle(path=MODEL_PATH):
    return joblib.load(path)


def compute_shap_values(model, X: pd.DataFrame):
    """Return (shap_values_for_churn_class, base_value) for every row in X.

    TreeExplainer only supports tree-based models. If the best model
    selected in train_model.py ever turns out to be Logistic Regression,
    this raises immediately rather than silently returning garbage -- SHAP
    would need shap.LinearExplainer with the fitted scaler instead, which
    is a different code path not implemented here.
    """
    model_type = type(model).__name__
    if model_type not in ("RandomForestClassifier", "XGBClassifier"):
        raise ValueError(
            f"TreeExplainer does not support {model_type}. "
            "Re-run explain.py's linear-model path (not implemented) or "
            "retrain so a tree-based model wins the comparison."
        )

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)

    if isinstance(raw, list):
        # Older SHAP API: one (n_samples, n_features) array per class.
        churn_class_values = raw[1]
        base_value = explainer.expected_value[1]
    elif raw.ndim == 3:
        # Newer SHAP API: single (n_samples, n_features, n_classes) array.
        churn_class_values = raw[:, :, 1]
        base_value = explainer.expected_value[1]
    else:
        churn_class_values = raw
        base_value = explainer.expected_value

    return churn_class_values, float(base_value)


def get_top_factors(customer_id, shap_df: pd.DataFrame, X: pd.DataFrame, ids: pd.Series, top_n=5):
    """Return the top-N SHAP factors for one customer, sorted by |impact|.

    Each factor is the customer's raw feature value alongside how many
    percentage points of churn probability that feature contributed --
    this is what gets shown in the app table and fed to the GenAI layer.
    """
    row_mask = ids.values == customer_id
    if not row_mask.any():
        raise KeyError(f"customer_id {customer_id} not found in test set")
    row_idx = np.where(row_mask)[0][0]

    shap_row = shap_df.iloc[row_idx]
    feature_row = X.iloc[row_idx]

    factors = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "value": feature_row.values,
            "shap_value": shap_row.values,
        }
    )
    factors["abs_impact"] = factors["shap_value"].abs()
    factors["direction"] = np.where(factors["shap_value"] > 0, "increases churn risk", "decreases churn risk")
    factors = factors.sort_values("abs_impact", ascending=False).head(top_n)
    return factors.drop(columns="abs_impact").reset_index(drop=True)


def main():
    bundle = load_model_bundle()
    model = bundle["model"]

    X_test_full = pd.read_csv(TEST_FEATURES_PATH)
    ids = X_test_full[ID_COLUMN]
    X_test = X_test_full[FEATURE_COLUMNS]

    print(f"Computing SHAP values for {len(X_test):,} test customers ...")
    shap_values, base_value = compute_shap_values(model, X_test)
    shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLUMNS)

    joblib.dump(
        {"shap_values": shap_df, "base_value": base_value, "ids": ids, "feature_columns": FEATURE_COLUMNS},
        SHAP_VALUES_PATH,
    )
    print(f"Saved SHAP values to {SHAP_VALUES_PATH}")

    example_id = ids.iloc[0]
    print(f"\nExample -- top factors for customer {example_id}:")
    print(get_top_factors(example_id, shap_df, X_test, ids, top_n=5).to_string(index=False))


if __name__ == "__main__":
    main()
