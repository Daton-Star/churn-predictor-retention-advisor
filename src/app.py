"""Streamlit UI: pick an at-risk customer, see their SHAP risk factors, and
generate an on-demand plain-English explanation + guarded retention action
from Gemini.

Run locally: `streamlit run src/app.py`
Deploy: push to GitHub, point Streamlit Community Cloud at this file, and
set GEMINI_API_KEY in the app's Secrets -- see README for details.

This file intentionally contains no business logic of its own: it only
loads pre-computed artifacts from models/ (built by train_model.py and
explain.py) and calls into explain.py / genai_advisor.py. That keeps the
UI layer thin and means the exact same scoring/explanation logic is used
whether it's invoked from the CLI or from this app.
"""

import joblib
import pandas as pd
import streamlit as st

from config import FEATURE_COLUMNS, ID_COLUMN, MODEL_PATH, SHAP_VALUES_PATH, TEST_FEATURES_PATH, TEST_PREDICTIONS_PATH
from explain import get_top_factors
from genai_advisor import generate_customer_explanation

st.set_page_config(page_title="Churn Predictor & Retention Advisor", layout="wide")


@st.cache_resource
def load_artifacts():
    model_bundle = joblib.load(MODEL_PATH)
    shap_bundle = joblib.load(SHAP_VALUES_PATH)
    predictions = pd.read_csv(TEST_PREDICTIONS_PATH)
    test_features = pd.read_csv(TEST_FEATURES_PATH)
    return model_bundle, shap_bundle, predictions, test_features


def main():
    st.title("Intelligent Churn Predictor & Retention Advisor")
    st.caption(
        "Trained on the UCI Online Retail II dataset. Select an at-risk customer to see "
        "the statistical factors driving their churn score, then generate a plain-English "
        "explanation and a guarded retention recommendation with Gemini."
    )

    try:
        model_bundle, shap_bundle, predictions, test_features = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Run the pipeline first:\n\n"
            "`python src/data_prep.py && python src/train_model.py && python src/explain.py`"
        )
        st.stop()

    shap_df = shap_bundle["shap_values"]

    at_risk = predictions[predictions["y_pred"] == 1].sort_values("churn_probability", ascending=False)

    st.sidebar.header("Model")
    st.sidebar.write(f"**Algorithm:** {model_bundle['model_name']}")
    st.sidebar.write(f"**At-risk customers in test set:** {len(at_risk):,} / {len(predictions):,}")

    if at_risk.empty:
        st.warning("No customers were flagged as at-risk in the test set.")
        st.stop()

    options = [
        f"Customer {row.customer_id}  —  {row.churn_probability:.0%} churn probability"
        for row in at_risk.itertuples()
    ]
    id_lookup = dict(zip(options, at_risk["customer_id"]))

    selected_label = st.selectbox("Select an at-risk customer", options)
    customer_id = id_lookup[selected_label]

    customer_row = predictions[predictions[ID_COLUMN] == customer_id].iloc[0]
    col1, col2 = st.columns(2)
    col1.metric("Predicted churn probability", f"{customer_row['churn_probability']:.1%}")
    col2.metric("Model prediction", "At risk" if customer_row["y_pred"] == 1 else "Not at risk")

    # test_features carries the exact feature vector explain.py computed
    # SHAP values against, so the table shown here matches the model input.
    top_factors = get_top_factors(customer_id, shap_df, test_features[FEATURE_COLUMNS], test_features[ID_COLUMN], top_n=5)

    st.subheader("Top SHAP risk factors")
    st.caption(
        "How much each feature pushed this customer's predicted churn probability up or down, "
        "relative to the model's average prediction."
    )
    display_factors = top_factors.copy()
    display_factors["shap_value"] = display_factors["shap_value"].map(lambda v: f"{v:+.3f}")
    st.dataframe(display_factors, use_container_width=True, hide_index=True)

    st.subheader("Gemini retention recommendation")
    if st.button("Generate explanation", type="primary"):
        with st.spinner("Calling Gemini..."):
            result = generate_customer_explanation(
                customer_id=customer_id,
                churn_probability=customer_row["churn_probability"],
                top_factors=top_factors.to_dict("records"),
            )

        if result["needs_manual_review"]:
            st.warning(f"⚠️ Flagged for manual review: {result['error']}")
            if result["explanation"]:
                st.write(f"**Model's explanation (unvalidated):** {result['explanation']}")
        else:
            st.success(f"**Recommended action:** {result['recommended_action']}")
            st.write(result["explanation"])


if __name__ == "__main__":
    main()
