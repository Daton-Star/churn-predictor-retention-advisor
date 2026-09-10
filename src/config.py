"""Shared constants for the churn predictor pipeline.

Centralizing paths and business rules here means the churn threshold, the
random seed, and the guardrail action list are each defined exactly once and
referenced everywhere else -- so a reviewer (or an interviewer) only has to
look in one place to see every assumption the project makes.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_PATH = PROJECT_ROOT / "data_raw" / "online_retail_II.xlsx"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"
CUSTOMER_FEATURES_PATH = DATA_PROCESSED_DIR / "customer_features.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
TEST_PREDICTIONS_PATH = MODELS_DIR / "test_predictions.csv"
TEST_FEATURES_PATH = MODELS_DIR / "test_features.csv"
SHAP_VALUES_PATH = MODELS_DIR / "shap_values.joblib"

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "model_comparison.json"

MLFLOW_TRACKING_URI = f"file:{PROJECT_ROOT / 'mlruns'}"
MLFLOW_EXPERIMENT_NAME = "churn-predictor"

# --- Business rules -------------------------------------------------------

# 90 days (~1 quarter) is a common churn window for a non-subscription
# retailer whose customers don't buy on a fixed cadence. It's long enough
# that we don't mislabel a customer who simply buys quarterly rather than
# monthly, but short enough that the retention team can still act before
# the relationship is fully dead. See data_prep.py for the inter-purchase
# gap analysis that validates this choice against the actual data.
CHURN_THRESHOLD_DAYS = 90

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5

# Feature columns used by the model. Kept as an explicit list (rather than
# "all numeric columns") so that adding a column to the features table never
# silently changes the model's input space.
#
# recency_days is deliberately NOT a model feature even though it's the
# most obviously "churn-related" number in the table: the churn label
# itself is defined as recency_days > CHURN_THRESHOLD_DAYS (see
# data_prep.py), so feeding recency_days into the model would just let it
# memorize the labeling rule instead of learning the underlying behavioral
# pattern (low frequency, low spend, narrow product range) that predicts
# churn *before* a customer's recency crosses the threshold. It's still
# computed and kept in customer_features.csv for display purposes (e.g.
# "last purchased 45 days ago" in the app), just excluded from training.
FEATURE_COLUMNS = [
    "frequency",
    "monetary",
    "avg_order_value",
    "distinct_products",
    "tenure_days",
    "avg_days_between_purchases",
    "cancellation_rate",
]
TARGET_COLUMN = "churned"
ID_COLUMN = "customer_id"

# GenAI guardrail: the model is only ever allowed to recommend one of these
# actions. Anything else gets flagged for manual review instead of being
# shown to a business user -- see genai_advisor.py.
ALLOWED_ACTIONS = [
    "send discount offer",
    "proactive support call",
    "loyalty upgrade",
    "re-engagement email",
    "no action needed",
]
