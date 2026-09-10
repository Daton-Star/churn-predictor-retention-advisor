"""Stage 2: compare candidate models and select/persist the best churn classifier.

Run as a script: `python src/train_model.py`
Input:  data_processed/customer_features.csv (from data_prep.py)
Output: models/churn_model.joblib, models/test_predictions.csv,
        models/test_features.csv, reports/model_comparison.json,
        and an MLflow run per model under mlruns/.

WHY SMOTE only on the training data
------------------------------------
SMOTE (Synthetic Minority Oversampling) manufactures new synthetic rows by
interpolating between real minority-class neighbors. If it's applied before
the train/test split -- or before a CV split -- some of those synthetic
points end up built from neighbors that straddle the split, so the
model is partly evaluated on points that are synthetic near-copies of its
own training data. That inflates every metric and hides how the model
would perform on genuinely unseen customers. The fix used here is an
imbalanced-learn Pipeline: SMOTE is refit inside each individual training
fold (never on the validation fold, and never on the held-out test set),
so evaluation always happens on real, untouched data.

WHY compare on AUC-ROC / F1 / precision / recall instead of accuracy
-----------------------------------------------------------------------
Accuracy rewards a model for predicting the majority class and is a poor
compass for a business decision like "who should retention spend $$ on".
AUC-ROC measures how well the model ranks churners above non-churners
across every possible decision threshold, which is exactly what matters
for prioritizing a limited retention budget. F1/precision/recall are
reported alongside it because AUC alone can't tell you whether the model
errs toward false alarms (wasted discounts) or missed churners (lost
customers) -- that trade-off is a business decision, not just a modeling
one.
"""

import json

import joblib
import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import (
    CUSTOMER_FEATURES_PATH,
    FEATURE_COLUMNS,
    ID_COLUMN,
    METRICS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    MODELS_DIR,
    N_CV_FOLDS,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_FEATURES_PATH,
    TEST_PREDICTIONS_PATH,
    TEST_SIZE,
)

CV_SCORING = ["roc_auc", "f1", "precision", "recall"]


def build_candidates():
    """Return {name: (imblearn Pipeline, mlflow flavor)} for the 3 models to compare.

    Logistic Regression gets a StandardScaler step because it's
    distance/gradient based and sensitive to feature scale; the tree
    ensembles split on raw thresholds and don't need it, so skipping it for
    them keeps the pipeline simple and keeps SHAP's TreeExplainer working
    directly on the model's native (unscaled) feature space later.
    """
    candidates = {
        "logistic_regression": (
            ImbPipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("smote", SMOTE(random_state=RANDOM_SEED)),
                    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
                ]
            ),
            "sklearn",
        ),
        "random_forest": (
            ImbPipeline(
                steps=[
                    ("smote", SMOTE(random_state=RANDOM_SEED)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1
                        ),
                    ),
                ]
            ),
            "sklearn",
        ),
        "xgboost": (
            ImbPipeline(
                steps=[
                    ("smote", SMOTE(random_state=RANDOM_SEED)),
                    (
                        "clf",
                        XGBClassifier(
                            n_estimators=300,
                            max_depth=5,
                            learning_rate=0.05,
                            random_state=RANDOM_SEED,
                            eval_metric="logloss",
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "xgboost",
        ),
    }
    return candidates


def evaluate_with_cv(pipeline, X_train, y_train):
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
    summary = {}
    for metric in CV_SCORING:
        key = f"test_{metric}"
        summary[f"cv_{metric}_mean"] = float(scores[key].mean())
        summary[f"cv_{metric}_std"] = float(scores[key].std())
    return summary


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = pd.read_csv(CUSTOMER_FEATURES_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    ids = df[ID_COLUMN]

    # Stratified split keeps the churn/non-churn ratio the same in both
    # halves, so the held-out test set is a fair miniature of the full
    # population rather than an accidentally easier or harder sample.
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Train: {len(X_train):,} rows ({y_train.mean():.1%} churn) | Test: {len(X_test):,} rows ({y_test.mean():.1%} churn)")

    candidates = build_candidates()
    comparison = {}

    for name, (pipeline, _flavor) in candidates.items():
        print(f"\n=== {name}: 5-fold stratified CV on training data ===")
        cv_summary = evaluate_with_cv(pipeline, X_train, y_train)
        comparison[name] = cv_summary
        print({k: round(v, 4) for k, v in cv_summary.items()})

        with mlflow.start_run(run_name=f"cv_{name}"):
            mlflow.log_param("model", name)
            mlflow.log_params({f"clf__{k}": v for k, v in pipeline.named_steps["clf"].get_params().items() if isinstance(v, (int, float, str, bool)) or v is None})
            mlflow.log_metrics(cv_summary)

    best_name = max(comparison, key=lambda n: comparison[n]["cv_roc_auc_mean"])
    print(f"\nBest model by mean CV AUC-ROC: {best_name}")

    best_pipeline, best_flavor = candidates[best_name]
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    print(f"\n=== Held-out test set performance ({best_name}) ===")
    print(test_metrics)
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {cm}")

    with mlflow.start_run(run_name=f"final_{best_name}"):
        mlflow.log_param("model", best_name)
        mlflow.log_metrics(test_metrics)
        mlflow.log_dict({"confusion_matrix": cm}, "confusion_matrix.json")
        if best_flavor == "xgboost":
            mlflow.xgboost.log_model(best_pipeline.named_steps["clf"], "model")
        else:
            mlflow.sklearn.log_model(best_pipeline.named_steps["clf"], "model")

    # Persist the fitted *classifier* (not the SMOTE step -- SMOTE only
    # exists to balance training data and has nothing to do at inference
    # time) plus the scaler, so downstream stages can call predict/SHAP
    # directly without re-running the whole pipeline.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fitted_clf = best_pipeline.named_steps["clf"]
    # Logistic Regression's coefficients are fit on scaled features, so its
    # fitted scaler has to travel with it for any downstream code (SHAP,
    # the Streamlit app) to produce correct predictions. Tree models have
    # no scaler step, so this is None for them -- callers must check.
    fitted_scaler = best_pipeline.named_steps.get("scaler")
    joblib.dump(
        {
            "model": fitted_clf,
            "scaler": fitted_scaler,
            "model_name": best_name,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")

    test_predictions = pd.DataFrame(
        {
            ID_COLUMN: ids_test.values,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "churn_probability": y_proba,
        }
    )
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    test_features_out = X_test.copy()
    test_features_out.insert(0, ID_COLUMN, ids_test.values)
    test_features_out.to_csv(TEST_FEATURES_PATH, index=False)
    print(f"Saved test predictions and features to {MODELS_DIR}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "cv_comparison": comparison,
        "best_model": best_name,
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "churn_rate": float(y.mean()),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved model comparison report to {METRICS_PATH}")


if __name__ == "__main__":
    main()
