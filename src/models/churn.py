"""Churn (repeat-purchase) modelling: splits, pipelines, evaluation.

Kept in src/ because the dashboard and inference pipeline need the identical
feature list, preprocessing and threshold that training used. A notebook copy
would drift.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

TARGET = "target_repeat"
TS_COL = "order_purchase_timestamp"

NUMERIC_FEATURES = [
    "revenue", "log_revenue", "freight", "freight_ratio", "avg_item_price",
    "n_items", "n_distinct_products", "n_sellers", "n_categories",
    "total_weight_g", "max_installments", "n_payment_methods",
    "delivery_days", "days_late", "days_early",
    "late_over_0d", "late_over_3d", "late_over_7d",
    "is_split_order", "late_flag_reliable",
    "review_score", "has_comment", "comment_length",
    "month_of_year", "dow",
    # month_index is deliberately EXCLUDED. It was added to absorb the E14
    # delivery drift, but the test period lies outside the training month range
    # and trees cannot extrapolate a linear trend. Dropping it raised Random
    # Forest PR-AUC lift from 1.19x to 1.32x. It IS kept for the review model,
    # where train and test share the same month range.
]

CATEGORICAL_FEATURES = ["customer_state", "primary_category", "primary_payment_type"]

# Columns that must NEVER enter the model. Listed explicitly so the guard below
# fails loudly if a future refactor reintroduces one.
LEAKY_COLUMNS = [
    "n_orders", "total_revenue", "avg_order_value", "last_order_ts",
    "days_to_second_order", "is_repeat", "is_repeat_raw", "target_repeat",
]


def assert_no_leakage(features: list[str]) -> None:
    bad = sorted(set(features) & set(LEAKY_COLUMNS))
    if bad:
        raise ValueError(f"Leaky columns in feature list: {bad}")


def time_split_three(
    df: pd.DataFrame, ts_col: str = TS_COL, val_size: float = 0.2,
    test_size: float = 0.2, target: str = TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test.

    Validation exists so the decision threshold is chosen WITHOUT touching
    test. Tuning a threshold on test and then reporting test metrics is a
    subtle but real form of leakage.
    """
    ordered = df.sort_values(ts_col).reset_index(drop=True)
    n = len(ordered)
    i_train = int(n * (1 - val_size - test_size))
    i_val = int(n * (1 - test_size))

    train, val, test = ordered.iloc[:i_train], ordered.iloc[i_train:i_val], ordered.iloc[i_val:]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        logger.info(
            "%-5s %6d rows | %s..%s | positives %d (%.2f%%)",
            name, len(part), part[ts_col].min().date(), part[ts_col].max().date(),
            part[target].sum(), part[target].mean() * 100,
        )
    return train.copy(), val.copy(), test.copy()


def _to_object_nan(X):
    """Convert extension-dtype columns to object with real np.nan."""
    return pd.DataFrame(X).astype(object).where(pd.notna(pd.DataFrame(X)), np.nan)


def _preprocessor(scale: bool, impute: bool,
                  numeric: list[str] | None = None,
                  categorical: list[str] | None = None) -> ColumnTransformer:
    """Numeric and categorical handling.

    Trees do not need scaling and XGBoost handles NaN natively, so imputation
    and scaling are switched off for it. Logistic regression needs both.
    """
    num_steps = []
    if impute:
        num_steps.append(("impute", SimpleImputer(strategy="median")))
    if scale:
        num_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(num_steps) if num_steps else "passthrough"

    # pandas "string" dtype stores missing as pd.NA, whose truth value raises
    # inside sklearn's imputer. Cast to plain object/np.nan first.
    categorical_pipe = Pipeline([
        ("to_object", FunctionTransformer(_to_object_nan, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=100,
                                sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric or NUMERIC_FEATURES),
        ("cat", categorical_pipe, categorical or CATEGORICAL_FEATURES),
    ])


def make_models(scale_pos_weight: float,
                numeric: list[str] | None = None,
                categorical: list[str] | None = None) -> dict[str, Pipeline]:
    """Three models on the same features.

    class_weight / scale_pos_weight rather than SMOTE: synthetic oversampling
    invents minority examples, and with only ~600 training positives those
    synthetics would dominate. Reweighting changes the loss, not the data.
    """
    models: dict[str, Pipeline] = {}

    models["logistic"] = Pipeline([
        ("prep", _preprocessor(True, True, numeric, categorical)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_SEED)),
    ])

    models["random_forest"] = Pipeline([
        ("prep", _preprocessor(False, True, numeric, categorical)),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=50,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_SEED)),
    ])

    from xgboost import XGBClassifier

    models["xgboost"] = Pipeline([
        ("prep", _preprocessor(False, False, numeric, categorical)),
        ("clf", XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=10, reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr", tree_method="hist",
            random_state=RANDOM_SEED, n_jobs=-1)),
    ])
    return models


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Threshold-free and threshold-dependent metrics together.

    PR-AUC is the headline. Both precision and recall concern the positive
    class, so PR-AUC ignores the huge easy-negative mass that inflates both
    accuracy and ROC-AUC on a 1.8% base rate.
    """
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    base = float(np.mean(y_true))
    pr_auc = average_precision_score(y_true, proba)
    return {
        "threshold": threshold,
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / base if base else np.nan,
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": (tp + tn) / len(y_true),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "flagged": int(pred.sum()),
        "base_rate": base,
    }


def tune_threshold(
    y_true: np.ndarray, proba: np.ndarray, min_precision: float | None = None
) -> float:
    """Pick a threshold on VALIDATION data.

    Default: maximise F1. If `min_precision` is given, take the threshold with
    the highest recall that still meets that precision -- which is the shape a
    real campaign brief takes ("we can tolerate 90% waste, not 98%").
    """
    grid = np.quantile(proba, np.linspace(0.50, 0.999, 200))
    best_t, best_score = 0.5, -1.0
    for t in np.unique(grid):
        pred = (proba >= t).astype(int)
        if pred.sum() == 0:
            continue
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        if min_precision is not None:
            score = r if p >= min_precision else -1.0
        else:
            score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_t, best_score = float(t), score
    return best_t


def shap_values_positive_class(explainer, X) -> np.ndarray:
    """SHAP values for the positive class, whatever shape the library returns.

    shap >= 0.45 returns different shapes per model family:
      * XGBClassifier          -> (n_samples, n_features)
      * RandomForestClassifier -> (n_samples, n_features, n_classes)
      * some versions          -> list of arrays, one per class

    Calling `.mean(axis=0)` on the 3-D case yields (n_features, n_classes),
    which fails with "Data must be 1-dimensional". Normalise here so callers
    always get a 2-D (n_samples, n_features) array.
    """
    values = explainer.shap_values(X)

    if isinstance(values, list):
        return np.asarray(values[1] if len(values) > 1 else values[0])

    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def explain(clf, X_background, X_explain) -> np.ndarray:
    """Pick the right SHAP explainer for the fitted estimator.

    TreeExplainer only handles tree ensembles. If model selection picks the
    logistic baseline -- which it can, and on this data it does -- passing it to
    TreeExplainer raises. LinearExplainer is exact and fast for linear models,
    so dispatch on type rather than assuming the winner is a tree.
    """
    import shap

    name = type(clf).__name__
    if name in {"LogisticRegression", "LinearSVC", "SGDClassifier", "Ridge"}:
        explainer = shap.LinearExplainer(clf, X_background)
    else:
        explainer = shap.TreeExplainer(clf)
    logger.info("SHAP explainer for %s: %s", name, type(explainer).__name__)
    return shap_values_positive_class(explainer, X_explain)


__all__ = [
    "shap_values_positive_class", "explain",
    "TARGET", "TS_COL", "NUMERIC_FEATURES", "CATEGORICAL_FEATURES",
    "LEAKY_COLUMNS", "assert_no_leakage", "time_split_three",
    "make_models", "evaluate", "tune_threshold",
]