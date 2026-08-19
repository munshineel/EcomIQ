"""Model-ready feature tables and time-aware splits.

Two targets, two grains:

    build_churn_features()  -> one row per customer_unique_id
                               target: repeated within CHURN_HORIZON_DAYS
    build_review_features() -> one row per order_id
                               target: review score <= 2

Both are built from first-order / at-order information only. Anything the
business could not have known at that moment is excluded.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import ANALYSIS_END, ANALYSIS_START, REPEAT_GAP_DAYS

logger = logging.getLogger(__name__)

CHURN_HORIZON_DAYS = 180
LOW_REVIEW_THRESHOLD = 2

# EDA G17: review score collapses at these thresholds, so encode them
# explicitly rather than trusting a model to find a step function in a
# continuous day count.
LATE_THRESHOLDS = (0, 3, 7)


# ---------------------------------------------------------------------------
# shared feature blocks
# ---------------------------------------------------------------------------
def add_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Threshold features around the promised delivery date."""
    out = df.copy()
    err = out["estimate_error_days"]

    out["days_late"] = err.clip(lower=0)
    out["days_early"] = (-err).clip(lower=0)
    for t in LATE_THRESHOLDS:
        out[f"late_over_{t}d"] = (err > t).astype("float").where(err.notna())

    # Multi-seller orders record one shipment's arrival, not the last, so the
    # late flag understates reality there (EDA B6). Flag it for the model.
    out["is_split_order"] = (out["n_sellers"] > 1).astype("float")
    out["late_flag_reliable"] = (out["n_sellers"] <= 1).astype("float")
    return out


def add_time_controls(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Order-month controls.

    EDA E14: the monthly late rate swings between roughly 1% and 21%. Without
    a time control the model learns that drift and reports it as a delivery
    effect. month_index is a linear trend; month_of_year captures season.
    """
    out = df.copy()
    ts = out[ts_col]
    start = pd.Timestamp(ANALYSIS_START)
    out["month_index"] = (
        (ts.dt.year - start.year) * 12 + (ts.dt.month - start.month)
    ).astype("float")
    out["month_of_year"] = ts.dt.month.astype("float")
    out["dow"] = ts.dt.dayofweek.astype("float")
    return out


def add_money_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_revenue"] = np.log1p(out["revenue"])
    out["freight_ratio"] = out["freight_ratio"].clip(upper=3.0)
    out["avg_item_price"] = out["revenue"] / out["n_items"].replace(0, np.nan)
    return out


# ---------------------------------------------------------------------------
# target 1: churn / repeat purchase
# ---------------------------------------------------------------------------
def build_churn_features(
    order_frame: pd.DataFrame,
    horizon_days: int = CHURN_HORIZON_DAYS,
    repeat_gap_days: int = REPEAT_GAP_DAYS,
) -> pd.DataFrame:
    """One row per customer. Target: did they return within `horizon_days`?

    Cohort rule: the first order must be early enough that a full horizon of
    observation exists inside the data. Customers whose first order is later
    than ANALYSIS_END - horizon are dropped -- keeping them would label
    "not yet returned" as "churned" and poison the target.
    """
    start, end = pd.Timestamp(ANALYSIS_START), pd.Timestamp(ANALYSIS_END)
    cutoff = end - pd.Timedelta(days=horizon_days)

    active = order_frame[order_frame["is_active"]].sort_values(
        ["customer_unique_id", "order_purchase_timestamp"]
    )
    active = active.assign(
        order_rank=active.groupby("customer_unique_id").cumcount()
    )

    firsts = active[active["order_rank"] == 0].set_index("customer_unique_id")
    laters = active[active["order_rank"] > 0]

    # Gap from each customer's first order to each subsequent order.
    gaps = (
        laters["order_purchase_timestamp"].to_numpy()
        - firsts["order_purchase_timestamp"]
        .reindex(laters["customer_unique_id"])
        .to_numpy()
    ) / np.timedelta64(1, "D")
    returned_ids = set(
        laters.loc[(gaps > repeat_gap_days) & (gaps <= horizon_days), "customer_unique_id"]
    )

    cohort = firsts[
        (firsts["order_purchase_timestamp"] >= start)
        & (firsts["order_purchase_timestamp"] <= cutoff)
    ].reset_index()

    df = add_delivery_features(cohort)
    df = add_time_controls(df, "order_purchase_timestamp")
    df = add_money_features(df)

    df["has_comment"] = df["has_comment"].fillna(False).astype("float")
    df["target_repeat"] = df["customer_unique_id"].isin(returned_ids).astype("int8")

    keep = [
        "customer_unique_id", "order_purchase_timestamp",
        "revenue", "log_revenue", "freight", "freight_ratio", "avg_item_price",
        "n_items", "n_distinct_products", "n_sellers", "n_categories",
        "total_weight_g", "max_installments", "n_payment_methods",
        "delivery_days", "days_late", "days_early",
        *[f"late_over_{t}d" for t in LATE_THRESHOLDS],
        "is_split_order", "late_flag_reliable",
        "review_score", "has_comment", "comment_length",
        "month_index", "month_of_year", "dow",
        "customer_state", "primary_category", "primary_payment_type",
        "target_repeat",
    ]
    out = df[keep].copy()

    logger.info(
        "churn features: %d rows | positives %d (%.2f%%) | window %s..%s",
        len(out), out.target_repeat.sum(), out.target_repeat.mean() * 100,
        out.order_purchase_timestamp.min().date(),
        out.order_purchase_timestamp.max().date(),
    )
    return out


# ---------------------------------------------------------------------------
# target 2: low review score
# ---------------------------------------------------------------------------
def build_review_features(
    order_frame: pd.DataFrame, threshold: int = LOW_REVIEW_THRESHOLD
) -> pd.DataFrame:
    """One row per delivered order with a review. Target: score <= threshold.

    `has_comment` and `comment_length` are EXCLUDED here: both are properties
    of the review itself, so using them to predict the score is circular.
    They stay in the churn table, where the review already happened.
    """
    start, end = pd.Timestamp(ANALYSIS_START), pd.Timestamp(ANALYSIS_END)

    df = order_frame[
        order_frame["is_fulfilled"]
        & order_frame["review_score"].notna()
        & order_frame["revenue"].notna()
        & order_frame["order_purchase_timestamp"].between(start, end)
    ].copy()

    df = add_delivery_features(df)
    df = add_time_controls(df, "order_purchase_timestamp")
    df = add_money_features(df)
    df["target_low_review"] = (df["review_score"] <= threshold).astype("int8")

    keep = [
        "order_id", "customer_unique_id", "order_purchase_timestamp",
        "revenue", "log_revenue", "freight", "freight_ratio", "avg_item_price",
        "n_items", "n_distinct_products", "n_sellers", "n_categories",
        "total_weight_g", "max_installments", "n_payment_methods",
        "delivery_days", "days_late", "days_early", "approval_hours",
        *[f"late_over_{t}d" for t in LATE_THRESHOLDS],
        "is_split_order", "late_flag_reliable",
        "month_index", "month_of_year", "dow",
        "customer_state", "primary_category", "primary_payment_type",
        "target_low_review",
    ]
    out = df[keep].copy()

    logger.info(
        "review features: %d rows | positives %d (%.2f%%)",
        len(out), out.target_low_review.sum(), out.target_low_review.mean() * 100,
    )
    return out


# ---------------------------------------------------------------------------
# time-aware split
# ---------------------------------------------------------------------------
def time_split(
    df: pd.DataFrame, ts_col: str = "order_purchase_timestamp", test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: earliest rows train, latest rows test.

    Never use a random split here. Both targets involve a future event, so a
    random split lets the model see later months during training and predict
    earlier ones -- optimistic scores that will not survive deployment.
    """
    ordered = df.sort_values(ts_col)
    cut = int(len(ordered) * (1 - test_size))
    train, test = ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()

    logger.info(
        "split: train %d (%s..%s) | test %d (%s..%s)",
        len(train), train[ts_col].min().date(), train[ts_col].max().date(),
        len(test), test[ts_col].min().date(), test[ts_col].max().date(),
    )
    return train, test


def split_summary(
    train: pd.DataFrame, test: pd.DataFrame, target: str,
    ts_col: str = "order_purchase_timestamp",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rows": [len(train), len(test)],
            "positives": [int(train[target].sum()), int(test[target].sum())],
            "positive_rate": [train[target].mean(), test[target].mean()],
            "start": [train[ts_col].min().date(), test[ts_col].min().date()],
            "end": [train[ts_col].max().date(), test[ts_col].max().date()],
        },
        index=["train", "test"],
    )


# ---------------------------------------------------------------------------
# segmentation features (unsupervised - no target, all customers)
# ---------------------------------------------------------------------------
SEGMENTATION_FEATURES = [
    "recency_days",
    "n_orders",
    "log_total_revenue",
    "log_avg_order_value",
    "total_items",
    "n_categories",
    "mean_freight_ratio",
    "mean_delivery_days",
    "mean_review_score",
    "mean_installments",
]


def build_segmentation_features(
    order_frame: pd.DataFrame, customer_frame: pd.DataFrame
) -> pd.DataFrame:
    """One row per customer, for clustering. No target column.

    Uses every customer, not the churn cohort: segmentation has no outcome to
    observe, so the 180-day cutoff does not apply.

    Note on frequency: 98% of customers have exactly one order, so `n_orders`
    is near-constant and will contribute almost nothing. It is kept because
    its uselessness is itself a finding worth showing, not hiding.
    """
    end = pd.Timestamp(ANALYSIS_END)

    active = order_frame[order_frame["is_active"]]
    per_cust = active.groupby("customer_unique_id").agg(
        mean_freight_ratio=("freight_ratio", "mean"),
        mean_installments=("max_installments", "mean"),
        n_categories_touched=("primary_category", "nunique"),
    )

    df = customer_frame.set_index("customer_unique_id").join(per_cust)
    df = df[df["first_order_ts"] <= end].copy()

    df["recency_days"] = (end - df["last_order_ts"]).dt.total_seconds() / 86400
    df["log_total_revenue"] = np.log1p(df["total_revenue"])
    df["log_avg_order_value"] = np.log1p(df["avg_order_value"])
    df["n_categories"] = df["n_categories_touched"]
    df["mean_freight_ratio"] = df["mean_freight_ratio"].clip(upper=3.0)

    out = df[SEGMENTATION_FEATURES + ["total_revenue", "customer_state"]].copy()
    logger.info("segmentation features: %d customers x %d features",
                len(out), len(SEGMENTATION_FEATURES))
    return out.reset_index()


def prepare_matrix(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Median-impute + add missing indicators, then return a numeric matrix.

    K-Means cannot accept NaN. Median imputation keeps the row; the companion
    indicator column preserves the fact that the value was missing, which for
    `mean_review_score` means "never reviewed" and is real information.
    """
    X = df[features].copy()
    for col in features:
        if X[col].isna().any():
            X[f"{col}_missing"] = X[col].isna().astype(float)
            X[col] = X[col].fillna(X[col].median())
    return X.to_numpy(dtype=float), X


__all__ = [
    "build_segmentation_features",
    "prepare_matrix",
    "SEGMENTATION_FEATURES",
    "build_churn_features", "build_review_features",
    "add_delivery_features", "add_time_controls", "add_money_features",
    "time_split", "split_summary",
    "CHURN_HORIZON_DAYS", "LOW_REVIEW_THRESHOLD", "LATE_THRESHOLDS",
]