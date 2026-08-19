"""Low-review-score model: predict a 1-2 star review at order time.

Same pipeline machinery as the churn model, different target and feature list.
Kept separate rather than parameterising one module, because the two problems
differ in ways that matter:

* grain       churn is per customer, this is per order
* base rate   1.8% vs 12.8%
* features    this one KEEPS month_index (train and test share a month range),
              and EXCLUDES has_comment / comment_length, which are properties
              of the review itself and would be circular
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "target_low_review"
TS_COL = "order_purchase_timestamp"

NUMERIC_FEATURES = [
    "revenue", "log_revenue", "freight", "freight_ratio", "avg_item_price",
    "n_items", "n_distinct_products", "n_sellers", "n_categories",
    "total_weight_g", "max_installments", "n_payment_methods",
    "delivery_days", "days_late", "days_early", "approval_hours",
    "late_over_0d", "late_over_3d", "late_over_7d",
    "is_split_order", "late_flag_reliable",
    # Kept here, unlike the churn model: train and test cover the same month
    # range, so the trend is interpolated rather than extrapolated.
    "month_index", "month_of_year", "dow",
]

CATEGORICAL_FEATURES = ["customer_state", "primary_category", "primary_payment_type"]

# Anything derived from the review would make the problem circular.
LEAKY_COLUMNS = [
    "review_score", "has_comment", "comment_length", "target_low_review",
    "sentiment_negative_proba",
]


def assert_no_leakage(features: list[str]) -> None:
    bad = sorted(set(features) & set(LEAKY_COLUMNS))
    if bad:
        raise ValueError(f"Leaky columns in feature list: {bad}")


__all__ = [
    "TARGET", "TS_COL", "NUMERIC_FEATURES", "CATEGORICAL_FEATURES",
    "LEAKY_COLUMNS", "assert_no_leakage",
]
