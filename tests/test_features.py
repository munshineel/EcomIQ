"""Feature layer: leakage guards, targets, threshold features, temporal splits.

The leakage tests are the point. Everything else here is arithmetic; a leak is
the failure that produces a great score and a worthless model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import (
    CHURN_HORIZON_DAYS, LATE_THRESHOLDS, build_churn_features,
    build_review_features, split_summary, time_split,
)
from src.models.churn import (
    CATEGORICAL_FEATURES, LEAKY_COLUMNS, NUMERIC_FEATURES, TARGET,
    assert_no_leakage, time_split_three, tune_threshold,
)
from tests.conftest import needs_raw


# --------------------------------------------------------------- leakage
def test_leakage_guard_rejects_known_leaky_columns():
    with pytest.raises(ValueError, match="Leaky columns"):
        assert_no_leakage(NUMERIC_FEATURES + ["total_revenue"])


def test_declared_feature_list_is_clean():
    assert_no_leakage(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_month_index_excluded_from_churn_features():
    """Removed deliberately: test months fall outside the training range and
    trees cannot extrapolate a linear trend. Dropping it raised Random Forest
    PR-AUC lift from 1.19x to 1.32x."""
    assert "month_index" not in NUMERIC_FEATURES


def test_no_customer_level_aggregate_in_features():
    assert not set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) & set(LEAKY_COLUMNS)


# ----------------------------------------------------------- churn features
@needs_raw
def test_churn_cohort_respects_the_observation_window(order_frame):
    """Every customer must have a full 180 days of observable history, or
    'has not returned yet' gets mislabelled as churned."""
    churn = build_churn_features(order_frame)
    latest = churn["order_purchase_timestamp"].max()
    cutoff = pd.Timestamp("2018-08-31") - pd.Timedelta(days=CHURN_HORIZON_DAYS)
    assert latest <= cutoff


@needs_raw
def test_churn_target_is_binary_and_rare(order_frame):
    churn = build_churn_features(order_frame)
    assert set(churn[TARGET].unique()) <= {0, 1}
    assert 0.01 < churn[TARGET].mean() < 0.03


@needs_raw
def test_churn_one_row_per_customer(order_frame):
    churn = build_churn_features(order_frame)
    assert churn["customer_unique_id"].is_unique


@needs_raw
def test_threshold_features_separate_the_target(order_frame):
    """G17 found a cliff, not a slope. late_over_7d should show a large gap in
    the low-review rate; if it does not, the feature is not doing its job."""
    reviews = build_review_features(order_frame)
    for t in LATE_THRESHOLDS:
        col = f"late_over_{t}d"
        late = reviews.loc[reviews[col] == 1, "target_low_review"].mean()
        on_time = reviews.loc[reviews[col] == 0, "target_low_review"].mean()
        assert late > on_time * 3, f"{col}: {late:.3f} vs {on_time:.3f}"


@needs_raw
def test_review_target_excludes_review_derived_features(order_frame):
    """has_comment and comment_length are properties of the review itself, so
    using them to predict its score would be circular."""
    reviews = build_review_features(order_frame)
    assert "has_comment" not in reviews.columns
    assert "comment_length" not in reviews.columns


@needs_raw
def test_nulls_are_preserved_not_imputed(order_frame):
    """Never-delivered orders must stay NaN. XGBoost handles them natively and
    they describe the angriest customers."""
    churn = build_churn_features(order_frame)
    assert churn["delivery_days"].isna().sum() > 0


# ------------------------------------------------------------------- splits
@needs_raw
def test_time_split_is_chronological(order_frame):
    churn = build_churn_features(order_frame)
    train, test = time_split(churn)
    assert train["order_purchase_timestamp"].max() <= test["order_purchase_timestamp"].min()


@needs_raw
def test_three_way_split_is_ordered_and_disjoint(order_frame):
    churn = build_churn_features(order_frame)
    train, val, test = time_split_three(churn)
    assert train["order_purchase_timestamp"].max() <= val["order_purchase_timestamp"].min()
    assert val["order_purchase_timestamp"].max() <= test["order_purchase_timestamp"].min()
    assert len(train) + len(val) + len(test) == len(churn)


@needs_raw
def test_split_summary_shape(order_frame):
    churn = build_churn_features(order_frame)
    train, test = time_split(churn)
    s = split_summary(train, test, TARGET)
    assert list(s.index) == ["train", "test"]


def test_tune_threshold_returns_a_valid_probability():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    proba = np.clip(y * 0.4 + rng.random(2000) * 0.6, 0, 1)
    t = tune_threshold(y, proba)
    assert 0.0 <= t <= 1.0