"""Analysis frames: grain, revenue reconciliation, the basket-split correction.

Grain is the load-bearing property here. A silent duplication in a join would
double-count revenue everywhere downstream and would not raise anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import REPEAT_GAP_DAYS
from src.data.prepare import ACTIVE_STATUSES, build_customer_frame, category_labels
from tests.conftest import needs_raw


@needs_raw
def test_order_frame_grain_is_one_row_per_order(order_frame):
    assert order_frame["order_id"].is_unique


@needs_raw
def test_customer_frame_grain_is_one_row_per_person(customer_frame):
    assert customer_frame["customer_unique_id"].is_unique


@needs_raw
def test_revenue_reconciles_to_raw_line_items(tables, order_frame):
    """The single most important test in the suite."""
    assert np.isclose(order_frame["revenue"].sum(), tables["order_items"]["price"].sum())


@needs_raw
def test_customer_count(customer_frame):
    assert len(customer_frame) == 94_986


@needs_raw
def test_basket_split_correction(customer_frame):
    """Raw repeat rate 3.04% -> corrected 1.92%, removing 1,063 basket splits.

    Olist assigns separate order_ids to one shopping session across sellers, so
    a second order minutes later is not a returning customer.
    """
    raw = customer_frame["is_repeat_raw"].sum()
    corrected = customer_frame["is_repeat"].fillna(False).sum()
    assert raw - corrected == 1_063
    assert round(corrected / len(customer_frame) * 100, 2) == 1.92


@needs_raw
def test_repeat_gap_comes_from_config(order_frame):
    """Changing REPEAT_GAP_DAYS must actually change the output -- guards
    against the value being hardcoded again."""
    strict = build_customer_frame(order_frame, repeat_gap_days=90)
    loose = build_customer_frame(order_frame, repeat_gap_days=0)
    assert strict["is_repeat"].sum() < loose["is_repeat"].sum()
    assert REPEAT_GAP_DAYS == 7


@needs_raw
def test_freight_ratio_has_no_infinities(order_frame):
    """775 orders have no line items. An unguarded division yields inf, which
    silently poisons every downstream mean and quantile without raising."""
    fr = order_frame["freight_ratio"]
    assert not np.isinf(fr.dropna()).any()


@needs_raw
def test_estimate_error_is_not_truncated(order_frame):
    """Computed via total_seconds()/86400, not .dt.days, which truncates toward
    zero and would report an order 0.4 days late as on time."""
    err = order_frame["estimate_error_days"].dropna()
    assert not np.allclose(err, err.round())


@needs_raw
def test_late_flag_matches_estimate_error(order_frame):
    sub = order_frame.dropna(subset=["estimate_error_days"])
    assert (sub["is_late"] == (sub["estimate_error_days"] > 0)).all()


@needs_raw
def test_active_statuses_superset_of_fulfilled(order_frame):
    assert order_frame.loc[order_frame["is_fulfilled"], "is_active"].all()
    assert set(order_frame.loc[order_frame["is_active"], "order_status"]) <= set(ACTIVE_STATUSES)


@needs_raw
def test_category_translation_covers_every_category(tables):
    """71 shipped translations for 73 categories; the two gaps are hardcoded."""
    labels = category_labels(tables)
    present = set(tables["products"]["product_category_name"].dropna())
    assert present <= set(labels), present - set(labels)