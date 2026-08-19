"""Building analysis-ready frames from the raw Olist tables.

Two outputs, with deliberately different grains:

    build_order_frame()     -> one row per order_id
    build_customer_frame()  -> one row per customer_unique_id

Everything downstream (EDA, segmentation, churn, forecasting) reads one of
these two. Keeping the grain explicit in the function name is not cosmetic:
joining a customer-grain frame onto an order-grain frame is the single
easiest way to silently double-count revenue.

Design notes
------------
* Reviews are keyed by order_id, not review_id. 789 review_id values are
  reused across different orders in the source, and 547 orders carry more
  than one review, so we keep the most recent review per order.
* Revenue means sum of `price` over line items. Freight is tracked
  separately because it is a cost pass-through, not product revenue.
* `estimate_error_days` is negative when an order arrives early. Olist pads
  its estimates heavily, so the median is around -12 days.
* No filtering by date or status happens here. Filtering is a per-phase
  decision, so the frame keeps everything and exposes the columns needed
  to filter. EDA needs the cancelled orders; forecasting does not.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import REPEAT_GAP_DAYS
from src.data.load import load_all

logger = logging.getLogger(__name__)

# Statuses where the customer actually received, or will receive, goods.
FULFILLED_STATUSES = ("delivered",)
# Statuses that represent real demand even if not yet delivered.
ACTIVE_STATUSES = ("delivered", "shipped", "invoiced", "processing", "approved")


def _aggregate_items(order_items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Collapse line items to one row per order.

    Also picks a `primary_category`: the category of the single most
    expensive item in the order. A mode would be arbitrary for two-item
    orders, and "most expensive" matches how a merchandiser would describe
    what the order was really for.
    """
    items = order_items.merge(
        products[["product_id", "product_category_name", "product_weight_g"]],
        on="product_id",
        how="left",
    )

    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_distinct_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        n_categories=("product_category_name", "nunique"),
        revenue=("price", "sum"),
        freight=("freight_value", "sum"),
        max_item_price=("price", "max"),
        total_weight_g=("product_weight_g", "sum"),
    )

    # idxmax on price gives the row index of the priciest item per order.
    priciest = items.loc[items.groupby("order_id")["price"].idxmax()]
    primary = priciest.set_index("order_id")["product_category_name"].rename(
        "primary_category"
    )

    return agg.join(primary)


def _aggregate_reviews(order_reviews: pd.DataFrame) -> pd.DataFrame:
    """One review per order: the most recently created one."""
    rv = order_reviews.sort_values("review_creation_date")
    rv = rv.drop_duplicates(subset="order_id", keep="last")

    out = rv.set_index("order_id")[
        ["review_score", "review_comment_message", "review_creation_date"]
    ].copy()
    out["has_comment"] = out["review_comment_message"].notna()
    out["comment_length"] = out["review_comment_message"].str.len().fillna(0).astype(int)
    return out


def _aggregate_payments(order_payments: pd.DataFrame) -> pd.DataFrame:
    """One row per order. Primary payment type = the largest payment."""
    pay = order_payments.sort_values("payment_value", ascending=False)
    primary = pay.drop_duplicates(subset="order_id", keep="first").set_index("order_id")[
        "payment_type"
    ].rename("primary_payment_type")

    agg = order_payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        n_payment_methods=("payment_sequential", "count"),
    )
    return agg.join(primary)


def category_labels(tables: dict[str, pd.DataFrame] | None = None) -> dict[str, str]:
    """Portuguese -> English category names.

    The shipped translation file covers 71 of the 73 categories present in
    `products`. The two gaps are hardcoded rather than left as NaN, because a
    missing label silently drops the category from every groupby-and-plot.
    """
    t = tables or load_all()
    mapping = dict(
        zip(
            t["category_translation"]["product_category_name"],
            t["category_translation"]["product_category_name_english"],
        )
    )
    mapping.setdefault("pc_gamer", "pc_gamer")
    mapping.setdefault(
        "portateis_cozinha_e_preparadores_de_alimentos", "kitchen_food_preparers"
    )
    return mapping


def build_order_frame(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per order, with customer, item, review and payment context."""
    t = tables or load_all()

    df = t["orders"].merge(
        t["customers"][
            ["customer_id", "customer_unique_id", "customer_state", "customer_city"]
        ],
        on="customer_id",
        how="left",
    )

    df = df.join(_aggregate_items(t["order_items"], t["products"]), on="order_id")
    df = df.join(_aggregate_reviews(t["order_reviews"]), on="order_id")
    df = df.join(_aggregate_payments(t["order_payments"]), on="order_id")

    # --- derived fields -------------------------------------------------
    purchase = df["order_purchase_timestamp"]

    df["delivery_days"] = (
        df["order_delivered_customer_date"] - purchase
    ).dt.total_seconds() / 86400
    df["approval_hours"] = (
        df["order_approved_at"] - purchase
    ).dt.total_seconds() / 3600
    # Negative = arrived before the promised date.
    df["estimate_error_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    df["is_late"] = df["estimate_error_days"] > 0

    # Guard the denominator: 775 orders have no items, so revenue is NaN.
    df["freight_ratio"] = np.where(
        df["revenue"].gt(0), df["freight"] / df["revenue"], np.nan
    )
    df["order_value"] = df["revenue"].fillna(0) + df["freight"].fillna(0)

    df["purchase_date"] = purchase.dt.date
    df["purchase_month"] = purchase.dt.to_period("M").dt.to_timestamp()
    df["purchase_dow"] = purchase.dt.dayofweek  # 0 = Monday
    df["purchase_hour"] = purchase.dt.hour

    df["is_fulfilled"] = df["order_status"].isin(FULFILLED_STATUSES)
    df["is_active"] = df["order_status"].isin(ACTIVE_STATUSES)

    logger.info("order frame: %d rows x %d cols", len(df), df.shape[1])
    return df


def build_customer_frame(
    order_frame: pd.DataFrame | None = None,
    repeat_gap_days: int = REPEAT_GAP_DAYS,
) -> pd.DataFrame:
    """One row per customer_unique_id.

    `repeat_gap_days` guards against basket splits. Olist assigns separate
    order_ids to items bought in a single session from different sellers,
    so a second order minutes after the first is not a returning customer.
    Orders closer together than this threshold are not counted as repeats.
    """
    of = order_frame if order_frame is not None else build_order_frame()
    active = of[of["is_active"]].copy()

    active = active.sort_values(["customer_unique_id", "order_purchase_timestamp"])
    active["order_rank"] = active.groupby("customer_unique_id").cumcount()

    agg = active.groupby("customer_unique_id").agg(
        n_orders=("order_id", "count"),
        total_revenue=("revenue", "sum"),
        total_freight=("freight", "sum"),
        avg_order_value=("revenue", "mean"),
        total_items=("n_items", "sum"),
        n_categories=("primary_category", "nunique"),
        first_order_ts=("order_purchase_timestamp", "min"),
        last_order_ts=("order_purchase_timestamp", "max"),
        mean_review_score=("review_score", "mean"),
        mean_delivery_days=("delivery_days", "mean"),
        any_late=("is_late", "max"),
        customer_state=("customer_state", "first"),
    )

    # Days between first and second order, for those who have a second.
    first = active[active["order_rank"] == 0].set_index("customer_unique_id")[
        "order_purchase_timestamp"
    ]
    later = active[active["order_rank"] > 0].set_index("customer_unique_id")[
        "order_purchase_timestamp"
    ]
    gaps = (later - first.reindex(later.index)).dt.total_seconds() / 86400
    agg["days_to_second_order"] = gaps.groupby(level=0).min()

    agg["is_repeat_raw"] = agg["n_orders"] > 1
    agg["is_repeat"] = agg["days_to_second_order"] > repeat_gap_days

    logger.info(
        "customer frame: %d rows | raw repeat %.2f%% | corrected repeat %.2f%%",
        len(agg),
        agg["is_repeat_raw"].mean() * 100,
        agg["is_repeat"].mean() * 100,
    )
    return agg.reset_index()


__all__ = [
    "category_labels",
    "build_order_frame",
    "build_customer_frame",
    "FULFILLED_STATUSES",
    "ACTIVE_STATUSES",
]