"""Precompute dashboard-ready artifacts.

Run once after the modelling notebooks:
    python scripts/build_dashboard_data.py

Why precompute: the Streamlit app must not read nine raw CSVs or load a 23MB
NearestNeighbors index on every session. This writes flat, filtered tables the
app can read in milliseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import load_all                       # noqa: E402
from src.data.prepare import build_order_frame, category_labels  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

ORDER_COLS = [
    "order_id", "customer_unique_id", "order_purchase_timestamp", "order_status",
    "revenue", "freight", "freight_ratio", "n_items", "n_sellers",
    "delivery_days", "estimate_error_days", "is_late", "review_score",
    "has_comment", "customer_state", "primary_category", "primary_payment_type",
    "max_installments", "is_fulfilled",
]


def main() -> None:
    print("loading raw tables...")
    tables = load_all()
    order_frame = build_order_frame(tables)
    labels = category_labels(tables)

    # --- 1. flat order table -------------------------------------------------
    orders = order_frame[ORDER_COLS].copy()
    orders["category"] = (
        orders["primary_category"].map(labels).fillna(orders["primary_category"])
    )
    orders = orders[orders["order_purchase_timestamp"].between("2017-01-01", "2018-08-31")]
    orders.to_parquet(PROCESSED / "dashboard_orders.parquet", index=False)
    print(f"  dashboard_orders.parquet      {len(orders):,} rows")

    # --- 2. precomputed recommendations -------------------------------------
    # Only for products with real sales volume. Recommending for the 18,117
    # products sold exactly once is not something a merchandiser would action,
    # and it would make the file 10x larger.
    bundle = joblib.load(MODELS / "recommender.joblib")
    model = bundle["content"]
    catalog = model.catalog_

    popular = catalog[catalog["units_sold"] >= 5].index.tolist()
    print(f"  precomputing top-10 for {len(popular):,} products with 5+ sales...")

    rows = []
    for i, pid in enumerate(popular):
        if i % 1000 == 0 and i:
            print(f"    {i:,}/{len(popular):,}")
        for rank, rec in enumerate(model.recommend(pid, 10), start=1):
            rows.append({"seed": pid, "rank": rank, "recommended": rec})
    recs = pd.DataFrame(rows)
    recs.to_parquet(PROCESSED / "top_recommendations.parquet", index=False)
    print(f"  top_recommendations.parquet   {len(recs):,} rows")

    # --- 3. slim catalogue for lookups --------------------------------------
    cat_out = catalog[["product_category_name", "price", "units_sold", "n_orders",
                       "product_weight_g", "volume_cm3", "product_photos_qty"]].copy()
    cat_out["category"] = (
        cat_out["product_category_name"].map(labels).fillna(cat_out["product_category_name"])
    )
    cat_out.reset_index().to_parquet(PROCESSED / "product_catalog.parquet", index=False)
    print(f"  product_catalog.parquet       {len(cat_out):,} rows")

    print("\ndone.")


if __name__ == "__main__":
    main()