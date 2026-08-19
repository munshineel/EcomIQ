"""Schema definitions for the Olist Brazilian E-Commerce dataset.

Nine CSV files, joined through a small set of surrogate keys. Declaring the
schema in one place means loaders can parse dates and dtypes correctly on
the first pass, and any missing or renamed file fails loudly and early.

Key relationships
-----------------
    customers.customer_id           <- orders.customer_id          (1:1)
    customers.customer_unique_id    -- the *real* person; one person may
                                       hold several customer_id values
    orders.order_id                 <- order_items.order_id        (1:N)
    orders.order_id                 <- order_payments.order_id     (1:N)
    orders.order_id                 <- order_reviews.order_id      (1:N)
    products.product_id             <- order_items.product_id      (1:N)
    sellers.seller_id               <- order_items.seller_id       (1:N)

Note that `customer_id` is order-scoped, not person-scoped. Any customer
level aggregation must group by `customer_unique_id` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableSpec:
    """Everything needed to load and validate one raw CSV."""

    name: str
    filename: str
    primary_key: tuple[str, ...]
    date_columns: tuple[str, ...] = ()
    dtypes: dict[str, str] = field(default_factory=dict)
    description: str = ""


TABLES: dict[str, TableSpec] = {
    "customers": TableSpec(
        name="customers",
        filename="olist_customers_dataset.csv",
        primary_key=("customer_id",),
        dtypes={
            "customer_id": "string",
            "customer_unique_id": "string",
            "customer_zip_code_prefix": "string",
            "customer_city": "string",
            "customer_state": "category",
        },
        description="One row per order-scoped customer key.",
    ),
    "orders": TableSpec(
        name="orders",
        filename="olist_orders_dataset.csv",
        primary_key=("order_id",),
        date_columns=(
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ),
        dtypes={
            "order_id": "string",
            "customer_id": "string",
            "order_status": "category",
        },
        description="Order header with lifecycle timestamps.",
    ),
    "order_items": TableSpec(
        name="order_items",
        filename="olist_order_items_dataset.csv",
        primary_key=("order_id", "order_item_id"),
        date_columns=("shipping_limit_date",),
        dtypes={
            "order_id": "string",
            "order_item_id": "int16",
            "product_id": "string",
            "seller_id": "string",
            "price": "float64",
            "freight_value": "float64",
        },
        description="Line items. Revenue lives here, not in orders.",
    ),
    "order_payments": TableSpec(
        name="order_payments",
        filename="olist_order_payments_dataset.csv",
        primary_key=("order_id", "payment_sequential"),
        dtypes={
            "order_id": "string",
            "payment_sequential": "int16",
            "payment_type": "category",
            "payment_installments": "int16",
            "payment_value": "float64",
        },
        description="One row per payment instrument used on an order.",
    ),
    "order_reviews": TableSpec(
        name="order_reviews",
        filename="olist_order_reviews_dataset.csv",
        primary_key=("review_id",),
        date_columns=("review_creation_date", "review_answer_timestamp"),
        dtypes={
            "review_id": "string",
            "order_id": "string",
            "review_score": "int8",
            "review_comment_title": "string",
            "review_comment_message": "string",
        },
        description="Review scores; free-text comments are Portuguese and often null.",
    ),
    "products": TableSpec(
        name="products",
        filename="olist_products_dataset.csv",
        primary_key=("product_id",),
        dtypes={
            "product_id": "string",
            "product_category_name": "string",
            # Misspelled as 'lenght' in the source files. Renamed on load.
            "product_name_lenght": "float32",
            "product_description_lenght": "float32",
            "product_photos_qty": "float32",
            "product_weight_g": "float32",
            "product_length_cm": "float32",
            "product_height_cm": "float32",
            "product_width_cm": "float32",
        },
        description="Product attributes. No product name or description text, only lengths.",
    ),
    "sellers": TableSpec(
        name="sellers",
        filename="olist_sellers_dataset.csv",
        primary_key=("seller_id",),
        dtypes={
            "seller_id": "string",
            "seller_zip_code_prefix": "string",
            "seller_city": "string",
            "seller_state": "category",
        },
        description="Marketplace sellers.",
    ),
    "geolocation": TableSpec(
        name="geolocation",
        filename="olist_geolocation_dataset.csv",
        primary_key=(),  # No unique key; many rows per zip prefix.
        dtypes={
            "geolocation_zip_code_prefix": "string",
            "geolocation_lat": "float64",
            "geolocation_lng": "float64",
            "geolocation_city": "string",
            "geolocation_state": "category",
        },
        description="Lat/lng points per zip prefix. Needs aggregating before joining.",
    ),
    "category_translation": TableSpec(
        name="category_translation",
        filename="product_category_name_translation.csv",
        primary_key=("product_category_name",),
        dtypes={
            "product_category_name": "string",
            "product_category_name_english": "string",
        },
        description="Portuguese to English category names.",
    ),
}

# Source files misspell two columns; normalise on load.
COLUMN_RENAMES: dict[str, str] = {
    "product_name_lenght": "product_name_length",
    "product_description_lenght": "product_description_length",
}

TABLE_NAMES: tuple[str, ...] = tuple(TABLES)

__all__ = ["TableSpec", "TABLES", "TABLE_NAMES", "COLUMN_RENAMES"]