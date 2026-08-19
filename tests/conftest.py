"""Shared fixtures. Session-scoped because loading the raw CSVs is the slowest
thing in the suite and nothing mutates them."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import PATHS
from src.data.load import load_all
from src.data.prepare import build_customer_frame, build_order_frame

RAW_PRESENT = (PATHS["raw"] / "olist_orders_dataset.csv").exists()
needs_raw = pytest.mark.skipif(not RAW_PRESENT, reason="raw Olist CSVs not in data/raw")


@pytest.fixture(scope="session")
def tables() -> dict[str, pd.DataFrame]:
    return load_all()


@pytest.fixture(scope="session")
def order_frame(tables) -> pd.DataFrame:
    return build_order_frame(tables)


@pytest.fixture(scope="session")
def customer_frame(order_frame) -> pd.DataFrame:
    return build_customer_frame(order_frame)