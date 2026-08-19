"""Loading the raw Olist CSVs.

`load_table` reads one file with the dtypes and date parsing declared in
schema.py. `load_all` returns every table in a dict. `check_integrity`
reports on keys and orphaned foreign keys so data problems surface here
rather than three phases downstream.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import PATHS
from src.data.schema import COLUMN_RENAMES, TABLES, TableSpec

logger = logging.getLogger(__name__)


def _resolve_path(spec: TableSpec, raw_dir: Path | None = None) -> Path:
    directory = raw_dir or PATHS["raw"]
    path = directory / spec.filename
    if not path.exists():
        available = sorted(p.name for p in directory.glob("*.csv"))
        raise FileNotFoundError(
            f"Expected {spec.filename} in {directory}. "
            f"Found: {available or 'no CSV files'}"
        )
    return path


def load_table(name: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Load a single raw table by its schema name."""
    if name not in TABLES:
        raise KeyError(f"Unknown table '{name}'. Known: {sorted(TABLES)}")

    spec = TABLES[name]
    path = _resolve_path(spec, raw_dir)

    df = pd.read_csv(
        path,
        dtype=spec.dtypes or None,
        parse_dates=list(spec.date_columns) or None,
    )
    df = df.rename(columns={k: v for k, v in COLUMN_RENAMES.items() if k in df.columns})

    logger.info("Loaded %s: %d rows x %d cols", name, len(df), df.shape[1])
    return df


def load_all(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load every raw table into a dict keyed by schema name."""
    return {name: load_table(name, raw_dir) for name in TABLES}


def check_integrity(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Audit primary keys, null rates and foreign-key coverage.

    Returns a tidy frame of findings, one row per check, so it can be
    printed in a notebook or asserted against in tests.
    """
    findings: list[dict[str, object]] = []

    for name, df in tables.items():
        spec = TABLES[name]
        findings.append({"table": name, "check": "row_count", "value": len(df), "detail": ""})

        if spec.primary_key:
            dupes = int(df.duplicated(subset=list(spec.primary_key)).sum())
            findings.append(
                {
                    "table": name,
                    "check": "duplicate_primary_keys",
                    "value": dupes,
                    "detail": " + ".join(spec.primary_key),
                }
            )

        null_cols = df.columns[df.isna().any()]
        for col in null_cols:
            findings.append(
                {
                    "table": name,
                    "check": "null_rate",
                    "value": round(float(df[col].isna().mean()), 4),
                    "detail": col,
                }
            )

    # Foreign keys worth verifying before any join.
    fk_checks = [
        ("orders", "customer_id", "customers", "customer_id"),
        ("order_items", "order_id", "orders", "order_id"),
        ("order_items", "product_id", "products", "product_id"),
        ("order_items", "seller_id", "sellers", "seller_id"),
        ("order_payments", "order_id", "orders", "order_id"),
        ("order_reviews", "order_id", "orders", "order_id"),
    ]
    for child, child_col, parent, parent_col in fk_checks:
        if child not in tables or parent not in tables:
            continue
        missing = int(
            (~tables[child][child_col].isin(set(tables[parent][parent_col]))).sum()
        )
        findings.append(
            {
                "table": child,
                "check": "orphaned_foreign_keys",
                "value": missing,
                "detail": f"{child}.{child_col} -> {parent}.{parent_col}",
            }
        )

    return pd.DataFrame(findings)


__all__ = ["load_table", "load_all", "check_integrity"]