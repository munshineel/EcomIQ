"""Data layer: schema, loading, integrity.

These tests assert the dataset facts every downstream decision rests on. If
Olist ever reissues the file with different contents, these fail loudly rather
than letting the models quietly train on different data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.load import check_integrity, load_table
from src.data.schema import COLUMN_RENAMES, TABLES
from tests.conftest import needs_raw

EXPECTED_ROWS = {
    "customers": 99_441, "orders": 99_441, "order_items": 112_650,
    "order_payments": 103_886, "order_reviews": 99_224, "products": 32_951,
    "sellers": 3_095, "category_translation": 71,
}


def test_schema_declares_nine_tables():
    assert len(TABLES) == 9


def test_every_spec_has_a_filename():
    for name, spec in TABLES.items():
        assert spec.filename.endswith(".csv"), name


@needs_raw
@pytest.mark.parametrize("name,expected", EXPECTED_ROWS.items())
def test_row_counts(tables, name, expected):
    assert len(tables[name]) == expected


@needs_raw
def test_misspelled_columns_are_renamed(tables):
    """Source files ship 'lenght' rather than 'length'."""
    cols = tables["products"].columns
    for wrong, right in COLUMN_RENAMES.items():
        assert wrong not in cols
        assert right in cols


@needs_raw
def test_date_columns_parsed_as_datetime(tables):
    for name, spec in TABLES.items():
        for col in spec.date_columns:
            assert pd.api.types.is_datetime64_any_dtype(tables[name][col]), f"{name}.{col}"


@needs_raw
def test_no_orphaned_foreign_keys(tables):
    audit = check_integrity(tables)
    orphans = audit[audit["check"] == "orphaned_foreign_keys"]
    assert (orphans["value"] == 0).all(), orphans[orphans["value"] > 0].to_dict("records")


@needs_raw
def test_review_id_is_not_unique(tables):
    """Documents a known source defect. Reviews must be keyed by order_id, not
    review_id -- which is what prepare.py does.

    Two distinct counts, easy to confuse: 789 distinct review_id values are
    reused, producing 814 duplicate rows (a few ids appear three times).
    """
    rv = tables["order_reviews"]
    duplicate_rows = rv["review_id"].duplicated().sum()
    distinct_reused = rv.loc[rv["review_id"].duplicated(keep=False), "review_id"].nunique()

    assert duplicate_rows == 814
    assert distinct_reused == 789
    # And they genuinely point at different orders, so they are not row dupes.
    reused = rv[rv["review_id"].duplicated(keep=False)]
    assert reused.groupby("review_id")["order_id"].nunique().gt(1).all()


@needs_raw
def test_missing_file_raises_with_a_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Expected olist_orders_dataset.csv"):
        load_table("orders", raw_dir=tmp_path)


def test_unknown_table_raises():
    with pytest.raises(KeyError, match="Unknown table"):
        load_table("not_a_table")