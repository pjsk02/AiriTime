"""Tests for CSVConnector (app/connectors/csv_connector.py)."""

from datetime import date
from pathlib import Path

import pytest

from app.connectors.csv_connector import CSVConnector
from app.features.store import FeatureStore


def _write_csv(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_fetch_canonical_columns_happy_path(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        "sales.csv",
        "location,date,item,qty_sold\n"
        "storeA,2024-01-01,burger,10\n"
        "storeA,2024-01-02,fries,5\n",
    )
    df = CSVConnector(csv_path).fetch()

    assert list(df.columns) == ["location", "date", "item", "qty_sold"]
    assert len(df) == 2
    assert df.loc[0, "location"] == "storeA"
    assert df.loc[0, "date"] == date(2024, 1, 1)
    assert df.loc[0, "item"] == "burger"
    assert df.loc[0, "qty_sold"] == 10.0
    assert df.loc[1, "qty_sold"] == 5.0
    assert df["qty_sold"].dtype == "float64"
    assert all(isinstance(d, date) for d in df["date"])


def test_fetch_column_aliases(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        "sales_alias.csv",
        "store,sale_date,menu_item,qty\n"
        "storeB,2024-02-10,salad,7\n",
    )
    df = CSVConnector(csv_path).fetch()

    # Aliased columns still normalize to the canonical schema/order.
    assert list(df.columns) == ["location", "date", "item", "qty_sold"]
    assert len(df) == 1
    assert df.loc[0, "location"] == "storeB"
    assert df.loc[0, "date"] == date(2024, 2, 10)
    assert df.loc[0, "item"] == "salad"
    assert df.loc[0, "qty_sold"] == 7.0


def test_fetch_missing_required_column_raises(tmp_path: Path) -> None:
    # No column matches any qty_sold/qty/quantity/units_sold alias.
    csv_path = _write_csv(
        tmp_path,
        "missing_qty.csv",
        "location,date,item\nstoreA,2024-01-01,burger\n",
    )
    with pytest.raises(ValueError):
        CSVConnector(csv_path).fetch()


def test_fetch_malformed_date_raises(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        "bad_date.csv",
        "location,date,item,qty_sold\nstoreA,not-a-date,burger,10\n",
    )
    with pytest.raises(ValueError):
        CSVConnector(csv_path).fetch()


def test_fetch_malformed_quantity_raises(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        "bad_qty.csv",
        "location,date,item,qty_sold\nstoreA,2024-01-01,burger,not-a-number\n",
    )
    with pytest.raises(ValueError):
        CSVConnector(csv_path).fetch()


def test_fetch_empty_csv_returns_empty_dataframe_with_columns(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, "empty.csv", "location,date,item,qty_sold\n")
    df = CSVConnector(csv_path).fetch()

    assert list(df.columns) == ["location", "date", "item", "qty_sold"]
    assert len(df) == 0


def test_csv_connector_to_feature_store_pipeline(tmp_path: Path) -> None:
    """End-to-end: CSVConnector.fetch() -> FeatureStore.upsert() -> .query()."""
    csv_path = _write_csv(
        tmp_path,
        "pipeline.csv",
        "location,date,item,qty_sold\n"
        "storeA,2024-01-01,burger,10\n"
        "storeA,2024-01-02,burger,12\n"
        "storeA,2024-01-01,fries,4\n",
    )
    rows = CSVConnector(csv_path).fetch()

    store = FeatureStore()
    store.upsert(rows)

    result = store.query("storeA", (date(2024, 1, 1), date(2024, 1, 2)))

    assert {"location", "date", "item", "qty_sold"}.issubset(result.columns)
    assert len(result) == 3
    assert set(result["item"]) == {"burger", "fries"}
    assert set(result["location"]) == {"storeA"}
