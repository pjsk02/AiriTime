"""Tests for FeatureStore (app/features/store.py)."""

from datetime import date

import pandas as pd
import pytest

from app.features.store import FeatureStore


def test_upsert_and_query_round_trip() -> None:
    store = FeatureStore()
    rows = pd.DataFrame(
        {
            "location": ["A", "A", "A", "B"],
            "date": [
                date(2024, 1, 3),
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 2),
            ],
            "item": ["fries", "salad", "burger", "burger"],
            "qty_sold": [3, 7, 10, 5],
        }
    )

    store.upsert(rows)
    result = store.query("A", (date(2024, 1, 1), date(2024, 1, 3)))

    assert list(result.columns[:4]) == ["location", "date", "item", "qty_sold"]
    assert len(result) == 3
    assert set(result["location"]) == {"A"}
    # sorted by (date, item): 2024-01-01/burger, 2024-01-01/salad, 2024-01-03/fries
    assert result["date"].tolist() == [
        date(2024, 1, 1),
        date(2024, 1, 1),
        date(2024, 1, 3),
    ]
    assert result["item"].tolist() == ["burger", "salad", "fries"]
    assert result["qty_sold"].tolist() == [10, 7, 3]


def test_upsert_overwrite_updates_key_without_losing_other_columns() -> None:
    store = FeatureStore()

    first = pd.DataFrame(
        {
            "location": ["A"],
            "date": [date(2024, 1, 1)],
            "item": ["burger"],
            "qty_sold": [5],
            "note": ["original"],
        }
    )
    store.upsert(first)

    # Second upsert for the SAME key changes qty_sold and doesn't mention
    # `note` at all -- it must be preserved, not wiped.
    second = pd.DataFrame(
        {
            "location": ["A"],
            "date": [date(2024, 1, 1)],
            "item": ["burger"],
            "qty_sold": [9],
        }
    )
    store.upsert(second)

    result = store.query("A", (date(2024, 1, 1), date(2024, 1, 1)))

    assert len(result) == 1
    assert result.loc[0, "qty_sold"] == 9
    assert result.loc[0, "note"] == "original"


def test_query_with_no_matching_rows_returns_empty_with_columns() -> None:
    store = FeatureStore()
    rows = pd.DataFrame(
        {
            "location": ["A"],
            "date": [date(2024, 1, 1)],
            "item": ["burger"],
            "qty_sold": [5],
        }
    )
    store.upsert(rows)

    # Wrong location entirely.
    result_wrong_location = store.query(
        "does-not-exist", (date(2024, 1, 1), date(2024, 1, 1))
    )
    assert list(result_wrong_location.columns) == ["location", "date", "item", "qty_sold"]
    assert len(result_wrong_location) == 0

    # Right location, date range outside all stored dates.
    result_wrong_range = store.query("A", (date(2025, 1, 1), date(2025, 1, 2)))
    assert list(result_wrong_range.columns) == ["location", "date", "item", "qty_sold"]
    assert len(result_wrong_range) == 0


def test_query_on_empty_store_returns_empty_with_columns() -> None:
    store = FeatureStore()
    result = store.query("A", (date(2024, 1, 1), date(2024, 1, 31)))

    assert list(result.columns) == ["location", "date", "item", "qty_sold"]
    assert len(result) == 0


def test_query_inclusive_date_range_boundaries() -> None:
    store = FeatureStore()
    rows = pd.DataFrame(
        {
            "location": ["A", "A", "A", "A"],
            "date": [
                date(2023, 12, 31),  # just before range - excluded
                date(2024, 1, 1),  # start boundary - included
                date(2024, 1, 5),  # end boundary - included
                date(2024, 1, 6),  # just after range - excluded
            ],
            "item": ["burger", "burger", "burger", "burger"],
            "qty_sold": [1, 2, 3, 4],
        }
    )
    store.upsert(rows)

    result = store.query("A", (date(2024, 1, 1), date(2024, 1, 5)))

    assert len(result) == 2
    assert set(result["date"]) == {date(2024, 1, 1), date(2024, 1, 5)}


def test_upsert_raises_on_missing_key_columns() -> None:
    store = FeatureStore()
    rows = pd.DataFrame(
        {
            "location": ["A"],
            "date": [date(2024, 1, 1)],
            # "item" column is missing entirely.
            "qty_sold": [5],
        }
    )
    with pytest.raises(ValueError):
        store.upsert(rows)


def test_upsert_empty_rows_is_a_noop() -> None:
    store = FeatureStore()
    empty = pd.DataFrame(columns=["location", "date", "item", "qty_sold"])
    store.upsert(empty)

    result = store.query("A", (date(2024, 1, 1), date(2024, 1, 31)))
    assert len(result) == 0
