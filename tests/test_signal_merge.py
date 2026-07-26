"""Tests for merge_signals (app/signals/merge.py).

Covers: broadcasting a per-(location, date) signal row across every
item-level sales row sharing that key, the row-count/cardinality guard
against a signal frame with duplicate (location, date) keys, fill-default
handling for sparse signals (holidays/events), the no-fill-needed case for
a dense signal (weather), and an end-to-end merge -> FeatureStore round
trip.
"""

from datetime import date

import pandas as pd
import pytest

from app.features.store import FeatureStore
from app.features.synthetic import generate_synthetic_sales
from app.signals.merge import merge_signals


def test_merge_broadcasts_signals_onto_every_item_row_sharing_a_date() -> None:
    sales_rows = pd.DataFrame(
        {
            "location": ["storeA", "storeA", "storeA", "storeA"],
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 2),
            ],
            "item": ["burger", "fries", "burger", "fries"],
            "qty_sold": [10.0, 5.0, 8.0, 4.0],
        }
    )
    holiday_df = pd.DataFrame(
        {
            "location": ["storeA"],
            "date": [date(2024, 1, 1)],
            "is_holiday": [1],
            "holiday_name": ["New Year's Day"],
        }
    )
    weather_df = pd.DataFrame(
        {
            "location": ["storeA", "storeA"],
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "temp_c": [5.0, 6.0],
            "precip_mm": [0.0, 2.0],
            "is_rain": [0, 1],
        }
    )

    merged = merge_signals(sales_rows, [holiday_df, weather_df])

    assert len(merged) == len(sales_rows)
    assert {
        "location",
        "date",
        "item",
        "qty_sold",
        "is_holiday",
        "holiday_name",
        "temp_c",
        "precip_mm",
        "is_rain",
    }.issubset(merged.columns)

    # Both item rows on 2024-01-01 (the holiday date) get the SAME broadcast
    # holiday/weather values -- the signal is per-(location, date), not per-item.
    day1 = merged[merged["date"] == date(2024, 1, 1)]
    assert len(day1) == 2
    assert (day1["is_holiday"] == 1).all()
    assert (day1["holiday_name"] == "New Year's Day").all()
    assert (day1["temp_c"] == 5.0).all()

    # 2024-01-02 has no holiday signal row -> filled default, but weather
    # (dense) still attaches its real value for that date.
    day2 = merged[merged["date"] == date(2024, 1, 2)]
    assert len(day2) == 2
    assert (day2["is_holiday"] == 0).all()
    assert (day2["temp_c"] == 6.0).all()
    assert (day2["is_rain"] == 1).all()


def test_cardinality_guard_row_count_preserved_with_sparse_signals() -> None:
    """N items x M dates sales rows merged with signals covering only a
    subset of dates must not multiply rows: len(result) == len(sales_rows)."""
    items = ["burger", "fries", "salad"]
    dates = [date(2024, 1, day) for day in range(1, 5)]  # 4 dates
    sales_rows = pd.DataFrame(
        [
            {"location": "storeA", "date": d, "item": item, "qty_sold": 1.0}
            for d in dates
            for item in items
        ]
    )
    assert len(sales_rows) == 12  # 3 items x 4 dates

    holiday_df = pd.DataFrame(
        {
            "location": ["storeA"],
            "date": [dates[0]],
            "is_holiday": [1],
            "holiday_name": ["New Year's Day"],
        }
    )
    event_df = pd.DataFrame(
        {
            "location": ["storeA"],
            "date": [dates[2]],
            "event_count": [3],
            "event_impact": [0.6],
        }
    )

    merged = merge_signals(sales_rows, [holiday_df, event_df])

    assert len(merged) == len(sales_rows) == 12


def test_merge_raises_on_duplicate_location_date_in_signal_frame() -> None:
    sales_rows = pd.DataFrame(
        {
            "location": ["storeA", "storeA"],
            "date": [date(2024, 1, 1), date(2024, 1, 1)],
            "item": ["burger", "fries"],
            "qty_sold": [10.0, 5.0],
        }
    )
    # Two rows for the SAME (location, date) -- a provider-side bug that
    # would multiply sales rows if not caught.
    dup_frame = pd.DataFrame(
        {
            "location": ["storeA", "storeA"],
            "date": [date(2024, 1, 1), date(2024, 1, 1)],
            "is_holiday": [1, 1],
            "holiday_name": ["A", "B"],
        }
    )

    with pytest.raises(RuntimeError):
        merge_signals(sales_rows, [dup_frame])


def test_fill_defaults_applied_for_dates_with_no_matching_signal_row() -> None:
    sales_rows = pd.DataFrame(
        {
            "location": ["storeA", "storeA"],
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "item": ["burger", "burger"],
            "qty_sold": [1.0, 2.0],
        }
    )
    holiday_df = pd.DataFrame(
        {
            "location": ["storeA"],
            "date": [date(2024, 1, 1)],
            "is_holiday": [1],
            "holiday_name": ["New Year's Day"],
        }
    )
    event_df = pd.DataFrame(
        {
            "location": ["storeA"],
            "date": [date(2024, 1, 1)],
            "event_count": [2],
            "event_impact": [0.4],
        }
    )

    merged = merge_signals(sales_rows, [holiday_df, event_df])

    day2 = merged[merged["date"] == date(2024, 1, 2)].iloc[0]
    assert day2["is_holiday"] == 0
    assert day2["holiday_name"] == ""
    assert day2["event_count"] == 0
    assert day2["event_impact"] == 0.0

    # NaN-free and cast back to int (a left-merge unmatched row upcasts int
    # columns to float; fillna + astype must restore int64).
    assert not merged["is_holiday"].isna().any()
    assert not merged["event_count"].isna().any()
    assert merged["is_holiday"].dtype == "int64"
    assert merged["event_count"].dtype == "int64"


def test_dense_weather_signal_needs_no_fill_and_produces_no_nans() -> None:
    sales_rows = pd.DataFrame(
        {
            "location": ["storeA", "storeA"],
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "item": ["burger", "burger"],
            "qty_sold": [1.0, 2.0],
        }
    )
    # Every sales date has a matching weather row -- dense, no gaps.
    weather_df = pd.DataFrame(
        {
            "location": ["storeA", "storeA"],
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "temp_c": [5.0, 6.0],
            "precip_mm": [0.0, 1.5],
            "is_rain": [0, 1],
        }
    )

    merged = merge_signals(sales_rows, [weather_df])

    assert not merged[["temp_c", "precip_mm", "is_rain"]].isna().any().any()


def test_end_to_end_merge_and_feature_store_round_trip() -> None:
    """generate_synthetic_sales -> merge_signals -> FeatureStore.upsert -> .query.

    Proves row count is preserved through the whole pipeline and both the
    original sales columns and the new signal columns are present on read-back.
    """
    sales_rows = generate_synthetic_sales(
        n_days=5,
        items=["burger", "fries"],
        location="storeA",
        start_date=date(2024, 1, 1),
        seed=1,
    )
    assert len(sales_rows) == 10  # 5 dates x 2 items

    dates = sorted(sales_rows["date"].unique())

    holiday_df = pd.DataFrame(
        {
            "location": ["storeA"],
            "date": [dates[0]],
            "is_holiday": [1],
            "holiday_name": ["New Year's Day"],
        }
    )
    weather_df = pd.DataFrame(
        {
            "location": ["storeA"] * len(dates),
            "date": dates,
            "temp_c": [float(i) for i in range(len(dates))],
            "precip_mm": [0.0] * len(dates),
            "is_rain": [0] * len(dates),
        }
    )
    event_df = pd.DataFrame(
        {
            "location": ["storeA"],
            "date": [dates[-1]],
            "event_count": [1],
            "event_impact": [0.2],
        }
    )

    merged = merge_signals(sales_rows, [holiday_df, weather_df, event_df])
    assert len(merged) == len(sales_rows)

    store = FeatureStore()
    store.upsert(merged)
    result = store.query("storeA", (dates[0], dates[-1]))

    assert {"location", "date", "item", "qty_sold"}.issubset(result.columns)
    assert {
        "is_holiday",
        "holiday_name",
        "temp_c",
        "precip_mm",
        "is_rain",
        "event_count",
        "event_impact",
    }.issubset(result.columns)
    assert len(result) == len(sales_rows)
