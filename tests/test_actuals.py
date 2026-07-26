"""Tests for ActualsStore and join_forecast_actuals (app/learning/actuals.py).

Covers the UPSERT semantics that distinguish ActualsStore from
`ForecastLog` (app/learning/forecast_log.py) -- exactly one true actual
survives per (location, date, item), so a second ingest for the same key
overwrites rather than appending -- plus the round-trip read, and the
full forecast-log -> actuals -> join_forecast_actuals pipeline (the
concrete proof of the phase's "graded forecasts" end state), including
that orphan rows on either side of the join are correctly dropped (inner
join).
"""

from datetime import date, timedelta

import pandas as pd

from app.features.synthetic import DEFAULT_ITEMS, generate_synthetic_sales
from app.features.synthetic_signals import generate_synthetic_future_signals
from app.learning.actuals import ActualsStore, join_forecast_actuals
from app.learning.forecast_log import ForecastLog
from app.models.factor_model import FactorModel

_LOCATION = "demo_location"


def test_ingest_then_read_round_trips_values(tmp_path) -> None:
    store = ActualsStore(tmp_path / "actuals.jsonl")
    d0 = date(2026, 1, 1)
    d1 = date(2026, 1, 2)
    rows = pd.DataFrame(
        [
            {"location": _LOCATION, "date": d0, "item": "burger", "qty_sold": 12.0},
            {"location": _LOCATION, "date": d1, "item": "fries", "qty_sold": 30.5},
        ]
    )
    store.ingest(rows)

    read_back = store.read(_LOCATION, (d0, d1))

    assert len(read_back) == 2
    assert list(read_back.columns) == ["location", "date", "item", "qty_sold"]
    burger_row = read_back[read_back["item"] == "burger"].iloc[0]
    assert burger_row["qty_sold"] == 12.0
    assert burger_row["date"] == d0
    fries_row = read_back[read_back["item"] == "fries"].iloc[0]
    assert fries_row["qty_sold"] == 30.5


def test_ingest_twice_same_key_overwrites_second_value_wins(tmp_path) -> None:
    """Opposite of ForecastLog: exactly one row survives per (location,
    date, item), and it must be the value from the SECOND ingest."""
    store = ActualsStore(tmp_path / "actuals.jsonl")
    d0 = date(2026, 1, 1)

    first = pd.DataFrame([{"location": _LOCATION, "date": d0, "item": "burger", "qty_sold": 10.0}])
    store.ingest(first)

    second = pd.DataFrame([{"location": _LOCATION, "date": d0, "item": "burger", "qty_sold": 99.0}])
    store.ingest(second)

    read_back = store.read(_LOCATION, (d0, d0))

    assert len(read_back) == 1
    assert read_back.iloc[0]["qty_sold"] == 99.0


def test_full_round_trip_forecast_log_actuals_join(tmp_path) -> None:
    """Concrete proof of the phase's "graded forecasts" end state: log
    forecasts via ForecastLog, ingest actuals via ActualsStore for the same
    (and some different) dates/items, then join and confirm the joined
    frame has the expected error columns and correct (inner-join) row
    count -- with at least one orphan on each side excluded.
    """
    history = generate_synthetic_sales(n_days=120, location=_LOCATION, seed=0)
    model = FactorModel()
    model.fit(history)

    reference_today = history["date"].max() + timedelta(days=1)
    date_range = (reference_today + timedelta(days=7), reference_today + timedelta(days=13))
    items = DEFAULT_ITEMS[:2]  # burger, fries

    signals = generate_synthetic_future_signals(_LOCATION, date_range, seed=1)
    frames = []
    for item in items:
        frame = signals.copy()
        frame["item"] = item
        frames.append(frame)
    future_features = pd.concat(frames, ignore_index=True)

    predictions = model.predict(future_features, reference_today=reference_today)

    forecast_log = ForecastLog(tmp_path / "forecast_log.jsonl")
    forecast_log.append(
        predictions, generated_for_date=reference_today, weights_version=1, model_name="factor_model_v1"
    )

    all_dates = sorted(future_features["date"].unique())
    # Actuals covering every forecasted date/item EXCEPT drop one row
    # (an orphan forecast: forecasted but never realized -- e.g. the
    # last day hasn't "happened" yet from the actuals ingestion side).
    orphan_forecast_date = all_dates[-1]
    actual_rows = []
    for d in all_dates:
        for item in items:
            if d == orphan_forecast_date and item == items[0]:
                continue  # this (date, item) forecast has no actual -> orphan forecast row
            actual_rows.append({"location": _LOCATION, "date": d, "item": item, "qty_sold": 15.0})

    # Add an orphan ACTUAL: an actual for a date never forecasted at all.
    orphan_actual_date = reference_today + timedelta(days=100)
    actual_rows.append({"location": _LOCATION, "date": orphan_actual_date, "item": items[0], "qty_sold": 42.0})

    actuals_store = ActualsStore(tmp_path / "actuals.jsonl")
    actuals_store.ingest(pd.DataFrame(actual_rows))

    forecast_rows = forecast_log.read(location=_LOCATION)
    actuals_read_range = (all_dates[0], orphan_actual_date)
    actuals_rows = actuals_store.read(_LOCATION, actuals_read_range)

    joined = join_forecast_actuals(forecast_rows, actuals_rows)

    # Expected columns: forecast columns plus actual_qty, error, abs_error.
    for expected_col in ("actual_qty", "error", "abs_error", "p50", "attribution"):
        assert expected_col in joined.columns
    assert "qty_sold" not in joined.columns  # renamed to actual_qty

    # Row count: total forecasted rows minus the one dropped orphan
    # forecast (the orphan actual contributes no row since it has no
    # matching forecast either).
    expected_rows = len(predictions) - 1
    assert len(joined) == expected_rows

    # The orphan forecast (date, item) must not appear in the join.
    orphan_forecast_present = (
        (joined["date"] == orphan_forecast_date) & (joined["item"] == items[0])
    ).any()
    assert not orphan_forecast_present

    # The orphan actual's date must not appear in the join either.
    assert not (joined["date"] == orphan_actual_date).any()

    # error/abs_error correctness for a sanity-checked row.
    sample = joined.iloc[0]
    assert sample["error"] == sample["actual_qty"] - sample["p50"]
    assert sample["abs_error"] == abs(sample["error"])
