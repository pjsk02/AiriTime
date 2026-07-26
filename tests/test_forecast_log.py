"""Tests for ForecastLog (app/learning/forecast_log.py).

Covers the append-only LOG semantics that distinguish it from
`ActualsStore` (app/learning/actuals.py) -- multiple entries legitimately
survive for the SAME (location, date, item) across different
`generated_for_date` values, since the rolling +7..+13 horizon re-forecasts
the same calendar day daily as it approaches -- plus the read-back shape
(attribution deserializes to a real Python list of dicts, not a JSON
string), location/date_range filtering, and the empty-log case.
"""

from datetime import date, timedelta

import pandas as pd

from app.features.synthetic import DEFAULT_ITEMS, generate_synthetic_sales
from app.features.synthetic_signals import generate_synthetic_future_signals
from app.learning.forecast_log import ForecastLog
from app.models.factor_model import FactorModel

_LOCATION = "demo_location"
_OTHER_LOCATION = "other_location"


def _future_features_for_items(location: str, items: list, date_range: tuple, seed: int = 0) -> pd.DataFrame:
    signals = generate_synthetic_future_signals(location, date_range, seed=seed)
    frames = []
    for item in items:
        item_frame = signals.copy()
        item_frame["item"] = item
        frames.append(item_frame)
    return pd.concat(frames, ignore_index=True)


def _fitted_model() -> tuple[FactorModel, pd.DataFrame, date]:
    history = generate_synthetic_sales(n_days=120, location=_LOCATION, seed=0)
    model = FactorModel()
    model.fit(history)
    reference_today = history["date"].max() + timedelta(days=1)
    return model, history, reference_today


def test_append_and_read_round_trips_rows_columns_and_attribution(tmp_path) -> None:
    model, _history, reference_today = _fitted_model()
    date_range = (reference_today + timedelta(days=7), reference_today + timedelta(days=13))
    future_features = _future_features_for_items(_LOCATION, DEFAULT_ITEMS, date_range, seed=1)
    predictions = model.predict(future_features, reference_today=reference_today)

    log = ForecastLog(tmp_path / "forecast_log.jsonl")
    log.append(
        predictions,
        generated_for_date=reference_today,
        weights_version=1,
        model_name="factor_model_v1",
    )

    df = log.read()

    assert len(df) == len(predictions)
    assert list(df.columns) == [
        "location",
        "date",
        "item",
        "generated_for_date",
        "weights_version",
        "model_name",
        "p10",
        "p50",
        "p90",
        "attribution",
    ]

    # attribution must deserialize to a real Python list of dicts, not a
    # JSON string, with the original factor/direction/text/contribution
    # values intact.
    original_by_key = {
        (row.location, row.date, row.item): row.attribution for row in predictions.itertuples(index=False)
    }
    for row in df.itertuples(index=False):
        assert isinstance(row.attribution, list)
        assert all(isinstance(entry, dict) for entry in row.attribution)
        original = original_by_key[(row.location, row.date, row.item)]
        assert row.attribution == original
        for entry in row.attribution:
            assert set(entry.keys()) == {"factor", "direction", "text", "contribution"}


def test_append_twice_same_key_different_generated_for_date_retains_both(tmp_path) -> None:
    """The key distinguishing property vs ActualsStore: this is a LOG, not
    an upsert store -- re-forecasting the same (location, date, item) on a
    later day (as the rolling horizon approaches) must not overwrite the
    earlier entry."""
    model, _history, reference_today = _fitted_model()
    item = DEFAULT_ITEMS[0]
    target_date = reference_today + timedelta(days=13)

    # First forecast run: target_date is +13 out.
    features_first = generate_synthetic_future_signals(_LOCATION, (target_date, target_date), seed=1)
    features_first["item"] = item
    predictions_first = model.predict(features_first, reference_today=reference_today)

    # Second forecast run, a week later: same target_date, now only +6 out.
    generated_for_date_second = reference_today + timedelta(days=7)
    features_second = generate_synthetic_future_signals(_LOCATION, (target_date, target_date), seed=2)
    features_second["item"] = item
    predictions_second = model.predict(features_second, reference_today=generated_for_date_second)

    log = ForecastLog(tmp_path / "forecast_log.jsonl")
    log.append(predictions_first, generated_for_date=reference_today, weights_version=1, model_name="factor_model_v1")
    log.append(
        predictions_second, generated_for_date=generated_for_date_second, weights_version=1, model_name="factor_model_v1"
    )

    df = log.read(location=_LOCATION, date_range=(target_date, target_date))

    assert len(df) == 2
    generated_dates = sorted(df["generated_for_date"].tolist())
    assert generated_dates == sorted([reference_today, generated_for_date_second])
    # Both rows share the same (location, date, item) key.
    assert (df["location"] == _LOCATION).all()
    assert (df["date"] == target_date).all()
    assert (df["item"] == item).all()


def test_read_filters_by_location_and_date_range(tmp_path) -> None:
    model, _history, reference_today = _fitted_model()
    date_range = (reference_today + timedelta(days=7), reference_today + timedelta(days=13))
    future_features = _future_features_for_items(_LOCATION, DEFAULT_ITEMS, date_range, seed=1)
    predictions = model.predict(future_features, reference_today=reference_today)

    log = ForecastLog(tmp_path / "forecast_log.jsonl")
    log.append(predictions, generated_for_date=reference_today, weights_version=1, model_name="factor_model_v1")

    # location filter: a location never logged should yield nothing.
    other_df = log.read(location=_OTHER_LOCATION)
    assert other_df.empty

    same_location_df = log.read(location=_LOCATION)
    assert len(same_location_df) == len(predictions)

    # date_range filter: inclusive on both ends, restricting to a single day.
    single_day = date_range[0]
    day_df = log.read(location=_LOCATION, date_range=(single_day, single_day))
    assert len(day_df) == len(DEFAULT_ITEMS)
    assert (day_df["date"] == single_day).all()

    # A date range entirely before any logged date returns nothing.
    empty_range_df = log.read(date_range=(single_day - timedelta(days=100), single_day - timedelta(days=50)))
    assert empty_range_df.empty


def test_fresh_empty_log_read_returns_empty_correctly_shaped(tmp_path) -> None:
    log = ForecastLog(tmp_path / "never_written.jsonl")

    df = log.read()

    assert df.empty
    assert list(df.columns) == [
        "location",
        "date",
        "item",
        "generated_for_date",
        "weights_version",
        "model_name",
        "p10",
        "p50",
        "p90",
        "attribution",
    ]

    # Filtering an empty/nonexistent log must not crash either.
    filtered = log.read(location=_LOCATION, date_range=(date(2020, 1, 1), date(2020, 12, 31)))
    assert filtered.empty
