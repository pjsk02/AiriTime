"""Tests for FactorModel (app/models/factor_model.py).

Covers the properties `predict()` documents as guaranteed "by construction"
(p10<=p50<=p90 for every row, band widening with horizon offset, the
weather-only contribution cap), the exact attribution shape/ordering fed to
the owner-facing "why", and the refuse-to-extrapolate ValueError for an
(location, item) never seen by `fit()`.
"""

from datetime import timedelta

import pandas as pd
import pytest

from app.features.synthetic import DEFAULT_ITEMS, generate_synthetic_sales
from app.features.synthetic_signals import generate_synthetic_future_signals
from app.models.base import MODEL_REGISTRY
from app.models.factor_model import FactorModel, WEATHER_CONTRIBUTION_CAP

_LOCATION = "demo_location"


def _future_features_for_items(
    location: str, items: list[str], date_range: tuple, seed: int = 0
) -> pd.DataFrame:
    """Cross-join `generate_synthetic_future_signals`'s per-date rows with
    `items`, since `FactorModel.predict` expects one row per
    (location, date, item) but the synthetic signal generator is
    location/date-only (no item dimension)."""
    signals = generate_synthetic_future_signals(location, date_range, seed=seed)
    frames = []
    for item in items:
        item_frame = signals.copy()
        item_frame["item"] = item
        frames.append(item_frame)
    return pd.concat(frames, ignore_index=True)


def test_registered_in_model_registry() -> None:
    assert MODEL_REGISTRY["factor_model_v1"] is FactorModel


def test_predict_p10_le_p50_le_p90_for_every_row() -> None:
    history = generate_synthetic_sales(n_days=120, location=_LOCATION, seed=0)
    model = FactorModel()
    model.fit(history)

    reference_today = history["date"].max() + timedelta(days=1)
    date_range = (
        reference_today + timedelta(days=7),
        reference_today + timedelta(days=13),
    )
    future_features = _future_features_for_items(
        _LOCATION, DEFAULT_ITEMS, date_range, seed=1
    )

    predictions = model.predict(future_features, reference_today=reference_today)

    assert len(predictions) == len(future_features)
    for row in predictions.itertuples(index=False):
        assert row.p10 <= row.p50
        assert row.p50 <= row.p90


def test_band_widens_with_horizon_offset() -> None:
    history = generate_synthetic_sales(n_days=120, location=_LOCATION, seed=0)
    model = FactorModel()
    model.fit(history)

    reference_today = history["date"].max() + timedelta(days=1)
    item = DEFAULT_ITEMS[0]
    near_date = reference_today + timedelta(days=7)
    far_date = reference_today + timedelta(days=13)

    features = _future_features_for_items(
        _LOCATION, [item], (near_date, far_date), seed=1
    )
    near_row = features[features["date"] == near_date]
    far_row = features[features["date"] == far_date]

    near_pred = model.predict(near_row, reference_today=reference_today).iloc[0]
    far_pred = model.predict(far_row, reference_today=reference_today).iloc[0]

    near_spread = near_pred["p90"] - near_pred["p10"]
    far_spread = far_pred["p90"] - far_pred["p10"]

    # far_date is 6 days further past HORIZON_FLOOR than near_date, so its
    # widening factor (1 + 0.05*6 = 1.3x) always exceeds near_date's (1.0x)
    # by more than the +/-15% weekend multiplier swing could ever offset,
    # regardless of which of the two dates lands on a weekend.
    assert far_spread > near_spread


def test_weather_contribution_is_capped_but_holiday_is_not() -> None:
    history = generate_synthetic_sales(n_days=120, location=_LOCATION, seed=0)
    model = FactorModel()
    model.fit(history)

    item = DEFAULT_ITEMS[0]
    level = model._groups[(_LOCATION, item)].level

    reference_today = history["date"].max() + timedelta(days=1)
    target_date = reference_today + timedelta(days=7)

    weather_row = pd.DataFrame(
        [
            {
                "location": _LOCATION,
                "date": target_date,
                "item": item,
                "is_holiday": 0,
                "holiday_name": "",
                "precip_mm": 500.0,
                "is_rain": 1,
                "event_impact": 0.0,
            }
        ]
    )
    holiday_row = pd.DataFrame(
        [
            {
                "location": _LOCATION,
                "date": target_date,
                "item": item,
                "is_holiday": 1,
                "holiday_name": "Extreme Test Holiday",
                "precip_mm": 0.0,
                "is_rain": 0,
                "event_impact": 0.0,
            }
        ]
    )

    weather_pred = model.predict(weather_row, reference_today=reference_today).iloc[0]
    holiday_pred = model.predict(holiday_row, reference_today=reference_today).iloc[0]

    weather_attr = next(
        entry for entry in weather_pred["attribution"] if entry["factor"] == "weather"
    )
    holiday_attr = next(
        entry for entry in holiday_pred["attribution"] if entry["factor"] == "holiday"
    )

    weather_cap_amount = WEATHER_CONTRIBUTION_CAP * level
    assert abs(weather_attr["contribution"]) <= weather_cap_amount + 1e-9

    # Asymmetry: HOLIDAY_WEIGHT (0.25) exceeds WEATHER_CONTRIBUTION_CAP
    # (0.05), so an extreme holiday signal is NOT squeezed the way an
    # extreme rain signal is.
    assert abs(holiday_attr["contribution"]) > weather_cap_amount


def test_attribution_structure_and_ordering() -> None:
    history = generate_synthetic_sales(n_days=120, location=_LOCATION, seed=0)
    model = FactorModel()
    model.fit(history)

    item = DEFAULT_ITEMS[0]
    reference_today = history["date"].max() + timedelta(days=1)
    target_date = reference_today + timedelta(days=7)

    row = pd.DataFrame(
        [
            {
                "location": _LOCATION,
                "date": target_date,
                "item": item,
                "is_holiday": 1,
                "holiday_name": "Test Holiday",
                "precip_mm": 10.0,
                "is_rain": 1,
                "event_impact": 0.8,
            }
        ]
    )

    prediction = model.predict(row, reference_today=reference_today).iloc[0]
    attribution = prediction["attribution"]

    assert isinstance(attribution, list)
    factors = [entry["factor"] for entry in attribution]
    assert "day_of_week" in factors  # always present, per ForecastModel.predict's contract

    for entry in attribution:
        assert set(entry.keys()) == {"factor", "direction", "text", "contribution"}
        assert entry["direction"] in ("up", "down")
        # day_of_week uses a >=0 up/down split (zero counts as "up");
        # holiday/event/weather use a strict >0 split (zero counts as
        # "down") -- see FactorModel.predict's source for both rules.
        if entry["factor"] == "day_of_week":
            expected_direction = "up" if entry["contribution"] >= 0 else "down"
        else:
            expected_direction = "up" if entry["contribution"] > 0 else "down"
        assert entry["direction"] == expected_direction

    abs_contributions = [abs(entry["contribution"]) for entry in attribution]
    assert abs_contributions == sorted(abs_contributions, reverse=True)


def test_predict_raises_for_unseen_location_item() -> None:
    history = generate_synthetic_sales(
        n_days=120, items=["burger"], location=_LOCATION, seed=0
    )
    model = FactorModel()
    model.fit(history)

    reference_today = history["date"].max() + timedelta(days=1)
    target_date = reference_today + timedelta(days=7)

    unseen_row = pd.DataFrame(
        [{"location": _LOCATION, "date": target_date, "item": "never_seen_item"}]
    )

    with pytest.raises(ValueError):
        model.predict(unseen_row, reference_today=reference_today)
