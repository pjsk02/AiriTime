"""Tests for recalibrate() and its five safety rails (app/learning/recalibrate.py).

Real defaults read from the source: `DEFAULT_MAX_STEP=0.10`,
`DEFAULT_MIN_SAMPLES=5`, `DEFAULT_GENTLE_FRACTION=0.3`,
`DEFAULT_DRIFT_WINDOW=10`, `DEFAULT_DRIFT_THRESHOLD=0.15`,
`DEFAULT_LEARNING_RATE=0.5`.

For a single active recalibratable factor (share_f = 1.0 always), the
per-row normalized signal is `direction_signal / max(abs(p50), eps)` where
`direction_signal = factor_error * sign(contribution)` and
`factor_error = error * share_f = error` (single factor). `raw_relative_delta
= learning_rate * mean(normalized_signal)`; the applied cap is `max_step` if
the mean of the last `drift_window` normalized signals exceeds
`drift_threshold` in absolute value, else `max_step * gentle_fraction`;
`relative_delta = clip(raw_relative_delta, -cap, cap)`;
`new_value = max(0.0, old_value * (1 + relative_delta))`;
`applied_step = abs(new_value - old_value) / old_value` (0.0 if
`old_value == 0.0`).
"""

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from app.features.synthetic import DEFAULT_ITEMS, generate_synthetic_sales
from app.learning.recalibrate import recalibrate
from app.learning.weights_store import WeightsStore
from app.models.factor_model import DEFAULT_WEIGHTS, FactorModel

_LOCATION = "demo_location"


def _holiday_rows(n: int, error: float, contribution: float = 25.0, p50: float = 100.0) -> pd.DataFrame:
    """`n` graded rows, all with the SAME one-directional error for a single
    active recalibratable factor (holiday), on consecutive dates."""
    base_date = date(2026, 1, 1)
    rows = []
    for i in range(n):
        rows.append(
            {
                "location": _LOCATION,
                "item": "burger",
                "date": base_date + timedelta(days=i),
                "p50": p50,
                "actual_qty": p50 + error,
                "error": error,
                "attribution": [
                    {"factor": "holiday", "direction": "up", "text": "x", "contribution": contribution}
                ],
            }
        )
    return pd.DataFrame(rows)


def test_bounded_extreme_signal_is_clamped_not_rejected() -> None:
    """An extreme, consistent one-directional error (huge relative signal)
    would naively demand a giant weight change; with a small max_step, the
    ACTUAL relative change must be clamped to max_step, not rejected
    outright (the weight DOES change -- some nonzero step is applied)."""
    current_weights = dict(DEFAULT_WEIGHTS)
    graded_rows = _holiday_rows(n=12, error=400.0)  # actual = 500 vs p50 = 100, huge miss

    result = recalibrate(graded_rows, current_weights, max_step=0.05)

    old = current_weights["holiday_weight"]
    new = result["new_weights"]["holiday_weight"]
    relative_change = abs(new - old) / old

    assert relative_change <= 0.05 + 1e-9
    assert new != old  # bounded, not rejected -- some nonzero step was applied
    assert result["updates"]["holiday_weight"]["applied_step"] == pytest.approx(0.05, abs=1e-9)


def test_guarded_below_min_samples_leaves_weight_bit_for_bit_unchanged() -> None:
    """A factor active in fewer than min_samples rows must be left
    COMPLETELY unchanged -- not even a zero-sized nudge attempted."""
    current_weights = dict(DEFAULT_WEIGHTS)
    graded_rows = _holiday_rows(n=2, error=50.0)  # only 2 rows, min_samples=5

    result = recalibrate(graded_rows, current_weights, min_samples=5)

    assert result["new_weights"]["holiday_weight"] == current_weights["holiday_weight"]
    update = result["updates"]["holiday_weight"]
    assert update["n_samples"] < 5
    assert update["applied_step"] == 0.0


def test_drift_triggers_full_step_vs_gentle_fraction_for_mild_signal() -> None:
    """Same current_weights/max_step/gentle_fraction, two scenarios for the
    same factor:
      (a) mild, small net error (normalized signal = 3/100 = 0.03, well
          under drift_threshold=0.15) -> drift does NOT fire, applied_step
          = learning_rate * 0.03 = 0.015 (well under gentle cap
          max_step*gentle_fraction = 0.03, so NOT clamped).
      (b) strong, large one-directional error (normalized signal = 40/100
          = 0.40, over drift_threshold=0.15) -> drift DOES fire, applied_step
          = max_step = 0.10 (clamped at the full cap, since the raw signal
          learning_rate*0.40 = 0.20 exceeds it).
    """
    current_weights = dict(DEFAULT_WEIGHTS)
    max_step = 0.10
    gentle_fraction = 0.3

    mild_rows = _holiday_rows(n=12, error=3.0)  # normalized signal 0.03 per row
    strong_rows = _holiday_rows(n=12, error=40.0)  # normalized signal 0.40 per row

    mild_result = recalibrate(mild_rows, current_weights, max_step=max_step, gentle_fraction=gentle_fraction)
    strong_result = recalibrate(strong_rows, current_weights, max_step=max_step, gentle_fraction=gentle_fraction)

    mild_update = mild_result["updates"]["holiday_weight"]
    strong_update = strong_result["updates"]["holiday_weight"]

    assert mild_update["drift"] is False
    assert strong_update["drift"] is True

    gentle_cap = max_step * gentle_fraction
    assert mild_update["applied_step"] < gentle_cap  # not even clamped -- genuinely gentle
    assert strong_update["applied_step"] == pytest.approx(max_step, abs=1e-9)
    assert strong_update["applied_step"] > mild_update["applied_step"]


def test_reversible_rollback_reproduces_exact_old_weights_and_forecast(tmp_path) -> None:
    """Use a real WeightsStore: seed it, put() a recalibrated version,
    confirm latest() reflects it, then roll back and confirm the retrieved
    weights EXACTLY match the pre-recalibration weights -- and, concretely,
    that a FactorModel built from the rolled-back weights reproduces the
    exact original forecast (not just that the dict values match)."""
    store = WeightsStore(tmp_path / "weights.jsonl", initial_weights=dict(DEFAULT_WEIGHTS))
    original_record = store.latest()
    original_weights = dict(original_record.weights)

    recalibrated_weights = {"holiday_weight": 0.35, "event_weight": 0.10, "rain_weight": 0.15}
    store.put(recalibrated_weights, reason="test recalibration")
    assert store.latest().weights == recalibrated_weights

    rolled_back = store.rollback_to(original_record.version)
    assert rolled_back.weights == original_weights
    assert store.latest().weights == original_weights  # rollback is itself the new latest

    # get(<old_version>) alone must also retrieve the exact original weights.
    assert store.get(original_record.version).weights == original_weights

    # Concrete forecast-reproduction proof: fit two FactorModels on the SAME
    # history, one using weights straight from DEFAULT_WEIGHTS, one using
    # weights read back from the store via rollback; predict the SAME
    # future row; results must be numerically identical.
    history = generate_synthetic_sales(n_days=120, location=_LOCATION, seed=0)
    reference_today = history["date"].max() + timedelta(days=1)
    item = DEFAULT_ITEMS[0]
    target_date = reference_today + timedelta(days=7)
    future_row = pd.DataFrame(
        [
            {
                "location": _LOCATION,
                "date": target_date,
                "item": item,
                "is_holiday": 1,
                "holiday_name": "Test Holiday",
                "event_impact": 0.4,
                "is_rain": 1,
            }
        ]
    )

    model_direct = FactorModel(weights=original_weights)
    model_direct.fit(history)
    pred_direct = model_direct.predict(future_row, reference_today=reference_today).iloc[0]

    model_from_store = FactorModel(weights=store.get(original_record.version).weights)
    model_from_store.fit(history)
    pred_from_store = model_from_store.predict(future_row, reference_today=reference_today).iloc[0]

    assert pred_direct["p50"] == pytest.approx(pred_from_store["p50"], abs=1e-12)
    assert pred_direct["p10"] == pytest.approx(pred_from_store["p10"], abs=1e-12)
    assert pred_direct["p90"] == pytest.approx(pred_from_store["p90"], abs=1e-12)


def test_nan_row_does_not_zero_out_or_unbound_a_weight() -> None:
    """Regression test for a real bug an opus-level audit found: `np.clip`
    passes NaN through UNCHANGED (it does not clamp it), and
    `max(0.0, old_value * (1.0 + nan))` evaluates to `0.0` under Python's
    `max` semantics -- so a single row with a non-finite `p50` (or any
    NaN-producing signal) used to silently zero out every factor active in
    that row in one cycle, while misreporting `applied_step` as having
    honored `max_step`. A zeroed weight is also a permanent stuck state
    (0.0 * anything is always 0.0), so this was unrecoverable.

    `_normalized_signals_by_factor` now skips any row whose `p50`,
    `contribution`, or resulting normalized signal is not finite -- treat
    a bad observation as "no evidence," never as "clamp to zero."
    """
    current_weights = dict(DEFAULT_WEIGHTS)
    good_rows = _holiday_rows(n=11, error=5.0)  # mild, well-behaved signal
    nan_row = _holiday_rows(n=1, error=400.0, p50=float("nan"))
    graded_rows = pd.concat([good_rows, nan_row], ignore_index=True)

    result = recalibrate(graded_rows, current_weights, max_step=0.05)

    new = result["new_weights"]["holiday_weight"]
    old = current_weights["holiday_weight"]
    update = result["updates"]["holiday_weight"]

    assert math.isfinite(new)
    assert new != 0.0  # must not have been silently zeroed by the bad row
    assert abs(new - old) / old <= 0.05 + 1e-9  # still respects max_step
    # The NaN row must not even count as a valid sample.
    assert update["n_samples"] == 11


def test_weights_store_get_returns_a_copy_not_a_live_reference(tmp_path) -> None:
    """Regression test for a real bug an opus-level audit found: `get()`/
    `latest()`/`history()` used to hand back the live `WeightsRecord` held
    in `self._records`. `WeightsRecord` being `frozen=True` only stops
    *rebinding* its fields, not mutating the dict object its `.weights`
    field points to -- so `store.get(1).weights["x"] = 999` used to
    silently corrupt version 1's permanent history in place. A later
    `rollback_to(1)` would then replay the tampered value, breaking the
    "any past version stays retrievable forever" guarantee.
    """
    store = WeightsStore(tmp_path / "weights.jsonl", initial_weights=dict(DEFAULT_WEIGHTS))

    fetched = store.get(1)
    fetched.weights["holiday_weight"] = 999.0  # mutate the dict the caller got back

    assert store.get(1).weights["holiday_weight"] == DEFAULT_WEIGHTS["holiday_weight"]
    assert store.latest().weights["holiday_weight"] == DEFAULT_WEIGHTS["holiday_weight"]

    # Same property for latest()'s and history()'s returned records.
    store.latest().weights["holiday_weight"] = -1.0
    for record in store.history():
        record.weights["holiday_weight"] = -2.0
    assert store.get(1).weights["holiday_weight"] == DEFAULT_WEIGHTS["holiday_weight"]

    rolled_back = store.rollback_to(1)
    assert rolled_back.weights["holiday_weight"] == DEFAULT_WEIGHTS["holiday_weight"]


def test_monotonic_safety_net_holds_after_recalibration_for_dynamic_weights() -> None:
    """p10 <= p50 <= p90 is guaranteed "by construction" in FactorModel.predict
    regardless of the weights dict passed in (the guarantee comes from
    spread >= 0 and Z10 <= 0 <= Z90, entirely independent of the
    holiday/event/rain weight values) -- so it must keep holding even for
    NEWLY RECALIBRATED (dynamic) weights, not just the fixed module
    constants Phase 4's own tests (tests/test_factor_model.py) exercised.
    """
    current_weights = dict(DEFAULT_WEIGHTS)
    graded_rows = _holiday_rows(n=12, error=40.0)  # drift-triggering signal
    result = recalibrate(graded_rows, current_weights)
    new_weights = result["new_weights"]
    assert new_weights != current_weights  # sanity: recalibration actually moved something

    history = generate_synthetic_sales(n_days=120, location=_LOCATION, seed=0)
    model = FactorModel(weights=new_weights)
    model.fit(history)

    reference_today = history["date"].max() + timedelta(days=1)
    item = DEFAULT_ITEMS[0]
    dates = [reference_today + timedelta(days=offset) for offset in range(7, 14)]

    rows = [
        {
            "location": _LOCATION,
            "date": d,
            "item": item,
            "is_holiday": 0,
            "holiday_name": "",
            "event_impact": 0.0,
            "is_rain": 0,
        }
        for d in dates
    ]
    # Extreme/edge-case row: holiday AND rain AND high event_impact all on
    # the same row, at the +13 (max) horizon offset.
    rows.append(
        {
            "location": _LOCATION,
            "date": dates[-1],
            "item": item,
            "is_holiday": 1,
            "holiday_name": "Extreme Combined Holiday",
            "event_impact": 1.0,
            "is_rain": 1,
        }
    )
    future_features = pd.DataFrame(rows)

    predictions = model.predict(future_features, reference_today=reference_today)

    assert len(predictions) == len(future_features)
    for row in predictions.itertuples(index=False):
        assert row.p10 <= row.p50
        assert row.p50 <= row.p90
