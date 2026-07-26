"""End-to-end non-degradation test for the Phase 6 self-learning loop.

The synthetic sales generator (`app/features/synthetic.py::generate_synthetic_sales`,
confirmed by reading its source) has NO holiday/event/weather effect in its
true data-generating process -- only a per-item baseline, a Fri/Sat/Sun
weekend uplift, and Gaussian noise. This test merges in SYNTHETIC signal
columns (`app/features/synthetic_signals.py::generate_synthetic_future_signals`,
reused here for a HISTORICAL date range -- confirmed by reading its source
that `date_range` is just a plain `(start, end)` tuple with no
future-only/in-the-future requirement anywhere in the function) so
`is_holiday`/`event_impact`/`is_rain` are present and occasionally nonzero
throughout history. Because the model's PRIOR belief
(`DEFAULT_WEIGHTS`: holiday=0.25, event=0.15, rain=0.20) is WRONG here (the
true effect of all three is 0), there is real, learnable signal for
recalibration to correct toward -- holiday_weight in particular should
shrink cycle over cycle as the loop discovers holidays don't actually move
this synthetic venue's demand.

Cycle wiring note (see the task prompt's callout about
`walk_forward_backtest` not being designed with an injectable-weights model
in mind): `walk_forward_backtest`'s `model_cls: type[ForecastModel]` param is
called internally as exactly `model_cls()` (zero arguments) once per
eval_date -- see `app/models/backtest.py`'s `model = model_cls()` line.
`functools.partial(FactorModel, weights=current_weights)` is a legitimate,
minimal-footprint `type[ForecastModel]`-shaped callable for this: calling a
`functools.partial(...)` instance with no further arguments invokes
`FactorModel(weights=current_weights)` exactly as if that had been written
directly, so `walk_forward_backtest` needs no modification and no
subclassing trick -- it is handed the partial in place of a bare class. This
was verified interactively before writing this test (a `functools.partial`
is not a `type`, but Python does not enforce that annotation at runtime, and
the function only ever calls its `model_cls` argument, never instance/class
-checks it).
"""

import functools
from datetime import timedelta

from app.features.synthetic import generate_synthetic_sales
from app.features.synthetic_signals import generate_synthetic_future_signals
from app.learning.actuals import ActualsStore, join_forecast_actuals
from app.learning.forecast_log import ForecastLog
from app.learning.recalibrate import recalibrate
from app.learning.scorecard import Scorecard
from app.learning.weights_store import WeightsStore
from app.models.backtest import walk_forward_backtest
from app.models.factor_model import DEFAULT_WEIGHTS, FactorModel

_LOCATION = "demo_location"
_ITEMS = ["burger", "fries"]
_WINDOW_DAYS = 40
_N_CYCLES = 5
_FIRST_CYCLE_START_OFFSET = 200  # days after history start when cycle 0 begins


def test_recalibration_loop_does_not_degrade_wmape_across_cycles(tmp_path) -> None:
    n_days = _FIRST_CYCLE_START_OFFSET + _WINDOW_DAYS * _N_CYCLES + 10
    history = generate_synthetic_sales(n_days=n_days, location=_LOCATION, seed=0, items=_ITEMS)
    start = history["date"].min()
    end = history["date"].max()

    # Merge synthetic signal columns across the WHOLE history range (a
    # historical reuse of the future-signal generator) -- this is what
    # gives the model a wrongly-nonzero holiday/event/rain prior to correct
    # away from, since generate_synthetic_sales's own qty_sold never
    # actually depends on these columns.
    signals = generate_synthetic_future_signals(_LOCATION, (start, end), seed=7)
    history_merged = history.merge(signals, on=["location", "date"], how="left")

    # Confirm the merged history actually has learnable signal activity
    # (otherwise this test would vacuously pass without exercising
    # recalibrate() at all).
    assert history_merged["is_holiday"].sum() > 0
    assert (history_merged["is_rain"] == 1).sum() > 0
    assert (history_merged["event_impact"] > 0).sum() > 0

    weights_store = WeightsStore(tmp_path / "weights.jsonl", initial_weights=dict(DEFAULT_WEIGHTS))
    scorecard = Scorecard(tmp_path / "scorecard.jsonl")
    forecast_log = ForecastLog(tmp_path / "forecast_log.jsonl")
    actuals_store = ActualsStore(tmp_path / "actuals.jsonl")

    current_weights = dict(DEFAULT_WEIGHTS)
    weights_version = weights_store.latest().version
    weights_history = [dict(current_weights)]

    for cycle in range(_N_CYCLES):
        cycle_today = start + timedelta(days=_FIRST_CYCLE_START_OFFSET + cycle * _WINDOW_DAYS)
        eval_dates = [cycle_today + timedelta(days=i) for i in range(_WINDOW_DAYS)]
        assert eval_dates[-1] <= end  # never walk off the end of the generated history

        # (a) Walk-forward backtest THIS cycle's window using the CURRENT
        # (possibly-recalibrated) weights, via the functools.partial shim
        # described in the module docstring.
        model_cls = functools.partial(FactorModel, weights=current_weights)
        backtest_result = walk_forward_backtest(
            history_merged, model_cls, _LOCATION, eval_dates, min_train_rows=14
        )
        assert backtest_result["n_eval_rows"] > 0, "cycle produced no scoreable rows -- test window misconfigured"

        # (b) Log the resulting wmape into the Scorecard.
        scorecard.append(
            cycle_today, backtest_result["wmape"], backtest_result["skill_vs_naive"], weights_version
        )

        # Log this cycle's actual forecasts (full attribution) and ingest
        # the actuals that have since "arrived", so they can be graded.
        train_history = history_merged[history_merged["date"] < cycle_today]
        cycle_model = FactorModel(weights=current_weights)
        cycle_model.fit(train_history)
        future_features = history_merged[history_merged["date"].isin(eval_dates)].drop(columns=["qty_sold"])
        predictions = cycle_model.predict(future_features, reference_today=cycle_today - timedelta(days=7))
        forecast_log.append(
            predictions, generated_for_date=cycle_today, weights_version=weights_version, model_name="factor_model_v1"
        )

        actual_rows = history_merged[history_merged["date"].isin(eval_dates)][["location", "date", "item", "qty_sold"]]
        actuals_store.ingest(actual_rows)

        # (c) Grade recent forecasts against actuals and recalibrate().
        graded = join_forecast_actuals(
            forecast_log.read(location=_LOCATION), actuals_store.read(_LOCATION, (start, end))
        )
        recalibration = recalibrate(graded, current_weights)

        # (d) Advance current_weights to the recalibrated result and put a
        # new WeightsStore version -- this is what makes the NEXT cycle's
        # backtest genuinely use updated weights, not a no-op loop.
        current_weights = recalibration["new_weights"]
        new_record = weights_store.put(current_weights, reason=f"cycle {cycle} recalibration")
        weights_version = new_record.version
        weights_history.append(dict(current_weights))

    # current_weights must have genuinely changed at least once across the
    # loop -- otherwise this test would just be calling backtest 3+ times
    # with no real recalibration in between, defeating its purpose.
    assert weights_history[0] != weights_history[-1]
    # The wrongly-nonzero holiday prior (true effect is 0) should shrink as
    # recalibration discovers this synthetic venue's holidays don't move
    # demand.
    assert weights_history[-1]["holiday_weight"] < weights_history[0]["holiday_weight"]

    scorecard_df = scorecard.read()
    assert len(scorecard_df) == _N_CYCLES
    wmapes = scorecard_df.sort_values("date")["wmape"].tolist()

    # Hard non-degradation assertion: no later cycle's wmape may exceed the
    # FIRST cycle's (pre-/least-recalibrated) wmape by more than a small
    # tolerance. Deliberately not a strict-monotonic-improvement assertion
    # (noise could occasionally bump wmape slightly cycle to cycle) --
    # non-degradation, not strict improvement, is the property under test.
    baseline_wmape = wmapes[0]
    tolerance = 0.03
    for cycle_index, wmape in enumerate(wmapes):
        assert wmape <= baseline_wmape + tolerance, (
            f"cycle {cycle_index} wmape {wmape} regressed beyond baseline {baseline_wmape} + {tolerance}"
        )
