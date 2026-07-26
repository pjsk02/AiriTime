"""Tests for walk_forward_backtest (app/models/backtest.py).

Covers the returned dict's shape/sanity on a well-formed input, and --
most importantly -- a direct behavioral proof (not a trust-the-code-comment
assertion) that the no-lookahead guard (`history.date < eval_date`, strict)
actually holds: a spy `ForecastModel` records the max training date it was
ever fit on, and the test asserts every recorded max-seen-date is strictly
before its corresponding eval_date. Also covers the documented
skip-not-crash behavior for an eval_date with insufficient training history.
"""

import math
from datetime import date, timedelta

import pandas as pd

from app.features.synthetic import generate_synthetic_sales
from app.models.backtest import walk_forward_backtest
from app.models.base import ForecastModel
from app.models.factor_model import FactorModel

_LOCATION = "demo_location"


def test_backtest_returns_sane_wmape_and_skill_for_well_formed_input() -> None:
    history = generate_synthetic_sales(
        n_days=40, items=["burger"], location=_LOCATION, seed=0
    )
    start_date = history["date"].min()
    eval_dates = [start_date + timedelta(days=offset) for offset in (20, 21, 22, 23, 24)]

    result = walk_forward_backtest(
        history, FactorModel, _LOCATION, eval_dates, min_train_rows=14
    )

    assert set(result.keys()) == {"wmape", "skill_vs_naive", "n_eval_rows"}
    assert isinstance(result["wmape"], float)
    assert isinstance(result["skill_vs_naive"], float)
    assert isinstance(result["n_eval_rows"], float)

    assert not math.isnan(result["wmape"])
    assert not math.isnan(result["skill_vs_naive"])
    assert result["wmape"] >= 0.0
    # 1 item x 5 eval_dates, none skipped (ample training history, and
    # both the eval-date actual and the naive-baseline actual are present
    # throughout this 40-day continuous history).
    assert result["n_eval_rows"] == 5.0


def test_walk_forward_backtest_never_trains_on_eval_date_or_later() -> None:
    """Direct proof of the no-lookahead guard: a spy model records the max
    training date it was fit on (per fresh instance, via a shared
    class-level list, since `walk_forward_backtest` constructs a NEW
    `model_cls()` per eval_date). Test data is chosen so every eval_date
    has enough training rows, an actual on eval_date, and a naive
    reference 7 days prior -- i.e. none are skipped -- so the recorded
    max-seen-dates line up 1:1, in order, with `eval_dates`.
    """

    class _RecordingModel(ForecastModel):
        recorded_max_dates: list = []

        def __init__(self) -> None:
            self.max_seen_date = None

        def fit(self, history: pd.DataFrame) -> None:
            self.max_seen_date = history["date"].max()
            _RecordingModel.recorded_max_dates.append(self.max_seen_date)

        def predict(self, future_features: pd.DataFrame, reference_today: date) -> pd.DataFrame:
            records = [
                {
                    "location": row.location,
                    "date": row.date,
                    "item": row.item,
                    "p10": 1.0,
                    "p50": 1.0,
                    "p90": 1.0,
                    "attribution": [
                        {
                            "factor": "day_of_week",
                            "direction": "up",
                            "text": "spy",
                            "contribution": 0.0,
                        }
                    ],
                }
                for row in future_features.itertuples(index=False)
            ]
            return pd.DataFrame.from_records(
                records,
                columns=["location", "date", "item", "p10", "p50", "p90", "attribution"],
            )

    history = generate_synthetic_sales(
        n_days=40, items=["burger"], location=_LOCATION, seed=0
    )
    start_date = history["date"].min()
    eval_dates = [start_date + timedelta(days=offset) for offset in (20, 21, 22, 23, 24)]

    result = walk_forward_backtest(
        history, _RecordingModel, _LOCATION, eval_dates, min_train_rows=14
    )

    # Confirms no eval_date was skipped, so recorded_max_dates lines up
    # 1:1, in order, with eval_dates below.
    assert result["n_eval_rows"] == len(eval_dates)
    assert len(_RecordingModel.recorded_max_dates) == len(eval_dates)

    for eval_date, max_seen_date in zip(eval_dates, _RecordingModel.recorded_max_dates):
        assert max_seen_date < eval_date


def test_skips_eval_dates_with_insufficient_training_history() -> None:
    history = generate_synthetic_sales(
        n_days=10, items=["burger"], location=_LOCATION, seed=0
    )
    start_date = history["date"].min()
    # Only 3 days of strictly-prior history are available here -- far
    # fewer than min_train_rows=14 -- so this eval_date must be skipped
    # (not raise, not crash) and produce an all-zero, fully-typed result.
    eval_date = start_date + timedelta(days=3)

    result = walk_forward_backtest(
        history, FactorModel, _LOCATION, [eval_date], min_train_rows=14
    )

    assert result == {"wmape": 0.0, "skill_vs_naive": 0.0, "n_eval_rows": 0.0}
