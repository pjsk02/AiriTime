"""Walk-forward backtest: wMAPE and skill vs a same-day-last-week naive baseline (PRD.md section 5, 14 phase 4).

Validates a registered `ForecastModel` on a venue's own history before it
is trusted (PRD.md section 5's "Backtesting (walk-forward)"): for each
evaluation date, fit a *fresh* model instance only on rows strictly before
that date, predict, and compare against the actual `qty_sold` already
recorded in `history` for that date -- which the freshly-fit model never
saw. This is the no-lookahead guarantee; see `walk_forward_backtest`'s
`train_history` line for exactly where it is enforced.
"""

from datetime import date, timedelta

import pandas as pd

from app.models.base import ForecastModel


def walk_forward_backtest(
    history: pd.DataFrame,
    model_cls: type[ForecastModel],
    location: str,
    eval_dates: list[date],
    min_train_rows: int = 14,
) -> dict[str, float]:
    """Walk-forward backtest: wMAPE and skill vs a same-day-last-week naive baseline.

    For each date in `eval_dates` (evaluated independently, in order): fit
    a fresh `model_cls()` ONLY on `history` rows strictly before that date
    (`history.date < eval_date` -- never `>=`, this is the leakage guard),
    predict for every item present in `history` for that (location,
    eval_date), and compare against the REAL actual `qty_sold` already
    recorded in `history` for that date (which the model itself never
    saw). A naive same-day-last-week baseline forecasts `eval_date` using
    the actual `qty_sold` from `eval_date - 7 days` for the same
    (location, item) -- also read directly from `history`, never from the
    model.

    wMAPE = sum(|actual - forecast|) / sum(actual), aggregated over every
    (item, eval_date) pair evaluated (skips eval dates with fewer than
    `min_train_rows` of training history available, or with no naive
    reference 7 days prior).

    skill = 1 - (model_wmape / naive_wmape); positive skill means the
    model beats the naive baseline.

    Args:
        history: FeatureStore-shaped rows (`location, date, item,
            qty_sold`, plus signal columns) spanning both the training
            period and every `eval_dates` entry -- i.e. actuals for the
            "future" dates must already be present in `history` for this
            backtest to score against them.
        model_cls: a `ForecastModel` subclass; a *new instance* is
            constructed and fit for every eval date, so no state leaks
            between eval dates either.
        location: restaurant/venue identifier to filter `history` on.
        eval_dates: dates to evaluate, each treated independently.
        min_train_rows: minimum number of strictly-prior training rows
            required before an eval date is scored; dates with less
            history are skipped rather than fit on too little data.

    Returns:
        {"wmape": float, "skill_vs_naive": float, "n_eval_rows": float}
        -- all zero if no eval date produced any scoreable rows.
    """
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"]).dt.date
    location_history = history[history["location"] == location]

    total_abs_error = 0.0
    total_actual = 0.0
    naive_abs_error = 0.0
    naive_actual = 0.0
    n_eval_rows = 0

    for eval_date in eval_dates:
        # --- Leakage guard ---
        # Strictly less-than: the freshly-fit model may only ever see rows
        # dated before the day it is being asked to predict. This one line
        # is the entire no-lookahead safety property of this backtest --
        # everything else is bookkeeping around it.
        train_history = location_history[location_history["date"] < eval_date]
        if len(train_history) < min_train_rows:
            continue

        eval_rows = location_history[location_history["date"] == eval_date]
        if eval_rows.empty:
            continue

        naive_date = eval_date - timedelta(days=7)
        naive_rows = location_history[location_history["date"] == naive_date]
        if naive_rows.empty:
            continue

        # Restrict to items with existing training history: an item that
        # first appears exactly on eval_date (no prior rows) has no fitted
        # state for `model.predict` to use, and `ForecastModel.predict` is
        # documented to raise rather than silently extrapolate for such a
        # combination. In every dataset this backtest is exercised against
        # today (the synthetic generator's fixed 4-item menu across all
        # days), this is identical to using every item in `eval_rows`.
        items = sorted(set(train_history["item"]) & set(eval_rows["item"]))
        if not items:
            continue

        model = model_cls()
        model.fit(train_history)

        future_features = eval_rows[eval_rows["item"].isin(items)].drop(columns=["qty_sold"])

        # `reference_today` only drives band width (p10/p90 spread) inside
        # `predict` -- this backtest scores p50 only, so the exact
        # reference date used for the horizon-offset calculation has no
        # effect on wMAPE/skill. Offsetting by 7 days keeps the call
        # shaped like a real +7..+13 forecast (offset == 7) rather than
        # passing `reference_today == eval_date` (offset == 0), which
        # would be a nonsensical horizon for this model but still score
        # identically since only p50 is compared below.
        predictions = model.predict(future_features, reference_today=eval_date - timedelta(days=7))

        actual_by_item = eval_rows.set_index("item")["qty_sold"]
        naive_by_item = naive_rows.set_index("item")["qty_sold"]

        for row in predictions.itertuples(index=False):
            item = row.item
            if item not in actual_by_item.index or item not in naive_by_item.index:
                continue

            actual = float(actual_by_item.loc[item])
            forecast = float(row.p50)
            naive_forecast = float(naive_by_item.loc[item])

            total_abs_error += abs(actual - forecast)
            total_actual += actual
            naive_abs_error += abs(actual - naive_forecast)
            naive_actual += actual
            n_eval_rows += 1

    if n_eval_rows == 0 or total_actual == 0.0:
        return {"wmape": 0.0, "skill_vs_naive": 0.0, "n_eval_rows": float(n_eval_rows)}

    wmape = total_abs_error / total_actual
    naive_wmape = naive_abs_error / naive_actual if naive_actual > 0 else 0.0
    skill_vs_naive = (1.0 - (wmape / naive_wmape)) if naive_wmape > 0 else 0.0

    return {
        "wmape": wmape,
        "skill_vs_naive": skill_vs_naive,
        "n_eval_rows": float(n_eval_rows),
    }
