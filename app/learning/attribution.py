"""Error attribution for the self-learning loop (PRD.md section 6.4 "Error attribution").

Two independent pieces:

1. `reconcile_forecast_row` -- verifies a logged forecast's `why`
   (attribution) list is a complete, non-lossy decomposition of `p50`,
   using `FactorModel.baseline_for` as an independent source of truth
   (computed from the model's fitted state, not from `why`/`p50`
   themselves).
2. `attribute_error` -- apportions a graded forecast's `error` (actual
   minus p50) across the recalibratable factors (`holiday`, `event`,
   `weather`) present in that row's `why` list, so
   `app/learning/recalibrate.py` can decide which weight to nudge and in
   which direction.

**Sign convention (read this before touching recalibrate.py):**
`attribute_error` returns, per active recalibratable factor `f`, a
`factor_error_f` with the SAME SIGN as `error` (since it's `error * share_f`
and `share_f >= 0`). `recalibrate.py` combines this with the factor's own
`contribution` sign via `direction_signal_f = factor_error_f *
sign(contribution_f)`:

- `direction_signal_f > 0`: the factor pushed p50 in a direction, and the
  actual missed EVEN FURTHER in that same direction -- the factor's
  effect was too WEAK. Increase that weight's magnitude.
- `direction_signal_f < 0`: the factor pushed p50 one way, but the miss
  is inconsistent with (or opposes) that push -- the factor's effect was
  too STRONG / working against reality. Decrease that weight's magnitude
  (toward, but never past, 0.0 -- a negative weight would invert the
  factor's documented meaning, e.g. a negative `holiday_weight` would mean
  holidays SUPPRESS demand).

Worked example: `holiday_contribution = +5.0`, `error = +8.0` (actual came
in even higher than the already-holiday-boosted p50). Only holiday is
active, so `share_holiday = 1.0` and `factor_error_holiday = +8.0`.
`sign(contribution_holiday) = +1`, so `direction_signal = +8.0 > 0` --
`holiday_weight` was too small, increase it. Conversely
`weather_contribution = -3.0`, `error = +2.0` (actual still came in above
the already-suppressed p50): `factor_error_weather = +2.0`,
`sign(contribution_weather) = -1`, so `direction_signal = -2.0 < 0` --
rain's suppression was too strong, decrease `rain_weight`.
"""

from datetime import date

import pandas as pd

from app.models.factor_model import FactorModel

# `day_of_week` is deliberately excluded: it has no scalar "weight" of its
# own -- it's re-fit fresh from history every `fit()` call (already a form
# of continuous recalibration built into Phase 4), so there is nothing
# here for Phase 6 to adjust for it.
_RECALIBRATABLE_FACTORS = ("holiday", "event", "weather")


def reconcile_forecast_row(
    model: FactorModel,
    location: str,
    item: str,
    target_date: date,
    p50: float,
    why: list[dict],
    tolerance: float = 1e-6,
) -> tuple[float, bool]:
    """Independently verify a logged forecast row's `why` list fully accounts for `p50`.

    Args:
        model: a `FactorModel` already `fit()` on history covering
            `(location, item)` -- used only for its `baseline_for`
            accessor, an independent source of truth computed from fitted
            state (level, trend, weekly_seasonal), not from `why`/`p50`.
        location: venue identifier.
        item: menu item identifier.
        target_date: the forecasted date.
        p50: the logged median forecast for this row.
        why: the logged `attribution` list for this row (each entry has a
            `"contribution"` key).
        tolerance: absolute tolerance for the reconstruction check. Use a
            larger value when reconciling against a JSON-rounded `p50`
            (e.g. from `forecast_latest.json`) rather than the log's
            full-precision `p50` -- this function itself just compares two
            floats within `tolerance`.

    Returns:
        `(reconstructed_p50, is_consistent)` where `reconstructed_p50 =
        baseline_for(...) + sum(entry["contribution"] for entry in why)`
        and `is_consistent = abs(reconstructed_p50 - p50) <= tolerance`.

        This is a REAL check, not a tautology: `baseline_for` is computed
        from the model's fitted state independently of `why`/`p50` -- if a
        future factor were ever added to `predict()` but someone forgot to
        add its contribution to `why`, this reconciliation would catch the
        shortfall (any factor present in the model's actual p50 math but
        absent from `why` breaks this check).

        The one case this can legitimately clamp/differ:
        `predict()`'s `p50 = max(0.0, baseline + factors...)` -- if the raw
        sum was negative, the logged `p50` is 0 but `baseline + sum(why)`
        would be negative. That case is still treated as consistent: this
        function compares `max(0.0, reconstructed_p50)` against the logged
        `p50`, not the raw (possibly negative) reconstruction.
    """
    baseline = model.baseline_for(location, item, target_date)
    raw_reconstructed = baseline + sum(entry["contribution"] for entry in why)
    reconstructed_p50 = max(0.0, raw_reconstructed)
    is_consistent = abs(reconstructed_p50 - p50) <= tolerance
    return reconstructed_p50, is_consistent


def attribute_error(row: pd.Series) -> dict[str, float]:
    """Apportion a forecast's `error` across the recalibratable factors present in `why`.

    Args:
        row: one row from `join_forecast_actuals`'s output -- has `p50`,
            `actual_qty`, `error` (`= actual_qty - p50`), and `attribution`
            (the logged `why` list; accepted here under the row's
            `attribution` key, matching `ForecastLog.read()`'s column
            name).

    Returns:
        A dict with a key for every one of `_RECALIBRATABLE_FACTORS` that
        was present (nonzero contribution) in `row`'s `why`/`attribution`
        list; absent factors get no key at all (not a `0.0` key), since
        "this factor wasn't active for this row" is different from "this
        factor was active and had zero apportioned error". Returns `{}` if
        no recalibratable factor was active for this row (e.g. a plain
        weekday with no holiday/event/rain).

        Apportionment: for each active recalibratable factor `f`,
        `share_f = |contribution_f| / sum(|contribution_g| for g in
        active recalibratable factors)`, and `factor_error_f = error *
        share_f`. See the module docstring for the sign convention
        `app/learning/recalibrate.py` builds on top of this.
    """
    why = row["attribution"]
    error = row["error"]

    active: dict[str, float] = {}
    for entry in why:
        factor = entry["factor"]
        if factor in _RECALIBRATABLE_FACTORS:
            contribution = float(entry["contribution"])
            if contribution != 0.0:
                active[factor] = contribution

    if not active:
        return {}

    total_magnitude = sum(abs(c) for c in active.values())
    if total_magnitude == 0.0:
        return {}

    return {factor: error * (abs(contribution) / total_magnitude) for factor, contribution in active.items()}
