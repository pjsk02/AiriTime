"""Recalibration: propose new FactorModel weights from graded forecast history (PRD.md section 6.4 "Recalibration").

Combines `app/learning/attribution.py::attribute_error`'s per-factor error
apportionment into a single, deterministic, safety-railed weight update per
recalibratable factor (`holiday_weight`, `event_weight`, `rain_weight`).
No randomness anywhere in this module: the same `graded_rows` + params
always produce the same output.

Five safety rails, all documented on the constants below and enforced in
`recalibrate()`:
  1. `min_samples` -- a factor with too little evidence is left completely
     unchanged (not even a zero-sized update).
  2. `max_step` -- the largest RELATIVE change a weight may take in one
     cycle, regardless of how strong the raw signal is.
  3. `gentle_fraction` -- normal (non-drift) updates use only a fraction
     of `max_step`, so routine noise nudges weights slowly.
  4. `drift_window` / `drift_threshold` -- a recent-window check that
     unlocks the FULL `max_step` (rather than the gentle fraction) only
     when there's a persistent, large, recent miss for that factor.
  5. A non-negative floor (`0.0`) on every weight -- a weight must never
     go negative, since that would invert the factor's documented meaning
     (see `attribution.py`'s worked reasoning).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.learning.attribution import attribute_error

DEFAULT_MAX_STEP = 0.10  # max relative change to a weight in one recalibration, e.g. 10%
DEFAULT_MIN_SAMPLES = 5  # minimum rows where a factor was active before its weight updates at all
DEFAULT_GENTLE_FRACTION = 0.3  # gentle (non-drift) updates use this fraction of max_step
DEFAULT_DRIFT_WINDOW = 10  # how many most-recent per-factor error observations the drift detector looks at
DEFAULT_DRIFT_THRESHOLD = 0.15  # if |windowed mean normalized signal| exceeds this, drift fires
DEFAULT_LEARNING_RATE = 0.5  # fraction of the raw mean signal translated into a proposed relative weight change, before clamping

# Maps attribute_error's factor names ("holiday"/"event"/"weather") to the
# corresponding key in a FactorModel weights dict
# ("holiday_weight"/"event_weight"/"rain_weight" -- note "weather" maps to
# "rain_weight", not "weather_weight": the weight constant that controls
# weather's raw pre-cap suppression magnitude is named RAIN_WEIGHT in
# app/models/factor_model.py).
_FACTOR_TO_WEIGHT_KEY = {
    "holiday": "holiday_weight",
    "event": "event_weight",
    "weather": "rain_weight",
}

_EPSILON = 1e-12


@dataclass(frozen=True)
class _FactorSignal:
    """One row's normalized, signed recalibration signal for a single factor."""

    date: object
    normalized_signal: float


def _normalized_signals_by_factor(graded_rows: pd.DataFrame) -> dict[str, list[_FactorSignal]]:
    """Compute each row's per-factor `direction_signal`, normalized to a small dimensionless number.

    For each row and each recalibratable factor present in that row's
    `why`/`attribution` list (per `attribute_error`), the raw
    `direction_signal_f = factor_error_f * sign(contribution_f)` (see
    `attribution.py`'s worked reasoning) is normalized by dividing by
    `max(abs(row.p50), _EPSILON)`. `p50` is used as the normalizing scale
    (rather than the model's internal `level`, which this module has no
    access to and deliberately avoids depending on) because it is already
    present on every graded row and is a reasonable proxy for the venue/
    item's typical demand scale -- the key property needed is that the
    normalized signal is a small dimensionless number ("relative demand
    miss attributable to this factor"), not a raw unit-of-qty number, so
    it can sensibly scale a RELATIVE weight change in `recalibrate()`.

    Returns a dict keyed by factor name ("holiday"/"event"/"weather"),
    each value a list of `_FactorSignal` in the same (date) order as
    `graded_rows`, one entry per row where that factor was active.
    """
    by_factor: dict[str, list[_FactorSignal]] = {factor: [] for factor in _FACTOR_TO_WEIGHT_KEY}

    ordered = graded_rows.sort_values("date")
    for row in ordered.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        factor_errors = attribute_error(row_series)
        if not factor_errors:
            continue

        why = row_series["attribution"]
        contribution_by_factor = {entry["factor"]: float(entry["contribution"]) for entry in why}
        p50_value = float(row_series["p50"])
        if not np.isfinite(p50_value):
            # A non-finite p50 makes both the normalizing scale and every
            # downstream signal undefined -- skip this row entirely for
            # every factor rather than let NaN/inf silently propagate into
            # recalibrate()'s mean/clip math (see below: NaN survives
            # np.clip unchanged, which would otherwise bypass max_step).
            continue
        scale = max(abs(p50_value), _EPSILON)

        for factor, factor_error in factor_errors.items():
            contribution = contribution_by_factor.get(factor, 0.0)
            if not (np.isfinite(factor_error) and np.isfinite(contribution)):
                continue
            sign = 1.0 if contribution > 0 else (-1.0 if contribution < 0 else 0.0)
            direction_signal = factor_error * sign
            normalized = direction_signal / scale
            if not np.isfinite(normalized):
                continue
            by_factor[factor].append(_FactorSignal(date=row_series["date"], normalized_signal=normalized))

    return by_factor


def recalibrate(
    graded_rows: pd.DataFrame,
    current_weights: dict[str, float],
    max_step: float = DEFAULT_MAX_STEP,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    gentle_fraction: float = DEFAULT_GENTLE_FRACTION,
    drift_window: int = DEFAULT_DRIFT_WINDOW,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> dict[str, object]:
    """Compute a new weights dict from graded (forecast vs actual) history, under hard safety rails.

    Deterministic given the same `graded_rows` + params (no randomness
    anywhere in this function).

    Args:
        graded_rows: output of `app.learning.actuals.join_forecast_actuals`
            -- at least `location, item, date, p50, actual_qty, error,
            attribution` (the logged `why` list).
        current_weights: `{"holiday_weight", "event_weight",
            "rain_weight"}` -- the weights this cycle starts from.
        max_step: see module docstring, rail 2.
        min_samples: see module docstring, rail 1.
        gentle_fraction: see module docstring, rail 3.
        drift_window: see module docstring, rail 4.
        drift_threshold: see module docstring, rail 4.
        learning_rate: fraction of the raw mean normalized signal
            translated into a proposed relative weight change, before
            clamping to the applicable step cap.

    Returns:
        `{"new_weights": {...all three keys, always...}, "updates": {...
        one entry per factor's weight key, ALWAYS present even when
        guarded/unchanged...}}`. See each field's inline documentation
        below.
    """
    signals_by_factor = _normalized_signals_by_factor(graded_rows)

    new_weights: dict[str, float] = dict(current_weights)
    updates: dict[str, dict[str, object]] = {}

    for factor, weight_key in _FACTOR_TO_WEIGHT_KEY.items():
        signals = signals_by_factor[factor]
        n_samples = len(signals)
        old_value = float(current_weights[weight_key])

        if n_samples < min_samples:
            # Rail 1: guarded -- leave this weight COMPLETELY unchanged,
            # bit-for-bit identical to current_weights[weight_key].
            new_weights[weight_key] = old_value
            updates[weight_key] = {
                "n_samples": n_samples,
                "drift": False,
                "applied_step": 0.0,
                "old": old_value,
                "new": old_value,
            }
            continue

        all_values = [s.normalized_signal for s in signals]
        windowed_values = all_values[-drift_window:]

        drift_fires = abs(float(np.mean(windowed_values))) > drift_threshold

        raw_relative_delta = learning_rate * float(np.mean(all_values))

        # Rail 3/4: gentle updates use only `gentle_fraction` of max_step;
        # a persistent, large, recent (windowed) miss unlocks the full
        # max_step.
        applied_cap = max_step if drift_fires else (max_step * gentle_fraction)

        # Rail 2: CLAMP, don't reject -- even if the raw signal wanted a
        # bigger move, cap it rather than skip the update (only the
        # min_samples check above causes an outright skip).
        relative_delta = float(np.clip(raw_relative_delta, -applied_cap, applied_cap))

        if not np.isfinite(relative_delta):
            # Belt-and-suspenders: `_normalized_signals_by_factor` already
            # filters non-finite per-row signals before they reach here, so
            # this should be unreachable in practice -- but `np.clip` passes
            # NaN through UNCHANGED rather than clamping it (a NaN survives
            # `np.clip(nan, -cap, cap)` as NaN), so if anything upstream
            # ever regresses that filter, a NaN `relative_delta` must never
            # be allowed to reach `old_value * (1.0 + relative_delta)`
            # below: that expression evaluates to `nan`, and
            # `max(0.0, nan)` returns `nan` (not `0.0`) under Python's `max`
            # semantics -- which would silently zero out `applied_step`'s
            # "no move occurred" branch while actually corrupting the
            # weight to NaN. Treat a non-finite proposal as "no valid
            # evidence this cycle" and leave the weight unchanged instead.
            new_weights[weight_key] = old_value
            updates[weight_key] = {
                "n_samples": n_samples,
                "drift": False,
                "applied_step": 0.0,
                "old": old_value,
                "new": old_value,
            }
            continue

        # Rail 5: floor at 0.0 -- a weight must never go negative.
        # Flooring at 0 can only ever pull `new_value` TOWARD 0, which
        # (combined with an already-capped `relative_delta`) can only make
        # the actual step SMALLER in magnitude than what the clamped
        # `relative_delta` promised -- it can never make the step LARGER,
        # so the applied_step computed below is guaranteed <= applied_cap
        # even after this floor.
        new_value = max(0.0, old_value * (1.0 + relative_delta))

        if old_value != 0.0:
            applied_step = abs(new_value - old_value) / old_value
        else:
            # Guard division-by-zero: if the weight was already 0.0, any
            # non-negative new_value represents an undefined relative
            # step; report 0.0 since old_value * (1 + relative_delta) is
            # itself 0.0 regardless of relative_delta, so no move occurred.
            applied_step = 0.0

        new_weights[weight_key] = new_value
        updates[weight_key] = {
            "n_samples": n_samples,
            "drift": drift_fires,
            "applied_step": applied_step,
            "old": old_value,
            "new": new_value,
        }

    return {"new_weights": new_weights, "updates": updates}
