"""Factor model: `demand = baseline + sum(factor_weight * factor_value) + residual` (PRD.md section 5).

The v1 hackathon forecaster (PRD.md section 11), registered in
`MODEL_REGISTRY["factor_model_v1"]` to match `config.yaml`'s
`model_name: factor_model_v1`. Deliberately transparent and dependency-light
(numpy only -- no scikit-learn/statsmodels): every forecast decomposes into
named, plain-English factor contributions (PRD.md section 6.3), and the
p10/p50/p90 band comes from a closed-form residual-sigma + z-score
computation rather than a fitted probabilistic model.

Per-(location, item) fitted state (`fit`):
  - `level`: mean `qty_sold` across the group's history.
  - `trend`: OLS slope of `qty_sold` on day-index (days since the group's
    earliest history date), via `numpy.polyfit(day_index, qty_sold, 1)[0]`.
    Defaults to 0.0 when fewer than 2 rows are available (a single point
    can't support a slope).
  - `weekly_seasonal[weekday]`: mean deviation of `qty_sold` from `level`
    (NOT from a detrended baseline) for each `date.weekday()` value seen
    in the group's history; additive, not multiplicative, so it composes
    as a simple addend in the attribution list. Weekdays absent from a
    short history default to a 0.0 deviation.
  - `residual_sigma`: population std-dev (`ddof=0`) of
    `actual - (level + trend * day_index + weekly_seasonal[weekday])`
    across the group's history -- the spread driver for p10/p90. Floored
    at `_MIN_RESIDUAL_SIGMA` so a flat/short history never yields a
    zero-width (degenerate) band.

Per-row forecast (`predict`):
  - `baseline = level + trend * day_index`, `day_index` continuing the
    same numbering scheme fit against (days since the group's earliest
    *fitted* date), so `trend` extrapolates forward correctly past the
    fitted window.
  - `day_of_week`: the fitted weekly seasonal deviation for the row's
    weekday. Never capped; always present in the attribution list (even
    when ~0), since it is structurally always in play.
  - `holiday`: `HOLIDAY_WEIGHT * level * is_holiday`. Never capped.
  - `event`: `EVENT_WEIGHT * level * event_impact`. Never capped.
  - `weather`: raw signal `-RAIN_WEIGHT * level * is_rain` (rain suppresses
    demand), then clipped to +/- `WEATHER_CONTRIBUTION_CAP * level`. This is
    the ONLY capped factor -- PRD.md section 5/9 requires weather to be
    down-weighted at the +7..+13 horizon this phase exclusively forecasts,
    so the cap here is unconditionally applied (not offset-dependent). A
    later near-term (+1..+6) phase that wants to relax/re-weight weather as
    the day nears would change how the cap is applied in *this* function --
    the constant itself can stay put.
  - `p50 = max(0.0, baseline + day_of_week + holiday + event + weather)`.
  - `spread` widens with horizon offset and is floored at a small epsilon
    so it is never negative; `p10 = max(0.0, p50 + Z10 * spread)`,
    `p90 = p50 + Z90 * spread`. Because `spread >= 0` and
    `Z10 <= 0 <= Z90` always, `p10 <= p50 <= p90` holds BY CONSTRUCTION --
    not by luck -- for every row, regardless of the input data.
"""

from datetime import date

import numpy as np
import pandas as pd

from app.models.base import MODEL_REGISTRY, ForecastModel

# Standard-normal 10th/90th percentile z-scores. Hardcoded (rather than
# computed via scipy.stats.norm.ppf) to avoid adding a scipy dependency
# for two well-known constants.
Z10 = -1.2816
Z90 = 1.2816

# Plausible holiday demand uplift: a holiday adds 25% of the group's
# fitted level to p50. A documented placeholder pending real per-venue
# recalibration (PRD.md section 5, Phase 5's recalibration loop).
HOLIDAY_WEIGHT = 0.25

# Plausible per-event demand uplift: at event_impact's max (1.0, i.e.
# event_count >= 5 per app/signals/events.py's _IMPACT_CAP), adds 15% of
# the group's fitted level.
EVENT_WEIGHT = 0.15

# Raw (pre-cap) rain suppression: full rain alone would suppress 20% of
# level. This raw magnitude is then squeezed through WEATHER_CONTRIBUTION_CAP
# below -- see that constant's docstring for why.
RAIN_WEIGHT = 0.20

# Weather is down-weighted at the +7..+13 horizon (PRD.md section 5: "Weather
# is down-weighted at the +7..+13 horizon and re-weighted as the day
# nears"; section 9: "Weather is unreliable at +7..+13 -- down-weight it
# there by design"). Regardless of the raw weather signal's magnitude
# (RAIN_WEIGHT * level above), its contribution to p50 is clipped to at
# most this fraction of the group's fitted level. This model only ever
# forecasts the +7..+13 window (config.yaml's horizon), so the cap is
# unconditionally active here -- it does not vary with the row's horizon
# offset. A later near-term (+1..+6) phase, where weather forecasts become
# reliable as the day nears, would relax or make this cap offset-dependent
# inside `FactorModel.predict` -- that is the one place to change, not this
# constant.
WEATHER_CONTRIBUTION_CAP = 0.05

# The horizon offset (days from `reference_today`) treated as the
# "baseline" spread -- matches config.yaml's default `horizon_start` (+7).
# Spread widens for offsets beyond this. Repeated here (rather than
# imported from app.config) to keep this module free of a config-shape
# dependency; keep in sync with config.yaml's horizon_start if that ever
# changes.
HORIZON_FLOOR = 7

# Fractional growth in spread per day beyond HORIZON_FLOOR. At offset=13
# (6 days past the floor), spread is 1 + 0.05*6 = 1.3x the base
# residual_sigma.
BAND_WIDENING_RATE = 0.05

# Real restaurant weekends are noisier than weekdays; give Fri/Sat/Sun
# rows (matching app/features/synthetic.py's weekend convention) a wider
# band via this extra multiplier on spread. Optional nice-to-have --
# applied multiplicatively after the horizon-widening term, so it never
# interferes with the offset-monotonicity guarantee (spread only grows).
_WEEKEND_WEEKDAYS = {4, 5, 6}
_WEEKEND_SPREAD_MULTIPLIER = 1.15

# Floor for a fitted group's residual_sigma so a short or perfectly flat
# history (sigma == 0) never produces a zero-width (degenerate) p10/p90
# band. Chosen as a small absolute quantity relative to typical synthetic
# item volumes (~20-40 units/day); a real deployment may want to scale
# this with `level` instead once real venues are calibrated.
_MIN_RESIDUAL_SIGMA = 0.5

# `spread` (after horizon widening and the weekend multiplier) is floored
# at this small positive epsilon so it is never zero or negative -- the
# thing that guarantees `p10 <= p50 <= p90` by construction (see module
# docstring) even for a pathological offset/sigma combination.
_MIN_SPREAD = 1e-6

# Below this absolute contribution, a factor's attribution entry is
# considered a numerical no-op and safe to display as "0" (used for the
# weather factor's inclusion check only -- holiday/event use an explicit
# truthy/positive check per their own semantics, not this epsilon).
_ATTRIBUTION_EPSILON = 1e-9

# Signal columns predict() tolerates being absent from `future_features`
# (per ForecastModel.fit's docstring: a signal absent from `history`/
# `future_features` entirely just contributes 0 for every row), with
# their documented Phase-3 fill defaults (app/signals/merge.py::DEFAULT_FILL
# plus the dense weather default).
_SIGNAL_DEFAULTS: dict[str, object] = {
    "is_holiday": 0,
    "holiday_name": "",
    "event_impact": 0.0,
    "is_rain": 0,
}


def _safe_bool(value: object) -> bool:
    """NaN-safe truthiness: `pandas`/`numpy` NaN is truthy in plain Python, which
    would wrongly flag a missing/unfilled signal value as "holiday"/"rain".
    """
    if pd.isna(value):
        return False
    return bool(value)


def _safe_float(value: object, default: float = 0.0) -> float:
    """NaN-safe float conversion, falling back to `default` for missing values."""
    if pd.isna(value):
        return default
    return float(value)


def _safe_str(value: object) -> str:
    """NaN-safe string conversion, falling back to `""` for missing values."""
    if pd.isna(value):
        return ""
    return str(value)


class _GroupState:
    """Fitted per-(location, item) state produced by `FactorModel.fit`."""

    __slots__ = ("earliest_date", "level", "trend", "weekly_seasonal", "residual_sigma")

    def __init__(
        self,
        earliest_date: date,
        level: float,
        trend: float,
        weekly_seasonal: dict[int, float],
        residual_sigma: float,
    ) -> None:
        self.earliest_date = earliest_date
        self.level = level
        self.trend = trend
        self.weekly_seasonal = weekly_seasonal
        self.residual_sigma = residual_sigma


class FactorModel(ForecastModel):
    """`demand = baseline + sum(factor_weight * factor_value) + residual` (PRD.md section 5).

    See the module docstring for the exact fit/predict formulas. Registered
    as `MODEL_REGISTRY["factor_model_v1"]` at the bottom of this module.
    """

    def __init__(self) -> None:
        """Start with no fitted groups; `fit()` must be called before `predict()`."""
        self._groups: dict[tuple[str, str], _GroupState] = {}

    def fit(self, history: pd.DataFrame) -> None:
        """Fit per-(location, item) level/trend/weekly-seasonal/residual-sigma state.

        Args:
            history: rows with at least `location, date, item, qty_sold`.
                Any signal columns present are ignored by `fit` -- only
                `predict` reads them (as row-level future features, not as
                something to fit weights against; the factor weights below
                are fixed module constants, not learned).
        """
        self._groups = {}
        for (location, item), group in history.groupby(["location", "item"], sort=False):
            group = group.sort_values("date")
            dates = list(group["date"])
            earliest_date = dates[0]
            day_index = np.array([(d - earliest_date).days for d in dates], dtype=float)
            qty = group["qty_sold"].to_numpy(dtype=float)
            weekday = np.array([d.weekday() for d in dates])

            level = float(qty.mean())

            if len(qty) >= 2:
                trend = float(np.polyfit(day_index, qty, 1)[0])
            else:
                trend = 0.0

            deviation_from_level = qty - level
            weekly_seasonal: dict[int, float] = {}
            for wd in range(7):
                mask = weekday == wd
                weekly_seasonal[wd] = (
                    float(deviation_from_level[mask].mean()) if mask.any() else 0.0
                )

            weekly_component = np.array([weekly_seasonal[wd] for wd in weekday])
            fitted = level + trend * day_index + weekly_component
            residual = qty - fitted
            sigma = float(residual.std(ddof=0)) if len(residual) > 0 else 0.0
            sigma = max(sigma, _MIN_RESIDUAL_SIGMA)

            self._groups[(location, item)] = _GroupState(
                earliest_date=earliest_date,
                level=level,
                trend=trend,
                weekly_seasonal=weekly_seasonal,
                residual_sigma=sigma,
            )

    def predict(self, future_features: pd.DataFrame, reference_today: date) -> pd.DataFrame:
        """Forecast p10/p50/p90 with attribution for each (location, date, item) row.

        Args:
            future_features: rows with `location, date, item` plus whatever
                of `is_holiday, holiday_name, event_impact, is_rain` are
                present; any of those columns entirely absent are treated
                as their documented Phase-3 fill default (0 / "" / 0.0 / 0)
                for every row -- see `_SIGNAL_DEFAULTS`.
            reference_today: see `ForecastModel.predict`.

        Returns:
            See `ForecastModel.predict`.

        Raises:
            ValueError: a row's `(location, item)` was never seen by
                `fit()` -- deliberately no silent/nonsense extrapolation
                for a group this model has no fitted state for.
        """
        working = future_features.copy()
        for column, default in _SIGNAL_DEFAULTS.items():
            if column not in working.columns:
                working[column] = default

        records = []
        for row in working.itertuples(index=False):
            location = row.location
            item = row.item
            row_date = row.date
            key = (location, item)

            state = self._groups.get(key)
            if state is None:
                raise ValueError(
                    f"FactorModel.predict: no fitted state for (location, item) "
                    f"= {key!r} -- fit() was never called with history for this "
                    "combination; refusing to extrapolate a nonsense prediction"
                )

            day_index = (row_date - state.earliest_date).days
            baseline = state.level + state.trend * day_index

            weekday = row_date.weekday()
            dow_contribution = state.weekly_seasonal.get(weekday, 0.0)

            is_holiday = _safe_bool(row.is_holiday)
            holiday_name = _safe_str(row.holiday_name)
            holiday_contribution = HOLIDAY_WEIGHT * state.level * float(is_holiday)
            include_holiday = is_holiday or bool(holiday_name)

            event_impact = _safe_float(row.event_impact)
            event_contribution = EVENT_WEIGHT * state.level * event_impact
            include_event = event_impact > 0

            is_rain = _safe_bool(row.is_rain)
            raw_weather = -RAIN_WEIGHT * state.level * float(is_rain)
            weather_cap = WEATHER_CONTRIBUTION_CAP * state.level
            weather_contribution = float(np.clip(raw_weather, -weather_cap, weather_cap))
            include_weather = abs(weather_contribution) > _ATTRIBUTION_EPSILON

            p50 = max(
                0.0,
                baseline + dow_contribution + holiday_contribution + event_contribution + weather_contribution,
            )

            offset = (row_date - reference_today).days
            widening = 1.0 + BAND_WIDENING_RATE * (offset - HORIZON_FLOOR)
            spread = state.residual_sigma * widening
            if weekday in _WEEKEND_WEEKDAYS:
                spread *= _WEEKEND_SPREAD_MULTIPLIER
            # Floored at a small positive epsilon -- never negative/zero.
            # Combined with Z10 <= 0 <= Z90 below, this is what makes
            # p10 <= p50 <= p90 hold by construction, not by luck.
            spread = max(spread, _MIN_SPREAD)

            p10 = max(0.0, p50 + Z10 * spread)
            p90 = p50 + Z90 * spread

            attribution = []
            dow_name = row_date.strftime("%A")
            is_weekend = weekday in _WEEKEND_WEEKDAYS
            attribution.append(
                {
                    "factor": "day_of_week",
                    "direction": "up" if dow_contribution >= 0 else "down",
                    "text": f"{'Weekend' if is_weekend else 'Weekday'} ({dow_name})",
                    "contribution": float(dow_contribution),
                }
            )
            if include_holiday:
                name_suffix = f" ({holiday_name})" if holiday_name else ""
                attribution.append(
                    {
                        "factor": "holiday",
                        "direction": "up" if holiday_contribution > 0 else "down",
                        "text": f"Holiday{name_suffix} boosts demand",
                        "contribution": float(holiday_contribution),
                    }
                )
            if include_event:
                attribution.append(
                    {
                        "factor": "event",
                        "direction": "up" if event_contribution > 0 else "down",
                        "text": "Nearby event drawing extra traffic",
                        "contribution": float(event_contribution),
                    }
                )
            if include_weather:
                attribution.append(
                    {
                        "factor": "weather",
                        "direction": "up" if weather_contribution > 0 else "down",
                        "text": "Rain expected -- demand slightly suppressed (weather down-weighted at +7..+13)",
                        "contribution": float(weather_contribution),
                    }
                )

            attribution.sort(key=lambda entry: abs(entry["contribution"]), reverse=True)

            records.append(
                {
                    "location": location,
                    "date": row_date,
                    "item": item,
                    "p10": float(p10),
                    "p50": float(p50),
                    "p90": float(p90),
                    "attribution": attribution,
                }
            )

        return pd.DataFrame.from_records(
            records, columns=["location", "date", "item", "p10", "p50", "p90", "attribution"]
        )


MODEL_REGISTRY["factor_model_v1"] = FactorModel
