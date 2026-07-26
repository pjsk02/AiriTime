"""Synthetic FUTURE-window signal generator for demos/backtests (PRD.md section 6.1, 14 phase 4).

`app/signals/*` (Phase 3) providers are real network callers: they can
only report holidays/weather/events for dates a live API actually has data
for (past dates, or a few days of real forecast). They cannot supply
features for the full +7..+13 rolling horizon in a dev/demo/backtest
environment with no network access and no "the future has already
happened" real data. This module fills that gap for `FactorModel.predict`'s
`future_features` input: it mirrors the column shapes
`app/signals/merge.py::merge_signals` would attach from real Phase-3
providers (`is_holiday, holiday_name, temp_c, precip_mm, is_rain,
event_count, event_impact`) exactly, so downstream code cannot tell the
difference between this and real merged Phase-3 output -- but every value
here is synthetically generated via a seeded RNG for dates that haven't
happened yet. Development/demo/backtest use only; a real deployment would
call the Phase-3 `SignalProvider`s instead for near-term dates.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

# ~1-in-20 days is a synthetic holiday.
_HOLIDAY_PROBABILITY = 0.05
_HOLIDAY_NAME = "Synthetic Holiday"

# temp_c ~ Normal(mean, std) -- a plausible mild-climate placeholder, not
# tied to any real calendar/season/location.
_TEMP_MEAN_C = 18.0
_TEMP_STD_C = 6.0

# On a "rainy" day (probability below), precip_mm is drawn from an
# exponential distribution with this mean; all other days are exactly 0.
_RAIN_PROBABILITY = 0.2
_RAIN_MEAN_MM = 4.0

# Most days have zero events; on a day with an event (probability below),
# draw 1 or 2 events.
_EVENT_PROBABILITY = 0.1
_EVENT_COUNT_CHOICES = [1, 2]
_EVENT_COUNT_WEIGHTS = [0.7, 0.3]

# event_impact = min(1.0, event_count / _IMPACT_CAP), matching
# app/signals/events.py::EventProvider's derivation (redocumented here
# rather than importing that module's private constant, to keep this
# generator independent of the real provider's internals).
_IMPACT_CAP = 5


def generate_synthetic_future_signals(
    location: str,
    date_range: tuple[date, date],
    seed: int | None = 0,
) -> pd.DataFrame:
    """Return synthetic per-(location, date) signal rows for a FUTURE window.

    Mirrors the column shapes `app/signals/merge.py::merge_signals` would
    attach from real Phase-3 providers (`is_holiday, holiday_name, temp_c,
    precip_mm, is_rain, event_count, event_impact`), but every value is
    synthetically generated for dates that haven't happened yet -- there
    is no real holiday calendar, weather forecast, or event listing here.
    Development/demo/backtest use only; a real deployment would call the
    Phase-3 `SignalProvider`s instead for near-term dates.

    Args:
        location: restaurant/venue identifier stamped onto every row.
        date_range: `(start, end)` dates, inclusive on both ends. Unlike
            the sparse real providers (holidays, events), this generator
            is dense: exactly one row is returned per date in the range,
            so callers never have to special-case a missing date the way
            they must for `HolidayProvider`/`EventProvider`.
        seed: seed passed to `numpy.random.default_rng` for reproducible,
            deterministic output; `None` uses nondeterministic entropy.

    Returns:
        A DataFrame with columns `location` (str), `date` (datetime.date),
        `is_holiday` (int, 0/1), `holiday_name` (str, `""` when not a
        holiday), `temp_c` (float), `precip_mm` (float, >= 0), `is_rain`
        (int, 0/1, `precip_mm > 0`), `event_count` (int, >= 0), and
        `event_impact` (float, 0..1). One row per date in `date_range`.
    """
    start, end = date_range
    n_days = (end - start).days + 1
    dates = [start + timedelta(days=i) for i in range(n_days)]

    rng = np.random.default_rng(seed)

    is_holiday = (rng.random(n_days) < _HOLIDAY_PROBABILITY).astype(int)
    holiday_name = np.where(is_holiday == 1, _HOLIDAY_NAME, "")

    temp_c = rng.normal(loc=_TEMP_MEAN_C, scale=_TEMP_STD_C, size=n_days)

    is_raining = rng.random(n_days) < _RAIN_PROBABILITY
    precip_mm = np.where(is_raining, rng.exponential(scale=_RAIN_MEAN_MM, size=n_days), 0.0)
    is_rain = (precip_mm > 0).astype(int)

    has_event = rng.random(n_days) < _EVENT_PROBABILITY
    drawn_counts = rng.choice(_EVENT_COUNT_CHOICES, size=n_days, p=_EVENT_COUNT_WEIGHTS)
    event_count = np.where(has_event, drawn_counts, 0).astype(int)
    event_impact = np.minimum(1.0, event_count / _IMPACT_CAP)

    return pd.DataFrame(
        {
            "location": location,
            "date": dates,
            "is_holiday": is_holiday,
            "holiday_name": holiday_name,
            "temp_c": temp_c,
            "precip_mm": precip_mm,
            "is_rain": is_rain,
            "event_count": event_count,
            "event_impact": event_impact,
        }
    )


# Illustrative demo enrichment only -- see `add_demo_weekend_event` below.
_DEMO_EVENT_WEEKDAYS_COUNTS = {5: 2, 4: 1}  # Saturday: 2 events, Friday: 1 event.


def add_demo_weekend_event(signals: pd.DataFrame) -> pd.DataFrame:
    """Deterministically layer a confirmed local event onto weekend rows, for demo richness only.

    Illustrative-demo helper, separate from `generate_synthetic_future_signals`
    itself: callers opt in by applying this to a FUTURE signal window only
    (never to the historical signals merged onto training/backtest data), so
    the model, the backtest, and the learning loop never see this override --
    it exists purely so the owner UI's "why this number" panel can show a
    multi-reason peak day (e.g. "Weekend" + "Nearby event") instead of a
    single day-of-week factor, without touching FactorModel's thresholds.

    Deterministically sets `event_count`/`event_impact` (using the same
    `event_impact = min(1.0, event_count / _IMPACT_CAP)` derivation as
    `generate_synthetic_future_signals`) on every Saturday (2 events) and
    Friday (1 event) row present in `signals` -- no RNG involved, so this is
    reproducible regardless of `seed`. Rows for other weekdays, and every
    other column, are returned unchanged. A row already carrying a stronger
    synthetic event (`event_count` from the RNG already >= the demo count)
    is left as-is rather than overwritten downward.

    Args:
        signals: a `generate_synthetic_future_signals`-shaped DataFrame for
            the FUTURE prediction window only.

    Returns:
        A copy of `signals` with `event_count`/`event_impact` raised (never
        lowered) on Friday/Saturday rows.
    """
    enriched = signals.copy()
    weekday = enriched["date"].apply(lambda d: d.weekday())
    for wd, demo_count in _DEMO_EVENT_WEEKDAYS_COUNTS.items():
        mask = weekday == wd
        boosted_count = np.maximum(enriched.loc[mask, "event_count"], demo_count)
        enriched.loc[mask, "event_count"] = boosted_count
        enriched.loc[mask, "event_impact"] = np.minimum(1.0, boosted_count / _IMPACT_CAP)
    return enriched
