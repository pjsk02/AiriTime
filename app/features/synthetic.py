"""Synthetic sales data generator for local development (PRD.md section 14).

Produces plausible fake item-level sales history shaped like the real
thing -- a per-item baseline, weekly seasonality (weekends busier than
weekdays), and random noise -- echoing the "baseline + weekly seasonality +
noise" sketch of the factor model (PRD.md section 5) without implementing
the full model. Lets the feature store, and later the model registry, be
exercised before a real POS connector is wired in.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

DEFAULT_ITEMS = ["burger", "fries", "salad", "soda"]

# Friday/Saturday/Sunday are treated as the busier "weekend" nights for a
# restaurant (date.weekday(): Monday=0 ... Sunday=6).
_WEEKEND_WEEKDAYS = {4, 5, 6}
_WEEKEND_FACTOR = 1.4
_WEEKDAY_FACTOR = 1.0
_NOISE_RELATIVE_SCALE = 0.15


def generate_synthetic_sales(
    n_days: int = 120,
    items: list[str] | None = None,
    location: str = "demo_location",
    start_date: date | None = None,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Generate `n_days` of synthetic item-level sales as normalized rows.

    Args:
        n_days: number of consecutive calendar days of history to generate
            (default 120; the PRD calls for at least 90 days of history).
        items: menu item names to generate; defaults to `DEFAULT_ITEMS`
            (4 named items) if None.
        location: restaurant/venue identifier stamped on every row.
        start_date: first date of the generated history; defaults to
            `n_days` days before today if None.
        seed: seed passed to `numpy.random.default_rng`, for reproducible
            output (tests can assert on statistical properties, e.g.
            "weekend mean > weekday mean", against a fixed seed).

    Returns:
        A DataFrame of normalized rows -- columns `location` (str), `date`
        (datetime.date), `item` (str), `qty_sold` (float) -- one row per
        (day, item) pair, directly upsertable into a `FeatureStore` via
        `FeatureStore.upsert(...)`. Each item gets its own baseline level,
        a weekend uplift (Fri/Sat/Sun busier than Mon-Thu) plus independent
        Gaussian noise, clipped at 0 so quantities are never negative.
    """
    if items is None:
        items = DEFAULT_ITEMS
    if start_date is None:
        start_date = date.today() - timedelta(days=n_days)

    rng = np.random.default_rng(seed)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    records = []
    for item_idx, item in enumerate(items):
        baseline = 20.0 + 5.0 * item_idx
        for day in dates:
            weekly_factor = _WEEKEND_FACTOR if day.weekday() in _WEEKEND_WEEKDAYS else _WEEKDAY_FACTOR
            noise = rng.normal(loc=0.0, scale=baseline * _NOISE_RELATIVE_SCALE)
            qty_sold = max(0.0, baseline * weekly_factor + noise)
            records.append(
                {
                    "location": location,
                    "date": day,
                    "item": item,
                    "qty_sold": round(qty_sold, 2),
                }
            )

    return pd.DataFrame.from_records(records, columns=["location", "date", "item", "qty_sold"])
