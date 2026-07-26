"""Tests for the synthetic sales generator (app/features/synthetic.py)."""

from datetime import date

from app.features.store import FeatureStore
from app.features.synthetic import generate_synthetic_sales

# Fri/Sat/Sun per date.weekday() (Monday=0 ... Sunday=6), matching the
# implementation's _WEEKEND_WEEKDAYS = {4, 5, 6}.
_WEEKEND_WEEKDAYS = {4, 5, 6}


def test_default_call_covers_at_least_90_days_and_4_items() -> None:
    df = generate_synthetic_sales()

    assert df["date"].nunique() >= 90
    assert df["item"].nunique() >= 4


def test_qty_sold_is_never_negative() -> None:
    df = generate_synthetic_sales()
    assert (df["qty_sold"] >= 0).all()


def test_generation_is_deterministic_given_same_seed() -> None:
    start = date(2024, 1, 1)
    df1 = generate_synthetic_sales(n_days=30, seed=42, start_date=start)
    df2 = generate_synthetic_sales(n_days=30, seed=42, start_date=start)

    # Same seed + same start_date must reproduce identical rows.
    assert df1.equals(df2)


def test_different_seeds_produce_different_output() -> None:
    start = date(2024, 1, 1)
    df1 = generate_synthetic_sales(n_days=30, seed=1, start_date=start)
    df2 = generate_synthetic_sales(n_days=30, seed=2, start_date=start)

    assert not df1["qty_sold"].equals(df2["qty_sold"])


def test_weekend_mean_meaningfully_exceeds_weekday_mean() -> None:
    df = generate_synthetic_sales(n_days=120, seed=0)
    weekday_num = df["date"].map(lambda d: d.weekday())

    weekend_mean = df.loc[weekday_num.isin(_WEEKEND_WEEKDAYS), "qty_sold"].mean()
    weekday_mean = df.loc[~weekday_num.isin(_WEEKEND_WEEKDAYS), "qty_sold"].mean()

    # Generous threshold: implementation uses a 1.4x weekend factor, so a
    # 1.1x check has ample margin against noise without being a razor-thin
    # assertion that could flake.
    assert weekend_mean > weekday_mean * 1.1


def test_synthetic_output_is_directly_usable_by_feature_store() -> None:
    """Integration smoke test tying synthetic -> FeatureStore.upsert -> .query together."""
    store = FeatureStore()
    rows = generate_synthetic_sales(n_days=100, location="demo_location", seed=7)

    store.upsert(rows)

    start = rows["date"].min()
    end = rows["date"].max()
    assert isinstance(start, date)

    result = store.query("demo_location", (start, end))

    assert {"location", "date", "item", "qty_sold"}.issubset(result.columns)
    assert len(result) == len(rows)
    assert result["item"].nunique() >= 4
