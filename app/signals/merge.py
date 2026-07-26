"""Broadcast per-(location, date) signals onto (location, date, item) sales rows.

Sales rows (from `app/connectors/*` or `FeatureStore.query(...)`) are keyed
by `(location, date, item)`: several item rows share the same
`(location, date)`. Signal providers (`app/signals/holidays.py`,
`app/signals/weather.py`, `app/signals/events.py`) return rows keyed by
`(location, date)` only, per `app/signals/base.py::SignalProvider`. This
module is the one place responsible for joining the two: broadcasting each
signal row's feature columns across every item row that shares its
`(location, date)`, without multiplying rows, and filling in documented
defaults for signal columns that a sparse provider (holidays, events) left
absent for non-signal dates.
"""

from datetime import date

import pandas as pd

# Built-in fill defaults for sparse signal columns (holidays, events only
# emit rows for dates where something actually happened -- see their
# module docstrings). Weather is dense (one row per requested date) so it
# needs no default, but including weather columns here would be harmless.
# Callers may pass this dict as-is, or override/extend it via
# `fill_defaults`.
DEFAULT_FILL: dict[str, object] = {
    "is_holiday": 0,
    "holiday_name": "",
    "event_count": 0,
    "event_impact": 0.0,
}


def merge_signals(
    sales_rows: pd.DataFrame,
    signal_frames: list[pd.DataFrame],
    fill_defaults: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Left-merge each signal frame onto `sales_rows`, keyed by (location, date).

    Each frame in `signal_frames` is merged in turn with a left join on
    `(location, date)`, so `sales_rows`' item-level rows are preserved
    one-for-one; a signal row is broadcast across every item sharing its
    `(location, date)` rather than multiplying rows. Every signal frame
    must have at most one row per `(location, date)` -- after each merge,
    the result's row count is checked against `sales_rows`' row count, and
    a `RuntimeError` is raised if they differ, guarding against a provider
    bug (e.g. duplicate `(location, date)` rows) silently turning the join
    many-to-many and multiplying output rows.

    Args:
        sales_rows: a DataFrame with at least `location`, `date`, `item`
            columns (e.g. `CSVConnector.fetch()` or
            `FeatureStore.query(...)` output). Its row order and row count
            are preserved in the output.
        signal_frames: a list of per-(location, date) signal DataFrames
            (e.g. `[holiday_df, weather_df, event_df]`), each with at
            least `location` and `date` columns plus its own feature
            columns. May be sparse (not every date present).
        fill_defaults: column name -> default value, applied (via
            `fillna`) after all signal frames are merged, for columns that
            sparse providers may have left null. Defaults to
            `DEFAULT_FILL` if None; pass an extended/overridden dict for
            providers not covered by `DEFAULT_FILL`. Only columns present
            in the merged result are filled.

    Returns:
        A copy of `sales_rows` (same row count and order) with the feature
        columns from every frame in `signal_frames` merged in, ready for
        `FeatureStore.upsert(...)`.

    Raises:
        RuntimeError: a signal frame has more than one row for some
            `(location, date)` pair, which would multiply rows in the
            merge (a provider-side bug); or a signal frame reuses a column
            name already present in `sales_rows` or an earlier signal
            frame, which pandas would otherwise silently rename with
            `_x`/`_y` suffixes instead of raising.
    """
    if fill_defaults is None:
        fill_defaults = DEFAULT_FILL

    result = sales_rows.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.date
    expected_len = len(result)

    for signal_frame in signal_frames:
        frame = signal_frame.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date

        duplicated = frame.duplicated(subset=["location", "date"])
        if duplicated.any():
            raise RuntimeError(
                "signal frame has more than one row for some (location, date) "
                "pair; each signal frame must be one-row-per-(location, date) "
                "or the merge would multiply sales rows"
            )

        colliding = (set(result.columns) & set(frame.columns)) - {"location", "date"}
        if colliding:
            raise RuntimeError(
                f"signal frame reuses column name(s) {sorted(colliding)} already "
                "present in sales_rows or an earlier signal frame; pandas would "
                "silently suffix these with _x/_y instead of raising -- rename "
                "the column in the offending provider"
            )

        result = result.merge(frame, on=["location", "date"], how="left")

        if len(result) != expected_len:
            raise RuntimeError(
                f"merge_signals produced {len(result)} rows from "
                f"{expected_len} sales rows -- a signal frame multiplied "
                "rows instead of broadcasting onto (location, date)"
            )

    fillable = {
        column: default for column, default in fill_defaults.items() if column in result.columns
    }
    if fillable:
        result = result.fillna(value=fillable)
        # A left merge that leaves some rows unmatched upcasts an int
        # column to float (NaN has no int representation); restore the
        # documented int dtype now that fillna has removed all NaNs.
        for column, default in fillable.items():
            if isinstance(default, int) and not isinstance(default, bool):
                result[column] = result[column].astype("int64")

    return result
