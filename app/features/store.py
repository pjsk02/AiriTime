"""FeatureStore — the single (location, date, item)-keyed feature table.

Connectors (app/connectors/*) write sales rows into this store; later
phases (signal providers, PRD.md section 6.1) add feature columns such as
`is_holiday` keyed by the same (location, date, item) triple; the model
registry (Phase 4) reads from it. This is what makes the input layer
extensible instead of a pile of source-specific special cases (PRD.md
section 6.2).

Column schema (also documented in README.md "Feature store schema" --
downstream phases must key off these exact names):
  - location: str
  - date: datetime.date
  - item: str
  - qty_sold: float
  - ... plus whatever additional columns connectors/signal providers upsert.
"""

from datetime import date

import pandas as pd

_KEY_COLUMNS = ["location", "date", "item"]
_BASE_COLUMNS = _KEY_COLUMNS + ["qty_sold"]


class FeatureStore:
    """In-memory feature store keyed by (location, date, item).

    Wraps a single pandas DataFrame. `upsert` merges new rows in: existing
    columns are overwritten for matching keys, new columns are added, and
    columns from earlier upserts that the new rows don't mention are left
    untouched (a proper merge, not a naive overwrite). `query` reads back a
    location + inclusive date-range slice, sorted by (date, item).
    """

    def __init__(self) -> None:
        """Start with an empty store using the documented base schema."""
        self._df = pd.DataFrame(columns=_BASE_COLUMNS)

    def upsert(self, rows: pd.DataFrame) -> None:
        """Merge `rows` into the store, keyed by (location, date, item).

        Args:
            rows: a DataFrame with at least the `location`, `date`, `item`
                key columns -- as produced by `SalesConnector.fetch()` or
                (in later phases) a signal provider. Any other columns
                (e.g. `qty_sold`, or later `is_holiday`) are set/updated
                for the matching keys; columns already stored for a key but
                absent from `rows` are preserved. A `date` column holding
                `datetime`/`Timestamp` values is normalized to
                `datetime.date`. A no-op if `rows` is empty.

        Raises:
            ValueError: if `rows` is missing one of the key columns.
        """
        if rows.empty:
            return

        missing = [c for c in _KEY_COLUMNS if c not in rows.columns]
        if missing:
            raise ValueError(f"rows is missing required key column(s): {missing}")

        rows = rows.copy()
        rows["date"] = pd.to_datetime(rows["date"]).dt.date
        rows = rows.set_index(_KEY_COLUMNS)

        if self._df.empty:
            combined = rows
        else:
            existing = self._df.set_index(_KEY_COLUMNS)
            # `rows` (the new/incoming values) take precedence; existing
            # values only fill in where the incoming side is null or the
            # column is absent from `rows` altogether -- upsert semantics.
            combined = rows.combine_first(existing)

        self._df = self._order_columns(combined.reset_index())

    def query(self, location: str, date_range: tuple[date, date]) -> pd.DataFrame:
        """Return rows for `location` within the inclusive `date_range`.

        Args:
            location: restaurant/venue identifier to filter on.
            date_range: `(start, end)` dates, inclusive on both ends.

        Returns:
            A DataFrame sorted by (date, item) with the store's column
            schema (at least `location`, `date`, `item`, `qty_sold`).
            Empty (zero rows, correct columns) if nothing matches --
            including when the store itself has never been upserted into.
        """
        if self._df.empty:
            return self._df.copy()

        start, end = date_range
        mask = (
            (self._df["location"] == location)
            & (self._df["date"] >= start)
            & (self._df["date"] <= end)
        )
        result = self._df.loc[mask].sort_values(["date", "item"])
        return result.reset_index(drop=True)

    @staticmethod
    def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Key + qty_sold columns first (documented order), extras sorted after."""
        extra = sorted(c for c in df.columns if c not in _BASE_COLUMNS)
        ordered = [c for c in _BASE_COLUMNS if c in df.columns] + extra
        return df[ordered]
