"""SignalProvider interface -- the shared contract for per-(location, date) external signals.

Every signal provider (holidays, weather, events in this phase; more later,
per PRD.md section 6.1) implements this ABC and returns normalized rows keyed
by `(location, date)` -- NOT by item. Sales rows in the feature store
(app/features/store.py) are keyed by `(location, date, item)`, so a later
merge step (app/signals/merge.py) broadcasts each signal row across every
item row sharing its `(location, date)`. New providers are added as new
files implementing `SignalProvider`, without editing anything downstream.
"""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class SignalProvider(ABC):
    """Abstract base for per-(location, date) external signal sources (PRD.md section 6.1).

    Concrete providers implement `fetch()` and must return normalized rows
    with at least these columns:
      - location (str): restaurant/venue identifier.
      - date (datetime.date): the signal date (a date, not a datetime).
      - plus one or more feature columns specific to the provider (e.g.
        `is_holiday`, `temp_c`, `event_count`).

    Rows are keyed by `(location, date)`, not `(location, date, item)`: a
    signal is shared by every menu item sold at that location on that date.
    `app/signals/merge.py::merge_signals` is the one place responsible for
    broadcasting these rows onto item-level sales rows and for filling in
    defaults where a provider's output is sparse (e.g. non-holiday dates).

    Kept intentionally minimal and stable, mirroring how `SalesConnector`
    (app/connectors/base.py) is the only contract the feature store depends
    on: this is the only contract the merge step depends on, so future
    providers can be added later without touching the merge step, the
    feature store, or any provider already in the repo.
    """

    @abstractmethod
    def fetch(self, location: str, date_range: tuple[date, date]) -> pd.DataFrame:
        """Return normalized signal rows for `location` within `date_range`.

        Args:
            location: restaurant/venue identifier stamped on every row.
            date_range: `(start, end)` dates, inclusive on both ends, for
                which to return signal data.

        Returns:
            A pandas DataFrame with at least the columns `location` (str)
            and `date` (datetime.date), plus one or more feature columns.
            At most one row per `(location, date)`. A provider may return
            fewer rows than there are dates in `date_range` (a "sparse"
            provider, e.g. holidays only rows on actual holidays) -- an
            empty DataFrame (correct columns, zero rows) is a valid result
            when there is no signal to report for the whole range.
        """
        raise NotImplementedError
