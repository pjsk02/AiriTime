"""SalesConnector interface — the shared contract for item-level sales sources.

Every connector (CSV in this phase; later Toast/Square, per PRD.md section
6.1) implements this ABC and returns normalized rows, so the feature store
(app/features/store.py) never needs to special-case a data source. New
connectors are added as new files implementing `SalesConnector`, without
editing anything downstream.
"""

from abc import ABC, abstractmethod

import pandas as pd


class SalesConnector(ABC):
    """Abstract base for item-level sales data sources (PRD.md section 6.1).

    Concrete connectors implement `fetch()` and must return normalized rows
    with at least these columns:
      - location (str): restaurant/venue identifier.
      - date (datetime.date): the sales date (a date, not a datetime).
      - item (str): menu item identifier/name.
      - qty_sold (numeric): units sold of that item, on that date, at that
        location.

    Kept intentionally minimal and stable: this is the only contract the
    feature store depends on (PRD.md section 6.2), so future connectors
    (Toast, Square, loyalty/voucher feeds) can be added later without
    touching the feature store or any connector already in the repo.
    """

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Return normalized sales rows.

        Returns:
            A pandas DataFrame with at least the columns `location` (str),
            `date` (datetime.date), `item` (str), and `qty_sold` (numeric).
            An empty DataFrame (correct columns, zero rows) is a valid
            result when there is no data to report.
        """
        raise NotImplementedError
