"""Public-holiday signal provider, modeled on the Nager.Date API (PRD.md section 6.1).

`GET https://date.nager.at/api/v3/PublicHolidays/{year}/{countryCode}` returns
a JSON array of objects shaped like:

    [{"date": "2026-01-01", "localName": "New Year's Day", "name": "New Year's Day", ...}, ...]

The API is year-scoped (one request per calendar year), so `fetch()` issues
one request per year spanned by the requested date range.

Per the sparse-provider convention documented in `app/signals/merge.py`,
this provider only returns rows for dates that actually are holidays;
`merge_signals` is responsible for filling `is_holiday=0, holiday_name=""`
for every other date.
"""

from datetime import date

import pandas as pd
import requests

from app.signals.base import SignalProvider

_API_URL = "https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
_REQUEST_TIMEOUT_SECONDS = 10


class HolidayProvider(SignalProvider):
    """Public holidays for a single country, sourced from the Nager.Date API.

    The network call (`_request`) and the payload normalization
    (`_normalize`) are deliberately separate: `_normalize` is a pure
    function of already-fetched data, so it can be exercised in tests with
    canned payloads and no live HTTP, and `_request` can be monkeypatched
    independently.
    """

    def __init__(self, country_code: str = "US") -> None:
        """Store the country code; no network call happens until `fetch()` runs.

        Args:
            country_code: ISO 3166-1 alpha-2 country code accepted by the
                Nager.Date API (default `"US"`).
        """
        self.country_code = country_code

    def fetch(self, location: str, date_range: tuple[date, date]) -> pd.DataFrame:
        """Return holiday rows for `location` within `date_range`.

        Issues one `_request` call per calendar year spanned by
        `date_range`, normalizes each year's payload via `_normalize`, then
        filters to the requested range and stamps `location`.

        Args:
            location: restaurant/venue identifier stamped on every row.
            date_range: `(start, end)` dates, inclusive on both ends.

        Returns:
            A DataFrame with columns `location` (str), `date`
            (datetime.date), `is_holiday` (int, always 1 -- only actual
            holidays appear here), and `holiday_name` (str). Only dates
            that are holidays are present (sparse); non-holiday dates are
            not returned. Empty (correct columns, zero rows) if no holiday
            falls within `date_range`.
        """
        start, end = date_range
        frames = []
        for year in range(start.year, end.year + 1):
            payload = self._request(year, self.country_code)
            frames.append(self._normalize(payload))

        normalized = (
            pd.concat(frames, ignore_index=True)
            if frames
            else self._empty_frame()
        )

        if normalized.empty:
            in_range = normalized
        else:
            in_range = normalized[
                (normalized["date"] >= start) & (normalized["date"] <= end)
            ].drop_duplicates(subset=["date"])

        result = in_range.copy()
        result["location"] = location
        return result[["location", "date", "is_holiday", "holiday_name"]].reset_index(
            drop=True
        )

    def _request(self, year: int, country_code: str) -> list[dict]:
        """Call the Nager.Date API for one year and return the raw parsed JSON.

        Isolated so tests can monkeypatch this method and exercise
        `_normalize` (or `fetch`) without any live HTTP call.

        Args:
            year: calendar year to request holidays for.
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            The parsed JSON response: a list of dicts, each with at least
            a `"date"` (`"YYYY-MM-DD"`) and `"localName"`/`"name"` key.
        """
        url = _API_URL.format(year=year, country_code=country_code)
        response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _normalize(payload: list[dict]) -> pd.DataFrame:
        """Normalize a raw Nager.Date payload (one year) into holiday rows.

        Pure function of already-fetched data -- no network access -- so
        tests can call it directly with canned payloads.

        Args:
            payload: a list of dicts as returned by `_request`, each with
                at least `"date"` (`"YYYY-MM-DD"`) and `"localName"` (falls
                back to `"name"` if `"localName"` is absent).

        Returns:
            A DataFrame with columns `date` (datetime.date), `is_holiday`
            (int, always 1), and `holiday_name` (str). Empty (correct
            columns, zero rows) if `payload` is empty.
        """
        if not payload:
            return pd.DataFrame(columns=["date", "is_holiday", "holiday_name"])

        records = [
            {
                "date": pd.to_datetime(entry["date"]).date(),
                "is_holiday": 1,
                "holiday_name": str(entry.get("localName") or entry.get("name") or ""),
            }
            for entry in payload
        ]
        return pd.DataFrame.from_records(
            records, columns=["date", "is_holiday", "holiday_name"]
        )

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        """An empty DataFrame with this provider's full output schema."""
        return pd.DataFrame(
            columns=["location", "date", "is_holiday", "holiday_name"]
        )
