"""Weather signal provider, modeled on the Open-Meteo forecast/archive API (PRD.md section 6.1).

The Open-Meteo daily endpoint returns JSON shaped like:

    {
        "daily": {
            "time": ["2026-01-01", "2026-01-02", ...],
            "temperature_2m_mean": [5.1, 6.3, ...],
            "precipitation_sum": [0.0, 2.4, ...]
        }
    }

Unlike holidays/events, weather is dense: the real API returns a value for
every requested day, so this provider returns one row per date in the
requested range (no sparsity, no fill-default needed downstream for these
columns -- see `app/signals/merge.py`).
"""

from datetime import date

import pandas as pd
import requests

from app.signals.base import SignalProvider

_API_URL = "https://api.open-meteo.com/v1/forecast"
_REQUEST_TIMEOUT_SECONDS = 10

# Any recorded precipitation counts as "rain" for this simple 0/1 feature;
# no attempt is made to distinguish rain from snow/sleet at this stage.
_RAIN_THRESHOLD_MM = 0.0


class WeatherProvider(SignalProvider):
    """Daily mean temperature and precipitation, sourced from Open-Meteo.

    The network call (`_request`) and the payload normalization
    (`_normalize`) are deliberately separate: `_normalize` is a pure
    function of already-fetched data, so it can be exercised in tests with
    canned payloads and no live HTTP, and `_request` can be monkeypatched
    independently.
    """

    def __init__(self, latitude: float, longitude: float) -> None:
        """Store the venue's coordinates; no network call happens until `fetch()` runs.

        Args:
            latitude: venue latitude in decimal degrees, passed to the
                Open-Meteo API.
            longitude: venue longitude in decimal degrees, passed to the
                Open-Meteo API.
        """
        self.latitude = latitude
        self.longitude = longitude

    def fetch(self, location: str, date_range: tuple[date, date]) -> pd.DataFrame:
        """Return one weather row per date in `date_range` for `location`.

        Note: `location` here is the restaurant/venue name string used
        throughout the feature store; it is stamped onto the output rows
        as-is and is not used to derive coordinates -- coordinates come
        from `self.latitude`/`self.longitude`, set at construction.

        Args:
            location: restaurant/venue identifier stamped on every row.
            date_range: `(start, end)` dates, inclusive on both ends.

        Returns:
            A DataFrame with columns `location` (str), `date`
            (datetime.date), `temp_c` (float), `precip_mm` (float), and
            `is_rain` (int, 0/1, derived from `precip_mm > 0`). One row
            per date in `date_range` (dense, no missing days).
        """
        start, end = date_range
        payload = self._request(start, end, self.latitude, self.longitude)
        normalized = self._normalize(payload)

        result = normalized.copy()
        result["location"] = location
        return result[
            ["location", "date", "temp_c", "precip_mm", "is_rain"]
        ].reset_index(drop=True)

    def _request(
        self, start: date, end: date, latitude: float, longitude: float
    ) -> dict:
        """Call the Open-Meteo API for `[start, end]` and return the raw parsed JSON.

        Isolated so tests can monkeypatch this method and exercise
        `_normalize` (or `fetch`) without any live HTTP call.

        Args:
            start: first date to request (inclusive).
            end: last date to request (inclusive).
            latitude: decimal-degree latitude.
            longitude: decimal-degree longitude.

        Returns:
            The parsed JSON response: a dict with a `"daily"` key holding
            parallel lists `"time"`, `"temperature_2m_mean"`, and
            `"precipitation_sum"`.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "auto",
        }
        response = requests.get(_API_URL, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _normalize(payload: dict) -> pd.DataFrame:
        """Normalize a raw Open-Meteo payload into daily weather rows.

        Pure function of already-fetched data -- no network access -- so
        tests can call it directly with canned payloads.

        Args:
            payload: a dict as returned by `_request`, with a `"daily"`
                key holding parallel lists `"time"`,
                `"temperature_2m_mean"`, and `"precipitation_sum"`.

        Returns:
            A DataFrame with columns `date` (datetime.date), `temp_c`
            (float), `precip_mm` (float), and `is_rain` (int, 0/1). Empty
            (correct columns, zero rows) if `payload["daily"]["time"]` is
            empty or the `"daily"` key is absent.
        """
        daily = payload.get("daily", {})
        times = daily.get("time", [])

        if not times:
            return pd.DataFrame(columns=["date", "temp_c", "precip_mm", "is_rain"])

        temps = daily.get("temperature_2m_mean", [])
        precips = daily.get("precipitation_sum", [])

        records = []
        for idx, day_str in enumerate(times):
            precip_mm = float(precips[idx]) if idx < len(precips) else 0.0
            records.append(
                {
                    "date": pd.to_datetime(day_str).date(),
                    "temp_c": float(temps[idx]) if idx < len(temps) else float("nan"),
                    "precip_mm": precip_mm,
                    "is_rain": int(precip_mm > _RAIN_THRESHOLD_MM),
                }
            )
        return pd.DataFrame.from_records(
            records, columns=["date", "temp_c", "precip_mm", "is_rain"]
        )
