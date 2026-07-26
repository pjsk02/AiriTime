"""Local-events signal provider, modeled on the Ticketmaster Discovery API (PRD.md section 6.1).

The Ticketmaster Discovery API's event-search endpoint returns JSON shaped
like:

    {
        "_embedded": {
            "events": [
                {"dates": {"start": {"localDate": "2026-01-01"}}, ...},
                ...
            ]
        }
    }

possibly across multiple pages; this provider reads a single page (real
pagination is out of scope for this mock-shaped provider).

Per the sparse-provider convention documented in `app/signals/merge.py`,
this provider only returns rows for dates that actually have events;
`merge_signals` is responsible for filling `event_count=0,
event_impact=0.0` for every other date.

`event_impact` is explicitly a raw derived feature, not a weighting/
importance model: it is `min(1.0, event_count / _IMPACT_CAP)`, i.e. a
0..1 signal that saturates once `event_count` reaches `_IMPACT_CAP` events
on a single day. Any actual importance weighting is a factor-model concern
(PRD.md section 5), not this provider's job.
"""

from datetime import date

import pandas as pd
import requests

from app.signals.base import SignalProvider

_API_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
_REQUEST_TIMEOUT_SECONDS = 10

# event_impact = min(1.0, event_count / _IMPACT_CAP) -- a plain, documented
# derivation, not a weighting/importance model (see module docstring).
_IMPACT_CAP = 5


class EventProvider(SignalProvider):
    """Local event counts for a market, sourced from the Ticketmaster Discovery API.

    The network call (`_request`) and the payload normalization
    (`_normalize`) are deliberately separate: `_normalize` is a pure
    function of already-fetched data, so it can be exercised in tests with
    canned payloads and no live HTTP, and `_request` can be monkeypatched
    independently.
    """

    def __init__(self, market_id: str) -> None:
        """Store the Ticketmaster market/DMA identifier; no network call yet.

        Args:
            market_id: Ticketmaster "market" (or DMA) identifier to search
                events within, as accepted by the Discovery API's
                `marketId` query parameter.
        """
        self.market_id = market_id

    def fetch(self, location: str, date_range: tuple[date, date]) -> pd.DataFrame:
        """Return per-date event counts for `location` within `date_range`.

        Args:
            location: restaurant/venue identifier stamped on every row.
            date_range: `(start, end)` dates, inclusive on both ends.

        Returns:
            A DataFrame with columns `location` (str), `date`
            (datetime.date), `event_count` (int), and `event_impact`
            (float, 0..1). Only dates with at least one event are present
            (sparse); dates with zero events are not returned. Empty
            (correct columns, zero rows) if no event falls within
            `date_range`.
        """
        start, end = date_range
        payload = self._request(self.market_id, start, end)
        normalized = self._normalize(payload)

        if normalized.empty:
            in_range = normalized
        else:
            in_range = normalized[
                (normalized["date"] >= start) & (normalized["date"] <= end)
            ]

        result = in_range.copy()
        result["location"] = location
        return result[
            ["location", "date", "event_count", "event_impact"]
        ].reset_index(drop=True)

    def _request(self, market_id: str, start: date, end: date) -> dict:
        """Call the Ticketmaster Discovery API and return the raw parsed JSON.

        Isolated so tests can monkeypatch this method and exercise
        `_normalize` (or `fetch`) without any live HTTP call. Reads a
        single page of results; real pagination is out of scope.

        Args:
            market_id: Ticketmaster market/DMA identifier.
            start: first date of interest (inclusive).
            end: last date of interest (inclusive).

        Returns:
            The parsed JSON response: a dict optionally containing an
            `"_embedded"` key with an `"events"` list, each event a dict
            with at least `dates.start.localDate` (`"YYYY-MM-DD"`).
        """
        params = {
            "marketId": market_id,
            "startDateTime": f"{start.isoformat()}T00:00:00Z",
            "endDateTime": f"{end.isoformat()}T23:59:59Z",
        }
        response = requests.get(_API_URL, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _normalize(payload: dict) -> pd.DataFrame:
        """Normalize a raw Ticketmaster payload into per-date event counts.

        Pure function of already-fetched data -- no network access -- so
        tests can call it directly with canned payloads.

        Args:
            payload: a dict as returned by `_request`, optionally
                containing `payload["_embedded"]["events"]`, a list of
                event dicts each with at least
                `event["dates"]["start"]["localDate"]`.

        Returns:
            A DataFrame with columns `date` (datetime.date), `event_count`
            (int, number of events that day), and `event_impact` (float,
            0..1, `min(1.0, event_count / _IMPACT_CAP)`). One row per
            distinct date that has at least one event. Empty (correct
            columns, zero rows) if there are no events in `payload`.
        """
        events = payload.get("_embedded", {}).get("events", [])

        if not events:
            return pd.DataFrame(columns=["date", "event_count", "event_impact"])

        event_dates = []
        for event in events:
            local_date = event.get("dates", {}).get("start", {}).get("localDate")
            if local_date is None:
                continue
            event_dates.append(pd.to_datetime(local_date).date())

        counts = pd.Series(event_dates, name="date").value_counts()
        result = counts.rename("event_count").reset_index()
        result.columns = ["date", "event_count"]
        result["event_impact"] = (result["event_count"] / _IMPACT_CAP).clip(upper=1.0)
        return result.sort_values("date").reset_index(drop=True)[
            ["date", "event_count", "event_impact"]
        ]
