"""Tests for the Phase 3 signal providers (app/signals/holidays.py,
app/signals/weather.py, app/signals/events.py).

No test in this file makes a real HTTP call: the pure `_normalize`
staticmethods are exercised directly with hand-built canned payloads, and
`fetch()` is exercised with `_request` monkeypatched to return a canned
payload -- this file never imports `requests` and never lets a provider's
real `_request` run, so nothing here can reach the network.
"""

from datetime import date

import pytest

from app.signals.events import EventProvider
from app.signals.holidays import HolidayProvider
from app.signals.weather import WeatherProvider


# ---------------------------------------------------------------------------
# HolidayProvider
# ---------------------------------------------------------------------------


def test_holiday_normalize_pure_payload_to_sparse_rows() -> None:
    payload = [
        {"date": "2024-01-01", "localName": "New Year's Day", "name": "New Year's Day"},
        {"date": "2024-07-04", "name": "Independence Day"},  # no localName -> falls back to name
    ]

    result = HolidayProvider._normalize(payload)

    assert list(result.columns) == ["date", "is_holiday", "holiday_name"]
    assert len(result) == 2
    assert (result["is_holiday"] == 1).all()
    assert result.loc[0, "date"] == date(2024, 1, 1)
    assert result.loc[0, "holiday_name"] == "New Year's Day"
    assert result.loc[1, "holiday_name"] == "Independence Day"


def test_holiday_normalize_empty_payload_returns_empty_with_columns() -> None:
    result = HolidayProvider._normalize([])

    assert list(result.columns) == ["date", "is_holiday", "holiday_name"]
    assert len(result) == 0


def test_holiday_fetch_monkeypatched_stamps_location_and_filters_date_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = HolidayProvider(country_code="US")
    payload = [
        {"date": "2024-01-01", "localName": "New Year's Day"},
        {"date": "2024-07-04", "localName": "Independence Day"},  # outside requested range
    ]
    # Monkeypatching directly on the instance -- no `self` in the fake, since
    # an instance attribute is not bound as a descriptor. This proves fetch()
    # never touches the real `_request`/`requests.get` at all.
    monkeypatch.setattr(provider, "_request", lambda year, country_code: payload)

    result = provider.fetch("storeA", (date(2024, 1, 1), date(2024, 1, 10)))

    assert list(result.columns) == ["location", "date", "is_holiday", "holiday_name"]
    # Sparse: only the in-range holiday appears, not one row per day of the
    # 10-day requested range.
    assert len(result) == 1
    assert result.loc[0, "location"] == "storeA"
    assert result.loc[0, "date"] == date(2024, 1, 1)
    assert result.loc[0, "is_holiday"] == 1


def test_holiday_fetch_sparse_returns_no_row_for_non_holiday_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = HolidayProvider(country_code="US")
    payload = [{"date": "2024-01-01", "localName": "New Year's Day"}]
    monkeypatch.setattr(provider, "_request", lambda year, country_code: payload)

    result = provider.fetch("storeA", (date(2024, 1, 1), date(2024, 1, 5)))

    # Requested range spans 5 days but only the actual holiday date is present.
    assert len(result) == 1
    assert result["date"].tolist() == [date(2024, 1, 1)]


# ---------------------------------------------------------------------------
# WeatherProvider
# ---------------------------------------------------------------------------


def test_weather_normalize_pure_payload_derives_is_rain_from_precip() -> None:
    payload = {
        "daily": {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "temperature_2m_mean": [5.0, 10.0, -2.5],
            "precipitation_sum": [0.0, 2.4, 0.0],
        }
    }

    result = WeatherProvider._normalize(payload)

    assert list(result.columns) == ["date", "temp_c", "precip_mm", "is_rain"]
    assert len(result) == 3
    assert result.loc[0, "precip_mm"] == 0.0
    assert result.loc[0, "is_rain"] == 0
    assert result.loc[1, "precip_mm"] == 2.4
    assert result.loc[1, "is_rain"] == 1
    assert result.loc[1, "temp_c"] == 10.0
    assert result.loc[2, "is_rain"] == 0


def test_weather_normalize_empty_payload_returns_empty_with_columns() -> None:
    result = WeatherProvider._normalize({})

    assert list(result.columns) == ["date", "temp_c", "precip_mm", "is_rain"]
    assert len(result) == 0


def test_weather_fetch_monkeypatched_is_dense_one_row_per_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = WeatherProvider(latitude=42.36, longitude=-71.06)
    payload = {
        "daily": {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "temperature_2m_mean": [1.0, 2.0, 3.0],
            "precipitation_sum": [0.0, 1.0, 0.0],
        }
    }
    monkeypatch.setattr(
        provider, "_request", lambda start, end, latitude, longitude: payload
    )

    result = provider.fetch("storeA", (date(2024, 1, 1), date(2024, 1, 3)))

    assert list(result.columns) == ["location", "date", "temp_c", "precip_mm", "is_rain"]
    # Dense: exactly one row per date in the requested range, no gaps.
    assert len(result) == 3
    assert set(result["location"]) == {"storeA"}
    assert result["date"].tolist() == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    assert result.loc[1, "is_rain"] == 1


# ---------------------------------------------------------------------------
# EventProvider
# ---------------------------------------------------------------------------


def test_event_normalize_pure_payload_counts_per_date_and_derives_impact() -> None:
    payload = {
        "_embedded": {
            "events": [
                {"dates": {"start": {"localDate": "2024-01-01"}}},
                {"dates": {"start": {"localDate": "2024-01-01"}}},
                {"dates": {"start": {"localDate": "2024-01-02"}}},
            ]
        }
    }

    result = EventProvider._normalize(payload)

    assert list(result.columns) == ["date", "event_count", "event_impact"]
    assert len(result) == 2  # two distinct dates, not three rows

    row_a = result[result["date"] == date(2024, 1, 1)].iloc[0]
    assert row_a["event_count"] == 2
    assert row_a["event_impact"] == pytest.approx(0.4)  # 2 / _IMPACT_CAP(5)

    row_b = result[result["date"] == date(2024, 1, 2)].iloc[0]
    assert row_b["event_count"] == 1
    assert row_b["event_impact"] == pytest.approx(0.2)

    assert (result["event_impact"] >= 0).all() and (result["event_impact"] <= 1.0).all()


def test_event_normalize_impact_saturates_at_one() -> None:
    events = [{"dates": {"start": {"localDate": "2024-01-01"}}}] * 6  # cap is 5
    payload = {"_embedded": {"events": events}}

    result = EventProvider._normalize(payload)

    assert result.loc[0, "event_count"] == 6
    assert result.loc[0, "event_impact"] == 1.0  # min(1.0, 6/5) == 1.0, not 1.2


def test_event_normalize_empty_payload_returns_empty_with_columns() -> None:
    result = EventProvider._normalize({})

    assert list(result.columns) == ["date", "event_count", "event_impact"]
    assert len(result) == 0


def test_event_fetch_monkeypatched_stamps_location_and_filters_date_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EventProvider(market_id="123")
    payload = {
        "_embedded": {
            "events": [
                {"dates": {"start": {"localDate": "2024-01-05"}}},
                {"dates": {"start": {"localDate": "2024-02-01"}}},  # outside requested range
            ]
        }
    }
    monkeypatch.setattr(provider, "_request", lambda market_id, start, end: payload)

    result = provider.fetch("storeA", (date(2024, 1, 1), date(2024, 1, 31)))

    assert list(result.columns) == ["location", "date", "event_count", "event_impact"]
    # Sparse: only the in-range event date appears, not one row per day of
    # the 31-day requested range.
    assert len(result) == 1
    assert result.loc[0, "location"] == "storeA"
    assert result.loc[0, "date"] == date(2024, 1, 5)
    assert result.loc[0, "event_count"] == 1
