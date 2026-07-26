"""Tests for Studio Layer 1: the ui/ static mount, /config, and that prior
routes/contracts (/health, /run, /actuals, /forecast/latest) survive the mount.

Uses FastAPI's TestClient against the real app in `app/main.py` -- no model
or learning logic is duplicated here.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_SECRET_ENV_VAR_NAMES = {
    "MARITIME_TOKEN",
    "POS_API_KEY",
    "POS_LOCATION_ID",
    "VISUAL_CROSSING_API_KEY",
    "TICKETMASTER_API_KEY",
}
_SECRET_LOOKING_SUBSTRINGS = ("token", "api_key", "apikey", "secret", "password")


def _flatten_values(obj):
    """Yield every leaf value in a nested dict/list structure."""
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _flatten_values(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _flatten_values(item)
    else:
        yield obj


def _flatten_keys(obj):
    """Yield every dict key (at any depth) in a nested dict/list structure."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _flatten_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _flatten_keys(item)


# ---------- static mount reachable, prior API routes intact ----------


def test_health_contract_unchanged() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_run_contract_unchanged() -> None:
    response = client.post("/run")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"location", "window", "model", "wmape", "skill_vs_naive", "n_items"}


def test_forecast_latest_contract_unchanged() -> None:
    client.post("/run")
    response = client.get("/forecast/latest")
    assert response.status_code == 200
    document = response.json()
    assert set(document.keys()) == {
        "location",
        "generated_for_date",
        "window",
        "horizon",
        "model",
        "quantile_target",
        "skill_vs_naive",
        "wmape",
        "items",
    }


def test_actuals_contract_unchanged() -> None:
    payload = {"rows": [{"location": "demo_location", "date": "2026-01-01", "item": "burger", "qty_sold": 12.0}]}
    response = client.post("/actuals", json=payload)
    assert response.status_code == 200
    assert response.json() == {"ingested_rows": 1}


def test_root_serves_owner_ui() -> None:
    response = client.get("/", follow_redirects=True)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AiriWheels" in response.text


def test_forecast_html_reachable_via_static_mount() -> None:
    response = client.get("/forecast.html")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "studioRoot" in response.text


# ---------- GET /config ----------


def test_config_returns_expected_keys() -> None:
    response = client.get("/config")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "horizon",
        "model_name",
        "cost_ratio",
        "critical_fractile",
        "factor_weights",
        "signals",
    }
    assert set(body["horizon"].keys()) == {"start_offset", "end_offset"}
    assert body["horizon"]["start_offset"] == 7
    assert body["horizon"]["end_offset"] == 13
    assert body["model_name"] == "factor_model_v1"
    assert set(body["cost_ratio"].keys()) == {"cost_underprep", "cost_overprep"}
    assert 0.0 <= body["critical_fractile"] <= 1.0
    assert set(body["factor_weights"].keys()) == {
        "holiday_weight",
        "event_weight",
        "rain_weight",
        "weather_contribution_cap",
    }
    assert set(body["signals"].keys()) == {"day_of_week", "holidays", "events", "weather", "loyalty"}


def test_config_contains_no_secret_looking_values() -> None:
    response = client.get("/config")
    body = response.json()

    keys = set(_flatten_keys(body))
    for key in keys:
        lowered = str(key).lower()
        assert not any(bad in lowered for bad in _SECRET_LOOKING_SUBSTRINGS), f"secret-looking key: {key}"

    values = list(_flatten_values(body))
    stringified = {str(v) for v in values}
    assert not (stringified & _SECRET_ENV_VAR_NAMES), "a known secret env-var name leaked as a value"
    for value in values:
        if isinstance(value, str):
            assert "dummy" not in value.lower(), f"a placeholder-secret-shaped value leaked: {value!r}"
            assert not value.startswith("mk_"), f"looks like a Maritime token: {value!r}"
