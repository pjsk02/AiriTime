"""Tests for the Phase 6 service endpoints (/run, /actuals, /forecast/latest).

Uses FastAPI's TestClient against the real app in `app/main.py`, which
wires straight into `app/service/pipeline.py`'s composition of the actual
pipeline/learning callables (`FactorModel`, `walk_forward_backtest`,
`build_forecast_document`, `write_forecast_json`, `ActualsStore`) -- no
model or learning logic is duplicated here.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.output.writer import DEFAULT_OUTPUT_PATH

client = TestClient(app)

_OUTPUT_PATH = Path(DEFAULT_OUTPUT_PATH)


def test_run_writes_file_and_returns_summary() -> None:
    response = client.post("/run")
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {"location", "window", "model", "wmape", "skill_vs_naive", "n_items"}
    assert body["location"] == "demo_location"
    assert body["model"] == "factor_model_v1"
    assert isinstance(body["wmape"], float)
    assert isinstance(body["skill_vs_naive"], float)
    assert body["n_items"] > 0
    assert set(body["window"].keys()) == {"start_date", "end_date"}

    assert _OUTPUT_PATH.exists()


def test_forecast_latest_returns_written_document() -> None:
    run_response = client.post("/run")
    assert run_response.status_code == 200

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
    assert document["location"] == "demo_location"
    assert len(document["items"]) > 0


def test_forecast_latest_404s_when_file_missing() -> None:
    backup = None
    if _OUTPUT_PATH.exists():
        backup = _OUTPUT_PATH.read_text(encoding="utf-8")
        _OUTPUT_PATH.unlink()

    try:
        response = client.get("/forecast/latest")
        assert response.status_code == 404
        assert "forecast_latest.json" in response.json()["detail"]
    finally:
        if backup is not None:
            _OUTPUT_PATH.write_text(backup, encoding="utf-8")


def test_actuals_round_trips_in_real_shape() -> None:
    payload = {
        "rows": [
            {"location": "demo_location", "date": "2026-01-01", "item": "burger", "qty_sold": 12.0},
            {"location": "demo_location", "date": "2026-01-02", "item": "fries", "qty_sold": 30.5},
        ]
    }

    response = client.post("/actuals", json=payload)
    assert response.status_code == 200
    assert response.json() == {"ingested_rows": 2}

    from app.learning.actuals import ActualsStore
    from datetime import date

    store = ActualsStore("app/output/actuals.jsonl")
    read_back = store.read("demo_location", (date(2026, 1, 1), date(2026, 1, 2)))
    assert len(read_back) == 2
