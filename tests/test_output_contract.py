"""Tests for the forecast_latest.json output contract (app/output/writer.py).

Covers `build_forecast_document`'s exact top-level and per-day key/type
contract (including the float->int rounding + re-clamp, and the
numpy-scalar-stripping that makes the result plain-`json`-serializable
without a custom encoder), item/day ordering, and `write_forecast_json`'s
round trip through disk.
"""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.config import AgentConfig
from app.output.writer import build_forecast_document, write_forecast_json

_LOCATION = "demo_location"
_GENERATED_FOR_DATE = date(2024, 1, 1)
_WINDOW = (date(2024, 1, 8), date(2024, 1, 14))
_HORIZON = (7, 13)


def _make_predictions() -> pd.DataFrame:
    """A small hand-built predictions frame: 2 items, out-of-order dates
    (to check ascending re-sort), and one numpy-scalar-valued row (to
    check the writer actually strips numpy types rather than relying on
    them already being plain Python)."""
    records = [
        {
            "location": _LOCATION,
            "date": date(2024, 1, 10),
            "item": "burger",
            "p10": np.float64(10.4),
            "p50": np.float64(12.5),
            "p90": np.float64(15.6),
            "attribution": [
                {
                    "factor": "holiday",
                    "direction": "up",
                    "text": "Holiday boosts demand",
                    "contribution": np.float64(3.0),
                },
                {
                    "factor": "day_of_week",
                    "direction": "up",
                    "text": "Weekend (Wednesday)",
                    "contribution": np.float64(1.5),
                },
            ],
        },
        {
            "location": _LOCATION,
            "date": date(2024, 1, 8),
            "item": "burger",
            "p10": 9.0,
            "p50": 9.0,
            "p90": 9.0,
            "attribution": [
                {
                    "factor": "day_of_week",
                    "direction": "down",
                    "text": "Weekday (Monday)",
                    "contribution": -0.2,
                },
            ],
        },
        {
            "location": _LOCATION,
            "date": date(2024, 1, 9),
            "item": "burger",
            "p10": 7.6,
            "p50": 8.4,
            "p90": 9.9,
            "attribution": [
                {
                    "factor": "day_of_week",
                    "direction": "up",
                    "text": "Weekday (Tuesday)",
                    "contribution": 0.4,
                },
            ],
        },
        {
            "location": _LOCATION,
            "date": date(2024, 1, 9),
            "item": "fries",
            "p10": 3.1,
            "p50": 4.0,
            "p90": 5.2,
            "attribution": [
                {
                    "factor": "event",
                    "direction": "up",
                    "text": "Nearby event drawing extra traffic",
                    "contribution": 0.6,
                },
                {
                    "factor": "day_of_week",
                    "direction": "up",
                    "text": "Weekday (Tuesday)",
                    "contribution": 0.3,
                },
            ],
        },
        {
            "location": _LOCATION,
            "date": date(2024, 1, 8),
            "item": "fries",
            "p10": 2.5,
            "p50": 3.0,
            "p90": 3.6,
            "attribution": [
                {
                    "factor": "day_of_week",
                    "direction": "down",
                    "text": "Weekday (Monday)",
                    "contribution": -0.1,
                },
            ],
        },
    ]
    return pd.DataFrame.from_records(
        records, columns=["location", "date", "item", "p10", "p50", "p90", "attribution"]
    )


def _build_sample_document() -> dict:
    quantile_target = AgentConfig().critical_fractile
    return build_forecast_document(
        location=_LOCATION,
        generated_for_date=_GENERATED_FOR_DATE,
        window=_WINDOW,
        horizon=_HORIZON,
        model_name="factor_v1",
        quantile_target=quantile_target,
        skill_vs_naive=0.12,
        wmape=0.34,
        predictions=_make_predictions(),
    )


def test_document_top_level_keys_and_types() -> None:
    quantile_target = AgentConfig().critical_fractile
    document = _build_sample_document()

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
    assert document["location"] == _LOCATION
    assert document["generated_for_date"] == "2024-01-01"
    assert type(document["generated_for_date"]) is str

    assert document["window"] == {"start_date": "2024-01-08", "end_date": "2024-01-14"}
    assert document["horizon"] == {"start_offset": 7, "end_offset": 13}
    assert type(document["horizon"]["start_offset"]) is int
    assert type(document["horizon"]["end_offset"]) is int

    assert document["model"] == "factor_v1"
    assert document["quantile_target"] == pytest.approx(quantile_target)
    assert document["skill_vs_naive"] == pytest.approx(0.12)
    assert document["wmape"] == pytest.approx(0.34)
    assert isinstance(document["items"], list)


def test_items_grouped_and_days_sorted_ascending_by_date() -> None:
    document = _build_sample_document()
    items = document["items"]

    assert [entry["item"] for entry in items] == ["burger", "fries"]

    burger_days = items[0]["days"]
    assert [day["date"] for day in burger_days] == [
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
    ]

    fries_days = items[1]["days"]
    assert [day["date"] for day in fries_days] == ["2024-01-08", "2024-01-09"]


def test_day_entry_keys_types_and_p10_le_p50_le_p90() -> None:
    document = _build_sample_document()

    for item_entry in document["items"]:
        for day in item_entry["days"]:
            assert set(day.keys()) == {"date", "dow", "p10", "p50", "p90", "plan_for", "why"}

            assert len(day["dow"]) == 3
            assert day["dow"] in {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}

            for key in ("p10", "p50", "p90", "plan_for"):
                value = day[key]
                assert type(value) is int, f"{key} was {type(value)!r}, not int"

            assert day["p10"] <= day["p50"] <= day["p90"]

            assert isinstance(day["why"], list)
            for entry in day["why"]:
                assert set(entry.keys()) == {"factor", "direction", "text", "contribution"}
                assert entry["direction"] in ("up", "down")
                assert type(entry["contribution"]) is float, (
                    f"contribution was {type(entry['contribution'])!r}, not float "
                    "-- a leftover numpy scalar would break json.dumps"
                )


def test_numpy_scalar_row_rounds_and_strips_to_plain_python_types() -> None:
    """The 2024-01-10 burger row was built with np.float64 p10/p50/p90 and
    an np.float64 contribution -- confirms the writer's float()/int()
    casts actually strip numpy scalar types rather than relying on the
    input already being plain Python."""
    document = _build_sample_document()
    burger_days = document["items"][0]["days"]
    day = next(d for d in burger_days if d["date"] == "2024-01-10")

    # round(10.4)=10, round(12.5)=12 (banker's rounding to even),
    # round(15.6)=16; already p10<=p50<=p90 so the re-clamp is a no-op.
    assert day["p10"] == 10
    assert day["p50"] == 12
    assert day["p90"] == 16
    assert day["dow"] == "Wed"

    # Attribution order (sorted by the model, by descending abs
    # contribution) survives serialization unchanged.
    assert [entry["factor"] for entry in day["why"]] == ["holiday", "day_of_week"]
    assert day["why"][0]["contribution"] == pytest.approx(3.0)
    assert type(day["why"][0]["contribution"]) is float


def test_document_is_json_dumps_safe() -> None:
    """The real proof there's no leftover numpy scalar hiding in the
    dict: a plain json.dumps (no custom encoder) must not raise."""
    document = _build_sample_document()
    serialized = json.dumps(document)
    assert json.loads(serialized) == document


def test_write_forecast_json_round_trips_and_matches_document(tmp_path) -> None:
    document = _build_sample_document()
    output_path = tmp_path / "forecast_latest.json"

    written_path = write_forecast_json(document, path=output_path)

    assert written_path == output_path
    assert output_path.exists()

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == document
