#!/usr/bin/env python3
"""Standalone JSON-contract self-check for the AiriWheels forecast document.

Usage:
    python ui/check_contract.py [path-to-json]

Defaults to ui/forecast_latest.json if no path is given. Validates the
forecast_latest.json contract documented in README.md ("Forecast model" /
"Output writer and the forecast_latest.json contract"), independent of any
app/ code -- stdlib only (json, sys, pathlib, datetime), no pandas/pydantic.

On success: prints a human-readable summary ending in a line containing
exactly "OK" and exits 0.
On any violation: prints a specific error to stderr and exits 1.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

VALID_DOW = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
VALID_DIRECTION = {"up", "down"}

DEFAULT_PATH = Path(__file__).resolve().parent / "forecast_latest.json"


class ContractError(Exception):
    """Raised with a specific, actionable message on any contract violation."""


def _is_number(value: object) -> bool:
    # bool is a subclass of int in Python; exclude it explicitly.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_keys(obj: dict, keys: list[str], where: str) -> None:
    if not isinstance(obj, dict):
        raise ContractError(f"{where}: expected an object/dict, got {type(obj).__name__}")
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ContractError(f"{where}: missing required key(s): {', '.join(missing)}")


def _check_iso_date(value: object, where: str) -> None:
    if not isinstance(value, str):
        raise ContractError(f"{where}: expected a string date, got {type(value).__name__}")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ContractError(f"{where}: '{value}' is not a valid ISO date (YYYY-MM-DD): {exc}") from exc


def _check_why_entry(entry: object, where: str) -> None:
    if not isinstance(entry, dict):
        raise ContractError(f"{where}: expected an object/dict, got {type(entry).__name__}")
    _require_keys(entry, ["factor", "direction", "text", "contribution"], where)

    if not isinstance(entry["factor"], str) or not entry["factor"]:
        raise ContractError(f"{where}.factor: expected a non-empty string")
    if entry["direction"] not in VALID_DIRECTION:
        raise ContractError(
            f"{where}.direction: expected one of {sorted(VALID_DIRECTION)}, got {entry['direction']!r}"
        )
    if not isinstance(entry["text"], str) or not entry["text"]:
        raise ContractError(f"{where}.text: expected a non-empty string")
    if not _is_number(entry["contribution"]):
        raise ContractError(
            f"{where}.contribution: expected a number, got {type(entry['contribution']).__name__}"
        )


def _check_day(day: object, where: str) -> None:
    if not isinstance(day, dict):
        raise ContractError(f"{where}: expected an object/dict, got {type(day).__name__}")
    _require_keys(day, ["date", "dow", "p10", "p50", "p90", "plan_for", "why"], where)

    _check_iso_date(day["date"], f"{where}.date")

    if day["dow"] not in VALID_DOW:
        raise ContractError(f"{where}.dow: expected one of {sorted(VALID_DOW)}, got {day['dow']!r}")

    for field in ("p10", "p50", "p90", "plan_for"):
        if not _is_int(day[field]):
            raise ContractError(
                f"{where}.{field}: expected an int, got {type(day[field]).__name__} ({day[field]!r})"
            )

    p10, p50, p90 = day["p10"], day["p50"], day["p90"]
    if not (p10 <= p50 <= p90):
        raise ContractError(
            f"{where}: expected p10 <= p50 <= p90, got p10={p10}, p50={p50}, p90={p90}"
        )

    why = day["why"]
    if not isinstance(why, list):
        raise ContractError(f"{where}.why: expected a list, got {type(why).__name__}")
    for i, entry in enumerate(why):
        _check_why_entry(entry, f"{where}.why[{i}]")


def _check_item(item: object, where: str) -> None:
    if not isinstance(item, dict):
        raise ContractError(f"{where}: expected an object/dict, got {type(item).__name__}")
    _require_keys(item, ["item", "days"], where)

    if not isinstance(item["item"], str) or not item["item"]:
        raise ContractError(f"{where}.item: expected a non-empty string")

    days = item["days"]
    if not isinstance(days, list) or len(days) == 0:
        raise ContractError(f"{where}.days: expected a non-empty list")

    for i, day in enumerate(days):
        _check_day(day, f"{where}.days[{i}]")


def check_contract(data: object) -> tuple[int, int]:
    """Validate the full forecast document. Returns (n_items, day_counts).

    Raises ContractError with a specific message on any violation.
    """
    if not isinstance(data, dict):
        raise ContractError(f"top-level: expected a JSON object, got {type(data).__name__}")

    _require_keys(
        data,
        [
            "location",
            "generated_for_date",
            "window",
            "horizon",
            "model",
            "quantile_target",
            "skill_vs_naive",
            "wmape",
            "items",
        ],
        "top-level",
    )

    if not isinstance(data["location"], str) or not data["location"]:
        raise ContractError("location: expected a non-empty string")

    _check_iso_date(data["generated_for_date"], "generated_for_date")

    window = data["window"]
    _require_keys(window, ["start_date", "end_date"], "window")
    _check_iso_date(window["start_date"], "window.start_date")
    _check_iso_date(window["end_date"], "window.end_date")

    horizon = data["horizon"]
    _require_keys(horizon, ["start_offset", "end_offset"], "horizon")
    if not _is_int(horizon["start_offset"]):
        raise ContractError(
            f"horizon.start_offset: expected an int, got {type(horizon['start_offset']).__name__}"
        )
    if not _is_int(horizon["end_offset"]):
        raise ContractError(
            f"horizon.end_offset: expected an int, got {type(horizon['end_offset']).__name__}"
        )

    if not isinstance(data["model"], str) or not data["model"]:
        raise ContractError("model: expected a non-empty string")

    for field in ("quantile_target", "skill_vs_naive", "wmape"):
        if not _is_number(data[field]):
            raise ContractError(f"{field}: expected a number, got {type(data[field]).__name__}")

    items = data["items"]
    if not isinstance(items, list) or len(items) == 0:
        raise ContractError("items: expected a non-empty list")

    day_counts = []
    for i, item in enumerate(items):
        _check_item(item, f"items[{i}]")
        day_counts.append(len(item["days"]))

    return len(items), day_counts


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH

    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: could not read {path}: {exc}", file=sys.stderr)
        return 1

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"ERROR: {path} is not valid JSON: {exc}", file=sys.stderr)
        return 1

    try:
        n_items, day_counts = check_contract(data)
    except ContractError as exc:
        print(f"ERROR: contract violation in {path}: {exc}", file=sys.stderr)
        return 1

    total_days = sum(day_counts)
    if day_counts and all(c == day_counts[0] for c in day_counts):
        days_desc = f"{day_counts[0]} days each"
    else:
        days_desc = f"day counts {day_counts}"

    print(
        f"Contract check passed: {path} -- {n_items} items, {days_desc}, "
        f"{total_days} total day-entries. OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
