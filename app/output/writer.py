"""Assemble and write `forecast_latest.json` (PRD.md sections 6.3, 8's `/forecast/latest`, 14 phase 4).

Turns a `ForecastModel.predict(...)`-shaped DataFrame (`location, date,
item, p10, p50, p90, attribution`) plus backtest/run metadata into the
owner-facing forecast document contract, and writes it to disk. This is
the one place that converts the model's float p10/p50/p90 into the
integer quantities the JSON contract exposes, and the one place that
computes `plan_for` from the newsvendor critical fractile (PRD.md
section 5: q* = Cu / (Cu + Co)).

Note on `quantile_target`: the document's `quantile_target` field must be
the caller's real, computed `AgentConfig.critical_fractile` (e.g. 0.6667
for the default `cost_underprep=2.0, cost_overprep=1.0`) -- NOT a
hardcoded `0.75`. `0.75` only ever appeared as an illustrative placeholder
in the contract example; nothing in this module hardcodes it.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

_DOW_ABBREVIATIONS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

DEFAULT_OUTPUT_PATH = "app/output/forecast_latest.json"


def compute_plan_for(p10: float, p50: float, p90: float, quantile_target: float) -> float:
    """Interpolate the q*-quantile qty from the model's p10/p50/p90 anchors.

    Piecewise-linear between (0.1, p10)-(0.5, p50) for quantile_target in
    [0.1, 0.5], and (0.5, p50)-(0.9, p90) for quantile_target in [0.5,
    0.9]; clamped to p10/p90 outside [0.1, 0.9]. No scipy/normal-inverse-CDF
    dependency needed -- monotonic by construction since p10<=p50<=p90.

    Args:
        p10: 10th-percentile forecast quantity.
        p50: 50th-percentile (median) forecast quantity.
        p90: 90th-percentile forecast quantity.
        quantile_target: the desired quantile to interpolate at, typically
            `AgentConfig.critical_fractile` (q* = Cu / (Cu + Co)).

    Returns:
        The interpolated quantity at `quantile_target`, as a float.
    """
    if quantile_target <= 0.1:
        return float(p10)
    if quantile_target >= 0.9:
        return float(p90)
    if quantile_target <= 0.5:
        fraction = (quantile_target - 0.1) / 0.4
        return float(p10 + fraction * (p50 - p10))
    fraction = (quantile_target - 0.5) / 0.4
    return float(p50 + fraction * (p90 - p50))


def build_forecast_document(
    location: str,
    generated_for_date: date,
    window: tuple[date, date],
    horizon: tuple[int, int],
    model_name: str,
    quantile_target: float,
    skill_vs_naive: float,
    wmape: float,
    predictions: pd.DataFrame,
) -> dict:
    """Assemble the forecast_latest.json contract as a plain dict (JSON-ready).

    Args:
        location: restaurant/venue identifier.
        generated_for_date: the date this forecast run was generated on
            (PRD.md section 8's daily cron "today").
        window: `(start_date, end_date)` of the served window, inclusive.
        horizon: `(start_offset, end_offset)` in days from "today" (the
            +7..+13 rolling horizon, PRD.md section 5).
        model_name: the registered model name used to produce
            `predictions` (e.g. `"factor_model_v1"`).
        quantile_target: the real computed `AgentConfig.critical_fractile`
            -- see module docstring. Used to compute each day's `plan_for`.
        skill_vs_naive: skill vs the naive same-day-last-week baseline,
            as computed by `app/models/backtest.py::walk_forward_backtest`.
        wmape: weighted MAPE, as computed by the same backtest.
        predictions: a DataFrame shaped like `ForecastModel.predict`'s
            return value -- columns `location, date, item, p10, p50, p90,
            attribution` -- for every (item, date) in the window.

    Returns:
        A plain (JSON-serializable) dict matching the `forecast_latest.json`
        contract: `location, generated_for_date, window, horizon, model,
        quantile_target, skill_vs_naive, wmape, items`. Each item's `days`
        list is sorted ascending by date; each day's `p10/p50/p90/plan_for`
        are ints (rounded from `predictions`' floats), with `p10`/`p90`
        re-clamped against the rounded `p50` as a final safety net, since
        independently rounding three floats can occasionally violate a
        `<=` that held before rounding.
    """
    window_start, window_end = window
    horizon_start, horizon_end = horizon

    doc_items = []
    for item, item_rows in predictions.groupby("item", sort=True):
        item_rows = item_rows.sort_values("date")
        days = []
        for row in item_rows.itertuples(index=False):
            p50_i = round(float(row.p50))
            p10_i = round(float(row.p10))
            p90_i = round(float(row.p90))
            plan_for = compute_plan_for(
                float(row.p10), float(row.p50), float(row.p90), quantile_target
            )
            plan_for_i = round(plan_for)

            # Belt-and-suspenders: independent rounding of three floats can
            # occasionally violate p10<=p50<=p90 even though it held
            # before rounding (e.g. p50=5.4->5, p10=5.49->5 is fine, but a
            # near-tie could flip the strict ordering). Re-clamp against
            # the rounded p50 as the final safety net.
            p10_i = min(p10_i, p50_i)
            p90_i = max(p90_i, p50_i)

            why = [
                {
                    "factor": entry["factor"],
                    "direction": entry["direction"],
                    "text": entry["text"],
                    "contribution": float(entry["contribution"]),
                }
                for entry in row.attribution
            ]

            days.append(
                {
                    "date": row.date.isoformat(),
                    "dow": _DOW_ABBREVIATIONS[row.date.weekday()],
                    "p10": int(p10_i),
                    "p50": int(p50_i),
                    "p90": int(p90_i),
                    "plan_for": int(plan_for_i),
                    "why": why,
                }
            )

        doc_items.append({"item": item, "days": days})

    return {
        "location": location,
        "generated_for_date": generated_for_date.isoformat(),
        "window": {"start_date": window_start.isoformat(), "end_date": window_end.isoformat()},
        "horizon": {"start_offset": horizon_start, "end_offset": horizon_end},
        "model": model_name,
        "quantile_target": float(quantile_target),
        "skill_vs_naive": float(skill_vs_naive),
        "wmape": float(wmape),
        "items": doc_items,
    }


def write_forecast_json(document: dict, path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Write `document` as indented JSON to `path`, creating parent dirs if needed.

    Args:
        document: a plain JSON-serializable dict, as produced by
            `build_forecast_document`.
        path: destination file path; defaults to
            `app/output/forecast_latest.json`. Parent directories are
            created if they do not already exist.

    Returns:
        The `Path` written to.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return output_path
