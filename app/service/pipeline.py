"""FastAPI service glue: composes the existing pipeline callables (no reimplementation).

`run_forecast_pipeline` mirrors `scripts/generate_demo_forecast.py::main` step
for step (same functions, same synthetic-data seeds) so `/run` produces the
same `forecast_latest.json` contract; it just also returns a small summary
dict for the HTTP response.
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd

from app.config import load_config
from app.features.synthetic import generate_synthetic_sales
from app.features.synthetic_signals import generate_synthetic_future_signals
from app.learning.actuals import ActualsStore
from app.models.backtest import walk_forward_backtest
from app.models.factor_model import FactorModel
from app.output.writer import DEFAULT_OUTPUT_PATH, build_forecast_document, compute_plan_for, write_forecast_json

LOCATION = "demo_location"

ACTUALS_PATH = "app/output/actuals.jsonl"


def run_forecast_pipeline(output_path: str = DEFAULT_OUTPUT_PATH) -> dict:
    """Run the same synthetic-data pipeline as `scripts/generate_demo_forecast.py`.

    Writes `forecast_latest.json` via the real `build_forecast_document` /
    `write_forecast_json` callables and returns a small summary dict for the
    `/run` endpoint's HTTP response.
    """
    config = load_config("config.yaml")

    history = generate_synthetic_sales(n_days=120, location=LOCATION, seed=0)
    history_signals = generate_synthetic_future_signals(
        LOCATION, (history["date"].min(), history["date"].max()), seed=7
    )
    history = history.merge(history_signals, on=["location", "date"], how="left")

    eval_dates = [history["date"].max() - timedelta(days=i) for i in range(13, 0, -1)]
    backtest = walk_forward_backtest(history, FactorModel, LOCATION, eval_dates)

    model = FactorModel()
    model.fit(history)

    reference_today = history["date"].max() + timedelta(days=1)
    window_start = reference_today + timedelta(days=config.horizon_start)
    window_end = reference_today + timedelta(days=config.horizon_end)

    future_signals = generate_synthetic_future_signals(LOCATION, (window_start, window_end), seed=1)
    items = sorted(history["item"].unique())
    future_features = future_signals.merge(pd.DataFrame({"item": items}), how="cross")

    predictions = model.predict(future_features, reference_today=reference_today)
    predictions["plan_for"] = predictions.apply(
        lambda r: compute_plan_for(r.p10, r.p50, r.p90, config.critical_fractile), axis=1
    )

    document = build_forecast_document(
        location=LOCATION,
        generated_for_date=reference_today,
        window=(window_start, window_end),
        horizon=(config.horizon_start, config.horizon_end),
        model_name=config.model_name,
        quantile_target=config.critical_fractile,
        skill_vs_naive=backtest["skill_vs_naive"],
        wmape=backtest["wmape"],
        predictions=predictions,
    )
    write_forecast_json(document, path=output_path)

    return {
        "location": document["location"],
        "window": document["window"],
        "model": document["model"],
        "wmape": document["wmape"],
        "skill_vs_naive": document["skill_vs_naive"],
        "n_items": len(document["items"]),
    }


def read_forecast_latest(path: str = DEFAULT_OUTPUT_PATH) -> dict | None:
    """Return the parsed `forecast_latest.json`, or None if it doesn't exist yet."""
    import json

    output_path = Path(path)
    if not output_path.exists():
        return None
    return json.loads(output_path.read_text(encoding="utf-8"))


def ingest_actuals(rows: list[dict], path: str = ACTUALS_PATH) -> int:
    """Ingest realized-sales rows via the real `ActualsStore`. Returns rows ingested."""
    frame = pd.DataFrame(rows, columns=["location", "date", "item", "qty_sold"])
    store = ActualsStore(path)
    store.ingest(frame)
    return len(frame)
