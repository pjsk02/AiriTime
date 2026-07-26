"""Generate a demo forecast_latest.json from synthetic data (no real POS/network).

Run from the repo root:
    python scripts/generate_demo_forecast.py

Fits FactorModel on synthetic sales+signals, runs a walk-forward backtest for
the printed skill/wmape numbers, predicts the +7..+13 window, and writes
app/output/forecast_latest.json -- then copy it into ui/ to view it:
    Copy-Item app/output/forecast_latest.json ui/forecast_latest.json -Force
"""

from datetime import timedelta

import pandas as pd

from app.config import load_config
from app.features.synthetic import generate_synthetic_sales
from app.features.synthetic_signals import generate_synthetic_future_signals
from app.models.backtest import walk_forward_backtest
from app.models.factor_model import FactorModel
from app.output.writer import build_forecast_document, compute_plan_for, write_forecast_json

LOCATION = "demo_location"


def main() -> None:
    config = load_config("config.yaml")

    history = generate_synthetic_sales(n_days=120, location=LOCATION, seed=0)
    history_signals = generate_synthetic_future_signals(
        LOCATION, (history["date"].min(), history["date"].max()), seed=7
    )
    history = history.merge(history_signals, on=["location", "date"], how="left")

    eval_dates = [history["date"].max() - timedelta(days=i) for i in range(13, 0, -1)]
    backtest = walk_forward_backtest(history, FactorModel, LOCATION, eval_dates)
    print("Backtest:", backtest)

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
    path = write_forecast_json(document)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
