"""AiriWheels demand-forecast agent — FastAPI service entrypoint.

Phase 1 (Scaffold): just the app instance and a `/health` check.
Phase 6 (Service): `/run`, `/actuals`, and `/forecast/latest` compose the
existing pipeline/learning callables in `app/service/pipeline.py` -- no
model or learning logic is reimplemented here.
Studio Layer 1: `/config` exposes read-only, non-secret engineer-plane
config (see its docstring for exactly what it does/doesn't return), and
the `ui/` folder is mounted as static files so the owner app and the
(hidden, `?studio=1`) Engineer Studio are served same-origin with the API
-- no separate `python -m http.server` needed anymore.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import load_config
from app.models.factor_model import DEFAULT_WEIGHTS, WEATHER_CONTRIBUTION_CAP
from app.service.pipeline import ingest_actuals, read_forecast_latest, run_forecast_pipeline

app = FastAPI(title="AiriWheels Demand Forecast Agent")


@app.get("/health")
def health() -> dict:
    """Liveness check. Must stay trivial and dependency-free."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the bare root to the owner UI (there is no `ui/index.html`)."""
    return RedirectResponse(url="/forecast.html")


class ActualRow(BaseModel):
    location: str
    date: str
    item: str
    qty_sold: float


class ActualsRequest(BaseModel):
    rows: list[ActualRow]


@app.post("/run")
def run() -> dict:
    """Run the forecast pipeline and (re)write `forecast_latest.json`."""
    return run_forecast_pipeline()


@app.post("/actuals")
def post_actuals(request: ActualsRequest) -> dict:
    """Ingest realized sales in the shape `ActualsStore.ingest` expects."""
    rows = [row.model_dump() for row in request.rows]
    ingested = ingest_actuals(rows)
    return {"ingested_rows": ingested}


@app.get("/forecast/latest")
def forecast_latest() -> dict:
    """Return the current `forecast_latest.json`, or 404 if none exists yet."""
    document = read_forecast_latest()
    if document is None:
        raise HTTPException(status_code=404, detail="forecast_latest.json does not exist yet — run /run first")
    return document


@app.get("/config")
def get_config() -> dict:
    """Read-only engineer-plane config for the Engineer Studio UI.

    Returns ONLY values already public in `config.yaml` and the
    `FactorModel` module's own fixed constants -- never anything from
    `os.environ`/`.env` (where this project's actual secrets, e.g.
    `MARITIME_TOKEN`/`POS_API_KEY`/`VISUAL_CROSSING_API_KEY`/
    `TICKETMASTER_API_KEY`, live -- see `.env.example`). `AgentConfig`
    (`app/config.py`) has no field that reads an env var or holds a
    credential, so this endpoint cannot leak one by construction.

    `factor_weights` reports `FactorModel.DEFAULT_WEIGHTS` (the weights
    `run_forecast_pipeline` actually fits/predicts with today, since it
    constructs `FactorModel()` with no override) plus the fixed
    `weather_contribution_cap` rail -- these are the real numbers driving
    predictions, not the config.yaml `factors:` booleans (documented in
    `AgentConfig.factors`'s own docstring as inert Phase-1 placeholders
    with no factor logic behind them yet).

    This endpoint is read-only: no request body, no way to change any of
    these values via the API. That stays true until a later phase.
    """
    config = load_config("config.yaml")
    return {
        "horizon": {"start_offset": config.horizon_start, "end_offset": config.horizon_end},
        "model_name": config.model_name,
        "cost_ratio": {
            "cost_underprep": config.cost_underprep,
            "cost_overprep": config.cost_overprep,
        },
        "critical_fractile": config.critical_fractile,
        "factor_weights": {**DEFAULT_WEIGHTS, "weather_contribution_cap": WEATHER_CONTRIBUTION_CAP},
        "signals": config.factors.model_dump(),
    }


# Mounted LAST, after every API route above, so this catch-all static
# handler for ui/ (owner app + hidden Engineer Studio) can never shadow
# /health, /run, /actuals, /forecast/latest, or /config -- FastAPI/Starlette
# match routes in registration order, and API routes registered first always
# win. `html=True` serves ui/forecast.html for both "/" and "/forecast.html".
_UI_DIR = Path(__file__).resolve().parent.parent / "ui"
app.mount("/", StaticFiles(directory=_UI_DIR, html=True), name="ui")
