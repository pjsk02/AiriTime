"""AiriWheels demand-forecast agent — FastAPI service entrypoint.

Phase 1 (Scaffold): just the app instance and a `/health` check.
Phase 6 (Service): `/run`, `/actuals`, and `/forecast/latest` compose the
existing pipeline/learning callables in `app/service/pipeline.py` -- no
model or learning logic is reimplemented here.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.service.pipeline import ingest_actuals, read_forecast_latest, run_forecast_pipeline

app = FastAPI(title="AiriWheels Demand Forecast Agent")


@app.get("/health")
def health() -> dict:
    """Liveness check. Must stay trivial and dependency-free."""
    return {"status": "ok"}


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
