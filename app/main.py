"""AiriWheels demand-forecast agent — FastAPI service entrypoint.

Phase 1 (Scaffold): just the app instance and a `/health` check.
Forecasting, connectors, signal providers, and the feature store come in
later roadmap phases (see PRD.md sections 6, 7, 14).
"""

from fastapi import FastAPI

app = FastAPI(title="AiriWheels Demand Forecast Agent")


@app.get("/health")
def health() -> dict:
    """Liveness check. Must stay trivial and dependency-free."""
    return {"status": "ok"}
