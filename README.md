# AiriWheels Demand Forecast Agent

A self-learning agent that forecasts short-horizon restaurant food demand as a
*range* (P10/P50/P90), explains why, and gets more accurate every day by
grading itself against what actually sold. See `../PRD.md` for full product
context.

**This is the Phase 1 "Scaffold" build** (see PRD.md section 14): a runnable
skeleton with a config loader and a `/health` check only. There is no
forecasting logic, no data connectors, no signal providers, no feature store,
and no Maritime wiring yet — those arrive in later roadmap phases.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run tests

```bash
pytest
```

## Run the server

```bash
uvicorn app.main:app --port 8080
```

Then check:

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

## Config

`config.yaml` holds engineer-plane placeholders for the rolling horizon
(+7..+13), the selected model, factor toggles, and the newsvendor cost ratio
(`q* = Cu / (Cu + Co)`). Loaded via `app/config.py::load_config`. Nothing in
`/health` depends on config loading — it is intentionally decoupled so the
liveness check stays robust.
