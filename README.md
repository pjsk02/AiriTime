# AiriWheels Demand Forecast Agent

A self-learning agent that forecasts short-horizon restaurant food demand as a
*range* (P10/P50/P90), explains why, and gets more accurate every day by
grading itself against what actually sold. See `../PRD.md` for full product
context.

**This build is on Phase 2 "Input layer"** (see PRD.md section 14): Phase 1
shipped a runnable skeleton with a config loader and a `/health` check; Phase
2 adds the `SalesConnector` interface, a CSV connector, the feature store,
and a synthetic dev-data generator (see "Feature store schema" below). There
is no signal providers, no model registry, no self-learning loop, and no
Maritime wiring yet — those arrive in later roadmap phases.

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

## Feature store schema

The feature store (`app/features/store.py::FeatureStore`) is a single table
keyed by **(location, date, item)** (PRD.md section 6.2). Data connectors,
and later signal providers and loyalty data, all upsert columns into it;
the model registry (a later phase) reads from it. Downstream phases must key
off these exact column names:

| Column     | Type            | Notes                                            |
|------------|-----------------|---------------------------------------------------|
| `location` | `str`           | Restaurant/venue identifier. Part of the key.      |
| `date`     | `datetime.date` | Sales date (a date, not a datetime). Part of the key. |
| `item`     | `str`           | Menu item identifier/name. Part of the key.        |
| `qty_sold` | `float`         | Units sold of that item, on that date, at that location. |

Additional columns (e.g. a future `is_holiday` from a signal provider) may be
present once upserted; they are added alongside the four columns above
without disturbing them.

- `FeatureStore.upsert(rows: pd.DataFrame) -> None` merges a DataFrame of
  rows into the store keyed by `(location, date, item)`: matching keys have
  their columns updated/overwritten by the new data, new columns are added,
  and columns from earlier upserts that the new rows don't mention are left
  untouched (a proper merge, not a naive overwrite).
- `FeatureStore.query(location: str, date_range: tuple[date, date]) -> pd.DataFrame`
  returns rows for one location within an inclusive date range, sorted by
  `(date, item)`. Returns an empty DataFrame with the schema above (not an
  error) when nothing matches, including when the store is empty.

### How the store gets populated

- `app/connectors/base.py::SalesConnector` is the abstract interface
  (`fetch() -> pd.DataFrame`) every sales source implements to produce rows
  matching this schema (PRD.md section 6.1). `app/connectors/csv_connector.py::CSVConnector`
  is the v1 CSV/Excel-upload implementation: it reads a CSV, accepts a small
  set of common column-name aliases (see its docstring), and returns
  normalized rows ready for `FeatureStore.upsert(...)`.
- `app/features/synthetic.py::generate_synthetic_sales(...)` generates
  plausible fake sales history (baseline + weekly seasonality + noise, per
  the PRD section 5 factor-model framing) for local development, also
  returned as normalized rows directly upsertable into a `FeatureStore`.
