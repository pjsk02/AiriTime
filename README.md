# AiriWheels Demand Forecast Agent

A self-learning agent that forecasts short-horizon restaurant food demand as a
*range* (P10/P50/P90), explains why, and gets more accurate every day by
grading itself against what actually sold. See `../PRD.md` for full product
context.

**This build is on Phase 4 "Forecast model"** (see PRD.md section 14):
Phase 1 shipped a runnable skeleton with a config loader and a `/health`
check; Phase 2 added the `SalesConnector` interface, a CSV connector, the
feature store, and a synthetic dev-data generator; Phase 3 added the
`SignalProvider` interface, holiday/weather/event providers, and the
`merge_signals` broadcast step (see "Feature store schema" and "Signal
columns" below); Phase 4 adds the model registry, a transparent factor
model producing P10/P50/P90 with attribution, a walk-forward backtest
(wMAPE + skill-vs-naive), a synthetic future-signal generator for
dev/demo/backtest use, and the `forecast_latest.json` output writer (see
"Forecast model" below). There is no self-learning loop (forecast log,
actuals ingestion, recalibration) and no Maritime wiring yet — those
arrive in later roadmap phases.

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

## Signal columns

Phase 3 adds external, per-`(location, date)` signal columns to the same
feature store schema above (still keyed by `(location, date, item)` --
signals are broadcast across every item row sharing a `(location, date)`):

| Column         | Type    | Notes                                                        |
|----------------|---------|---------------------------------------------------------------|
| `is_holiday`   | `int`   | 0/1. From `app/signals/holidays.py::HolidayProvider`.          |
| `holiday_name` | `str`   | Empty string `""` when not a holiday.                          |
| `temp_c`       | `float` | Daily mean temperature. From `app/signals/weather.py::WeatherProvider`. |
| `precip_mm`    | `float` | Daily precipitation total.                                     |
| `is_rain`      | `int`   | 0/1, derived from `precip_mm > 0`.                              |
| `event_count`  | `int`   | Number of local events that day. From `app/signals/events.py::EventProvider`. |
| `event_impact` | `float` | 0..1, `min(1.0, event_count / 5)` -- a raw derived feature, not a weighting/importance model. |

- `app/signals/base.py::SignalProvider` is the abstract interface
  (`fetch(location, date_range) -> pd.DataFrame`) every external signal
  source implements, mirroring `SalesConnector` but keyed by
  `(location, date)` instead of `(location, date, item)`.
- Holidays and events are **sparse** providers: they only return rows for
  dates where something actually happened (a holiday, an event). Weather is
  **dense**: it returns one row per requested date.
- `app/signals/merge.py::merge_signals(sales_rows, signal_frames,
  fill_defaults=None) -> pd.DataFrame` left-merges each signal frame onto
  `sales_rows` by `(location, date)`, broadcasting each signal row across
  every item row that shares its `(location, date)` -- row count and order
  are unchanged, no row multiplication. After merging, sparse-provider gaps
  are filled using `fill_defaults` (defaults to the module-level
  `DEFAULT_FILL = {"is_holiday": 0, "holiday_name": "", "event_count": 0,
  "event_impact": 0.0}` if not overridden; weather columns need no fill
  since they're already dense). The result is ready for
  `FeatureStore.upsert(...)`.

## Forecast model

Phase 4 adds the model registry (`app/models/base.py`), a transparent
factor model (`app/models/factor_model.py`), a walk-forward backtest
(`app/models/backtest.py`), a synthetic future-signal generator
(`app/features/synthetic_signals.py`), and the `forecast_latest.json`
output writer (`app/output/writer.py`).

### Model registry

`app/models/base.py::ForecastModel` is the abstract interface (PRD.md
section 6.1) every registered forecaster implements:

- `fit(history: pd.DataFrame) -> None` — fit on past FeatureStore-shaped
  rows (`location, date, item, qty_sold` plus whatever signal columns are
  present). Callers are responsible for only passing rows strictly before
  any date later passed to `predict`.
- `predict(future_features: pd.DataFrame, reference_today: date) -> pd.DataFrame`
  — forecast `p10, p50, p90` plus a factor-`attribution` list for each
  `(location, date, item)` row in `future_features` (no `qty_sold`
  column — that's what's being forecast). `reference_today` is used to
  compute each row's horizon offset `(row.date - reference_today).days`.

`MODEL_REGISTRY: dict[str, type[ForecastModel]]` maps a model name (e.g.
`config.yaml`'s `model_name: factor_model_v1`) to its class, so a factor
model, a future GBM quantile model, and a foundation model (PRD.md section
11 "Later") are interchangeable and comparable on the same backtest.

### FactorModel (`factor_model_v1`)

`app/models/factor_model.py::FactorModel` implements
`demand = baseline + Σ(factor_weight × factor_value) + residual` (PRD.md
section 5) with no ML dependency beyond numpy — every forecast is a
closed-form computation, chosen for interpretability.

**Fitted per-(location, item) state** (`fit`):
- `level` — mean `qty_sold` across the group's history.
- `trend` — OLS slope of `qty_sold` on day-index (`numpy.polyfit(...)[0]`;
  0.0 if fewer than 2 rows).
- `weekly_seasonal[weekday]` — mean deviation of `qty_sold` from `level`
  for each `date.weekday()` value seen (additive, defaults to 0.0 for
  weekdays absent from a short history).
- `residual_sigma` — population std-dev of
  `actual - (level + trend*day_index + weekly_seasonal[weekday])`,
  floored at `_MIN_RESIDUAL_SIGMA = 0.5` so a flat/short history never
  produces a zero-width band.

**Per-row forecast** (`predict`): `baseline = level + trend * day_index`
(day-index continuing the same numbering from `fit`, so `trend`
extrapolates correctly), plus up to four named, addend-style factor
contributions that become the plain-English `attribution` (PRD.md section
6.3):

| Factor | Formula | Capped? | Included in `attribution` when |
|---|---|---|---|
| `day_of_week` | fitted weekly seasonal deviation | No | always |
| `holiday` | `HOLIDAY_WEIGHT (0.25) * level * is_holiday` | No | `is_holiday` truthy or `holiday_name` set |
| `event` | `EVENT_WEIGHT (0.15) * level * event_impact` | No | `event_impact > 0` |
| `weather` | `-RAIN_WEIGHT (0.20) * level * is_rain`, then clipped | **Yes** — `WEATHER_CONTRIBUTION_CAP = 0.05` (±5% of `level`) | nonzero after clipping |

Weather is the only capped factor: PRD.md section 5/9 requires weather to
be down-weighted at the +7..+13 horizon this phase exclusively forecasts,
so the cap is unconditionally applied here (not offset-dependent) — a
later near-term (+1..+6) phase that re-weights weather as the day nears
would change how the cap is applied inside `FactorModel.predict`, not the
constant's value.

`p50 = max(0.0, baseline + day_of_week + holiday + event + weather)`.

**Band widening:** `spread = residual_sigma * (1 + BAND_WIDENING_RATE (0.05) * (offset - HORIZON_FLOOR (7)))`,
where `offset = (row.date - reference_today).days`; Fri/Sat/Sun rows get
an extra `_WEEKEND_SPREAD_MULTIPLIER (1.15)`. `spread` is floored at a
small positive epsilon (`_MIN_SPREAD = 1e-6`) so it is never negative.
`p10 = max(0.0, p50 + Z10 * spread)`, `p90 = p50 + Z90 * spread`, with
`Z10 = -1.2816`, `Z90 = 1.2816` (standard-normal 10th/90th percentile
z-scores). Because `spread >= 0` and `Z10 <= 0 <= Z90` always,
**`p10 <= p50 <= p90` holds by construction**, not by luck, for every row.

An unseen `(location, item)` combination at `predict` time raises a
`ValueError` rather than extrapolating a nonsense prediction.

`MODEL_REGISTRY["factor_model_v1"] = FactorModel` is registered at the
bottom of `app/models/factor_model.py`, matching `config.yaml`'s
`model_name: factor_model_v1`.

### Backtest

`app/models/backtest.py::walk_forward_backtest(history, model_cls,
location, eval_dates, min_train_rows=14) -> dict[str, float]` validates a
model on a venue's own history (PRD.md section 5): for each date in
`eval_dates`, a **fresh** `model_cls()` is fit only on
`history[(history.location == location) & (history.date < eval_date)]` —
the strict `<` is the complete no-lookahead guarantee, since the eval
date's actual `qty_sold` (read straight from `history`, never handed to
the model) is exactly what the freshly-fit model has never seen. A naive
same-day-last-week baseline (`qty_sold` from `eval_date - 7 days`, also
read straight from `history`) is scored the same way. Returns
`{"wmape": float, "skill_vs_naive": float, "n_eval_rows": float}`, where
`wmape = sum(|actual - forecast|) / sum(actual)` aggregated over every
scored `(item, eval_date)` pair, and
`skill_vs_naive = 1 - (model_wmape / naive_wmape)` (positive = model beats
naive).

### Synthetic future signals

`app/features/synthetic_signals.py::generate_synthetic_future_signals(location,
date_range, seed=0) -> pd.DataFrame` generates a dense (one row per date,
inclusive range), fully synthetic set of signal columns shaped exactly
like real Phase-3 `merge_signals` output (`is_holiday, holiday_name,
temp_c, precip_mm, is_rain, event_count, event_impact`), for FUTURE dates
a real provider can't yet supply. Deterministic given `seed`
(`numpy.random.default_rng`). Development/demo/backtest use only — a real
deployment calls the Phase-3 `SignalProvider`s for near-term dates
instead.

### Output writer and the `forecast_latest.json` contract

`app/output/writer.py` assembles and writes the owner-facing forecast
document:

```json
{
  "location": "demo_location",
  "generated_for_date": "YYYY-MM-DD",
  "window": {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"},
  "horizon": {"start_offset": 7, "end_offset": 13},
  "model": "factor_model_v1",
  "quantile_target": 0.6667,
  "skill_vs_naive": 0.0,
  "wmape": 0.0,
  "items": [
    {
      "item": "burger",
      "days": [
        {
          "date": "YYYY-MM-DD",
          "dow": "Sun",
          "p10": 0, "p50": 0, "p90": 0,
          "plan_for": 0,
          "why": [
            {"factor": "day_of_week", "direction": "up", "text": "Weekend day", "contribution": 0.0}
          ]
        }
      ]
    }
  ]
}
```

- `build_forecast_document(location, generated_for_date, window, horizon,
  model_name, quantile_target, skill_vs_naive, wmape, predictions) -> dict`
  assembles this as a plain JSON-ready dict from a `ForecastModel.predict`-
  shaped `predictions` DataFrame. `p10/p50/p90/plan_for` are rounded to
  ints for the JSON contract (`predict()` itself keeps floats, for
  backtest precision); `p10`/`p90` are re-clamped against the rounded
  `p50` after rounding as a final safety net, since independently rounding
  three floats can occasionally violate a `<=` that held before rounding.
- `write_forecast_json(document, path="app/output/forecast_latest.json") -> Path`
  writes the dict as indented JSON, creating parent directories if needed.
- **`quantile_target` is the real, computed `AgentConfig.critical_fractile`**
  (`q* = Cu / (Cu + Co)`, e.g. `0.6667` for the default
  `cost_underprep=2.0, cost_overprep=1.0`) — the `0.6667`/`0.75`-style
  numbers above are illustrative only, never hardcoded.

### Cost ratio → plan-for quantile

`compute_plan_for(p10, p50, p90, quantile_target) -> float` interpolates
the `quantile_target`-quantile order/prep quantity from the model's
p10/p50/p90 anchors (PRD.md section 5's newsvendor / critical fractile
`q* = Cu / (Cu + Co)`): piecewise-linear between `(0.1, p10)`-`(0.5, p50)`
for a target in `[0.1, 0.5]`, and `(0.5, p50)`-`(0.9, p90)` for a target
in `[0.5, 0.9]`; clamped to `p10`/`p90` outside `[0.1, 0.9]`. Monotonic by
construction since `p10 <= p50 <= p90`, with no scipy/normal-inverse-CDF
dependency needed. Feed it `AgentConfig.critical_fractile` to get the
cost-optimal `plan_for` number shown to the owner.
