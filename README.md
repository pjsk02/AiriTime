# AiriWheels Demand Forecast Agent

A self-learning agent that forecasts short-horizon restaurant food demand as a
*range* (P10/P50/P90), explains why, and gets more accurate every day by
grading itself against what actually sold. See `../PRD.md` for full product
context.

**This build is on Phase 6 "Self-learning loop"** (see PRD.md section 14):
Phase 1 shipped a runnable skeleton with a config loader and a `/health`
check; Phase 2 added the `SalesConnector` interface, a CSV connector, the
feature store, and a synthetic dev-data generator; Phase 3 added the
`SignalProvider` interface, holiday/weather/event providers, and the
`merge_signals` broadcast step (see "Feature store schema" and "Signal
columns" below); Phase 4 added the model registry, a transparent factor
model producing P10/P50/P90 with attribution, a walk-forward backtest
(wMAPE + skill-vs-naive), a synthetic future-signal generator for
dev/demo/backtest use, and the `forecast_latest.json` output writer (see
"Forecast model" below); Phase 5 added the owner-facing UI (see "Owner UI"
below); Phase 6 adds the self-learning loop — an append-only forecast
log, actuals ingestion, error attribution, safety-railed recalibration of
the factor model's weights, and a skill scorecard (see "Self-learning
loop" below). There is no Maritime wiring yet — that arrives in a later
roadmap phase.

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

## Owner UI

`ui/forecast.html` is a self-contained, single-file owner-facing view of
`forecast_latest.json` -- no build step, no npm/bundler, no external JS
files (inline `<style>`/`<script>` only; the only external resource is the
Google Fonts CDN). It renders the header stats, a dish-tab funnel strip
(P10/P50/P90 per day), a "why this number" attribution panel, a weekly P50
table, and an illustrative slow-day voucher callout -- all derived at
runtime from the fetched JSON, not hardcoded. This is a pure frontend
add-on; it does not change anything under `app/`, `tests/`, `config.yaml`,
or the `forecast_latest.json` contract itself.

### Regenerating `app/output/forecast_latest.json`

Follow the Phase 4 pipeline documented above under "Forecast model": fit a
registered `ForecastModel` (e.g. `factor_model_v1`) on feature-store
history, `predict()` over the +7..+13 horizon, then call
`app/output/writer.py`'s `build_forecast_document(...)` and
`write_forecast_json(document)` (defaults to `app/output/forecast_latest.json`).
See `app/models/base.py`, `app/models/factor_model.py`, and
`app/output/writer.py` for the exact functions, and the "Output writer and
the `forecast_latest.json` contract" section above for the JSON shape.

### Refreshing the UI's copy of the JSON

The UI reads `ui/forecast_latest.json` (a copy, kept alongside the HTML so
`fetch()` can resolve it with a relative path). After regenerating the real
file, refresh the copy:

```bash
# from the repo root
cp app/output/forecast_latest.json ui/forecast_latest.json
```

(PowerShell: `Copy-Item app/output/forecast_latest.json ui/forecast_latest.json -Force`)

### Opening `ui/forecast.html`

Because the page loads its data via `fetch('forecast_latest.json')`, some
browsers (notably Chrome) block that fetch under `file://` due to CORS,
even though the file sits right next to the HTML. If you open
`ui/forecast.html` directly by double-clicking it and see an inline "could
not load forecast data" message instead of the dashboard, serve the `ui/`
folder over a local static server instead:

```bash
cd ui
python -m http.server
```

Then visit `http://localhost:8000/forecast.html` in your browser.

### Validating the JSON contract

`ui/check_contract.py` is a standalone stdlib-only script (no pandas/
pydantic/app dependencies) that validates a `forecast_latest.json` file
against the contract documented above -- required keys, types, `date`/`dow`
formats, `p10 <= p50 <= p90`, and `why`-entry shape -- and exits non-zero
with a specific error on any violation:

```bash
python ui/check_contract.py                              # checks ui/forecast_latest.json
python ui/check_contract.py app/output/forecast_latest.json  # checks the real Phase 4 output
```

On success it prints a one-line summary ending in `OK` and exits `0`; on
failure it prints the specific key/item/day/field that failed to stderr
and exits `1`.

## Self-learning loop

Phase 6 adds `app/learning/` (PRD.md section 6.4): an append-only
forecast log, actuals ingestion, error attribution, safety-railed
recalibration of the factor model's three learned weights, and a skill
scorecard — the loop that lets the agent grade itself against what
actually sold and get more accurate over time.

```
predict() ─► ForecastLog (append)
                                       ActualsStore (upsert, webhook)
                                                │
                          join_forecast_actuals ┘
                                │
                    attribute_error (per-factor)
                                │
                          recalibrate()
                                │
                   WeightsStore.put(...) (append new version)
                                │
                  FactorModel(weights=...) ◄── next predict() cycle
                                │
                        Scorecard.append(...)
```

All four stores (`ForecastLog`, `ActualsStore`, `WeightsStore`,
`Scorecard`) are backed by simple, human-auditable JSON Lines files (one
JSON object per line, plain `json`/`pathlib`, no new dependency):

- **`app/learning/forecast_log.py::ForecastLog`** — append-only log of
  every forecast ever produced, keyed conceptually by `(location, date,
  item, generated_for_date)`. Multiple entries legitimately exist for the
  same `(location, date, item)` across different `generated_for_date`
  values, since the rolling +7..+13 horizon re-forecasts the same
  calendar day daily as it approaches — this is a log, not an upsert
  store; nothing is ever overwritten. `p10/p50/p90` are kept at full
  float precision (this is a separate, internal, higher-precision record
  for the learning loop only — it is not the Phase-4
  `forecast_latest.json` contract file, which is untouched and still does
  its own int-rounding independently). `append(predictions,
  generated_for_date, weights_version, model_name)` writes one line per
  row of a `ForecastModel.predict`-shaped DataFrame; `read(location=None,
  date_range=None)` reads rows back (optionally filtered by location
  and/or an inclusive range on the forecasted `date`), with `attribution`
  deserialized back into a Python list of dicts.
- **`app/learning/actuals.py::ActualsStore`** — realized sales, *upserted*
  keyed by `(location, date, item)` (unlike `ForecastLog`, there is
  exactly one true actual per key, so a later ingest legitimately
  overwrites — e.g. a corrected point-of-sale reconciliation — matching
  `FeatureStore`'s own upsert semantics). `ingest(rows)` takes `location,
  date, item, qty_sold`; `read(location, date_range)` mirrors
  `FeatureStore.query`'s shape. `join_forecast_actuals(forecast_rows,
  actuals_rows)` inner-joins logged forecasts with actuals on `(location,
  date, item)`, adding `actual_qty` (renamed from `qty_sold`), `error =
  actual_qty - p50`, and `abs_error = abs(error)` — rows with no actual
  yet or no forecast logged are dropped, the correct semantics for a
  "graded forecasts" frame.
- **`app/learning/weights_store.py::WeightsStore`** — append-only,
  versioned history of the three recalibratable weights
  (`holiday_weight, event_weight, rain_weight`). `put`/`rollback_to`
  never mutate or delete a prior record, they always append a new
  version, so any past version stays retrievable via `get(version)`
  forever — a caller can reconstruct the exact `FactorModel(weights=...)`
  that produced any historical forecast. `latest()`/`get(version)`/
  `history()` read back; `put(weights, reason)` appends version
  `latest().version + 1`; `rollback_to(version)` appends a *new* record
  reproducing an old version's weights (so the rollback itself is a new,
  auditable event), distinct from just calling `get(version)` directly to
  reuse old weights without recording anything.
- **`app/learning/scorecard.py::Scorecard`** — append-only skill history,
  one row per recalibration cycle: `append(cycle_date, wmape,
  skill_vs_naive, weights_version)`, `read()` returns all entries sorted
  by date ascending.

### Error attribution (`app/learning/attribution.py`)

Two functions:

- **`reconcile_forecast_row(model, location, item, target_date, p50, why,
  tolerance=1e-6)`** — independently verifies that a logged forecast
  row's `why`/`attribution` list fully accounts for its `p50`, using the
  new `FactorModel.baseline_for(location, item, target_date)` accessor
  (computed from the model's fitted state — level, trend,
  weekly_seasonal — independently of `why`/`p50`) as ground truth. Returns
  `(reconstructed_p50, is_consistent)` where `reconstructed_p50 =
  baseline_for(...) + sum(contribution for contribution in why)` (clamped
  at `0.0` to match `predict()`'s own `max(0.0, ...)` clamp) and
  `is_consistent = abs(reconstructed_p50 - p50) <= tolerance`. This is a
  real check, not a tautology: if a future factor were ever added to
  `predict()` but its contribution omitted from `why`, this reconciliation
  would catch the shortfall.
- **`attribute_error(row)`** — apportions a graded forecast's `error`
  (`actual_qty - p50`) across the recalibratable factors
  (`"holiday", "event", "weather"` — **not** `day_of_week`, which has no
  scalar weight of its own; it is re-fit fresh from history on every
  `fit()` call, already a form of continuous recalibration) present in
  that row's `why` list, proportional to each factor's share of total
  recalibratable-contribution magnitude: `share_f = |contribution_f| /
  sum(|contribution_g| for active g)`, `factor_error_f = error *
  share_f`. Returns a dict with a key only for factors that were active
  (nonzero contribution) in that row — an absent key means "not active
  for this row", distinct from a `0.0` value meaning "active with zero
  apportioned error". Returns `{}` if no recalibratable factor was active
  (e.g. a plain weekday with no holiday/event/rain).

**Sign convention (the crux of recalibrating in the right direction):**
`recalibrate.py` combines `attribute_error`'s output with each factor's
own `contribution` sign via `direction_signal_f = factor_error_f *
sign(contribution_f)`.
- `direction_signal_f > 0` — the factor pushed the forecast in some
  direction, and the actual missed *even further* in that same
  direction: the factor's effect was too **weak** → **increase** that
  weight.
- `direction_signal_f < 0` — the factor's effect ran against (or wasn't
  supported by) the realized miss: the factor's effect was too
  **strong** → **decrease** that weight's magnitude, floored at `0.0`
  (a weight must stay non-negative — a negative `holiday_weight`, for
  example, would mean holidays *suppress* demand, inverting the factor's
  documented meaning).

Worked example: `holiday_contribution = +5.0`, `error = +8.0` (actual
came in even higher than the already-holiday-boosted p50) → `holiday`
is under-weighted, `direction_signal > 0`, increase `holiday_weight`.
Conversely `weather_contribution = -3.0`, `error = +2.0` (actual still
came in above the already-suppressed p50) → rain's suppression was too
strong, `direction_signal < 0`, decrease `rain_weight`.

### Recalibration (`app/learning/recalibrate.py`)

`recalibrate(graded_rows, current_weights, max_step=0.10,
min_samples=5, gentle_fraction=0.3, drift_window=10,
drift_threshold=0.15, learning_rate=0.5) -> {"new_weights": {...},
"updates": {...}}` computes a new weights dict from graded (forecast vs.
actual) history under five hard safety rails, entirely deterministic (no
randomness) given the same inputs:

| Rail | Default | Protects against |
|---|---|---|
| `min_samples` | `5` | Updating a weight off too little evidence — a factor with fewer than `min_samples` active rows is left **completely unchanged** (not even a zero-sized update; identical bit-for-bit to `current_weights`). |
| `max_step` | `0.10` (10%) | A single cycle ever moving a weight too far, regardless of how strong the raw error signal looks — the largest *relative* change a weight may take in one recalibration. |
| `gentle_fraction` | `0.3` | Overreacting to routine noise — normal (non-drift) cycles are capped at `max_step * gentle_fraction` (3%), so weights drift slowly absent a persistent signal. |
| `drift_window` / `drift_threshold` | `10` / `0.15` | Under-reacting to a real, persistent shift — if the mean of the most recent `drift_window` per-row normalized signals exceeds `drift_threshold` in magnitude, the full `max_step` (not just the gentle fraction) is unlocked for that cycle. |
| Non-negative floor | `0.0` | A weight ever going negative, which would invert the factor's documented meaning (see the sign convention above). Flooring at `0.0` can only pull a proposed step *smaller* in magnitude, never larger, so it can never itself push the actual applied step past `max_step`. |

Each per-row signal is `direction_signal_f` (see the sign convention
above) normalized by `max(abs(row.p50), epsilon)` into a small
dimensionless number before averaging — `learning_rate` then scales the
mean signal into a proposed relative weight change, which is clamped to
the cycle's applicable step cap (`max_step` or the gentle fraction) and
applied as `new_weight = max(0.0, current_weight * (1 +
relative_delta))`. The returned `"updates"` dict always has an entry for
every one of the three weight keys (`applied_step=0.0, drift=False` when
guarded/unchanged), so a caller can audit exactly what happened — and
didn't happen — each cycle.

### `FactorModel(weights=...)` and `baseline_for`

To make recalibrated weights actually usable, `app/models/factor_model.py`
gained (Phase 6, fully backward-compatible — `FactorModel()` with no
arguments behaves exactly as before):

- A module-level `DEFAULT_WEIGHTS = {"holiday_weight": HOLIDAY_WEIGHT,
  "event_weight": EVENT_WEIGHT, "rain_weight": RAIN_WEIGHT}`, packaging
  the three recalibratable constants for injection.
  **`WEATHER_CONTRIBUTION_CAP` is deliberately excluded** and can never
  become recalibratable — it's a fixed policy rail (weather is
  down-weighted at +7..+13 by design, PRD.md section 5/9), not a learned
  parameter. Recalibration may adjust *how strongly* rain suppresses
  demand (`rain_weight`), never the hard ceiling on that suppression.
- `FactorModel(weights: dict[str, float] | None = None)` — `None` (the
  default) reproduces the original fixed-constant behavior exactly;
  passing any subset of `{"holiday_weight", "event_weight",
  "rain_weight"}` overrides those keys (others fall back to
  `DEFAULT_WEIGHTS`). This is the hook the recalibration loop uses to
  `fit()`/`predict()` with a previously-persisted or newly-recalibrated
  weights version (`WeightsStore`) without needing a new `ForecastModel`
  subclass.
- `baseline_for(location, item, target_date) -> float` — returns the
  fitted `level + trend * day_index` baseline for one row, with no
  factor contributions added, sharing its computation with `predict()`
  via a private `_baseline` helper (so there is exactly one
  implementation of "baseline"). Used by `reconcile_forecast_row` above
  as an independent ground truth. Raises the same `ValueError` as
  `predict()` for an unfitted `(location, item)`.
