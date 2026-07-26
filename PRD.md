# PRD — AiriWheels Restaurant Customer Demand Forecast Agent

**Version:** 0.1 (draft)
**Date:** 26 Jul 2026
**Status:** Pre-build / hackathon
**One line:** A self-learning agent that forecasts short-horizon restaurant food demand as a *range* (not a single number), explains why, and gets more accurate every day by grading itself against what actually sold.

---

## 0. How to use this document (note to the build agent)

This PRD is the source of truth for what we are building and why. Read it before writing code and refer back to it when a decision is ambiguous. Where this document and a prompt conflict, ask before proceeding. Terms written in **bold-italic** the first time they appear are defined in the Glossary (Section 12); use them precisely and consistently in code, comments, and API names. Numbers shown in examples are illustrative unless a section says otherwise.

---

## 1. Product summary

AiriWheels sells software to restaurants. It has three surfaces: a website, a customer-facing app (which issues vouchers/discounts to frequent customers as a loyalty mechanism), and a restaurant-owner app. This project adds a **demand-forecast agent** that predicts how much of each menu item a restaurant will sell, so owners can order ingredients, set staff rotas, and run promotions with confidence instead of guesswork.

The agent is built on a **_financial-engineering philosophy_**: treat demand like a quantitative forecasting problem. Forecast a full distribution (not a point estimate), size decisions to the *cost of being wrong*, backtest rigorously, measure skill against a naive baseline, and continuously recalibrate from realized error. This framing is also the product's main differentiator and must remain central.

---

## 2. Problem and why now

- Food is a restaurant's largest controllable cost (typically 28–35% of revenue) and its top profitability worry in 2026. Preventable food waste runs roughly 4–10% of purchases (~2–4% of revenue).
- Most operators forecast poorly: industry surveys put self-reported sales-forecast accuracy around 60%, even among those using tools. The realistic prize is a 1–3 percentage-point reduction in food cost.
- The two failure modes are asymmetric: **over-prep** wastes food; **under-prep** causes stockouts / "86'd" items / lost customers, which usually costs more. A distributional, cost-aware forecast is built to trade these off optimally.
- Incumbent tools (e.g. Crunchtime, Lineup.ai, ClearCOGS, Restaurant365, Supy) mostly emit point estimates from POS data alone. Our wedge: distributional forecasts + a genuine loyalty/voucher data advantage + a self-auditing, interpretable learning loop.

---

## 3. Users and the two planes

The product has two distinct surfaces that never touch. This separation is a hard requirement.

**Owner plane — the Owner app (consume).** Audience: a busy, non-technical restaurant owner/GM. They *read* the forecast and act on it. They receive a range per item per day, plain-English reasons, staffing hints, and a voucher lever for slow days. They do **not** get model dials — well-meaning manual overrides are a known cause of forecast degradation, and operator trust depends on explanations, not controls.

**Engineer plane — the Engineer studio (build/config).** Audience: AiriWheels engineers. They connect data sources, toggle signal providers, choose and tune a model from the registry, set the cost ratio and horizon, backtest, and review model health/drift. Changes here propagate to owners only through an explicit, backtest-gated **publish** step — never live on every keystroke.

---

## 4. Goals and non-goals

### Goals (v1)
- Produce a per-item, per-day demand forecast as **P10 / P50 / P90** for the **_rolling horizon_** of days **+7 to +13** (the week that starts 7 days from "today").
- Refresh daily; the same calendar day re-forecasts each morning and its range **tightens** as it approaches.
- Attach a plain-English reason (**_factor attribution_**) to every forecast.
- Log every forecast, ingest **_actuals_**, and **_recalibrate_** from per-factor error.
- Report **_skill_** vs a naive same-day-last-week baseline, plus wMAPE and a **_drift_** status.
- Extensible input layer: adding a POS connector, a signal provider, or a model is a *new file against an interface*, not a rewrite.
- Deploy cheaply and run near-unattended on **Maritime** (cron generates forecasts; webhook ingests actuals and recalibrates).

### Non-goals (v1 / hackathon)
- No automated purchase-order placement or automated labor scheduling (we *recommend*, we don't *execute*).
- No no-code "edit any model" UI. Engineers author models in the repo/code; the studio only *selects and tunes* registered models.
- No intraday/daypart forecasting in v1 (day-level only).
- No multi-location roll-up analytics in v1.
- Numbers in the prototype UI are placeholders until real POS data is wired in.

---

## 5. Core concepts (the financial-engineering framing — read carefully)

These map quant tools onto demand. Implement them by these definitions.

- **Factor model.** `demand = baseline + Σ(factor_weight × factor_value) + residual`. Baseline captures level + trend + weekly seasonality. Factors include day-of-week, holidays, confirmed local events, seasonality, weather, and loyalty signals. Chosen for interpretability: every forecast decomposes into named contributions.
- **Distributional forecast (P10/P50/P90).** Output a distribution per item/day, not one number. P50 = most likely; P10–P90 = likely range. Wider band = less certain (further out, or more volatile day).
- **Newsvendor / critical fractile.** The right order/prep quantity is a specific quantile of the demand distribution set by the cost ratio: **q\* = Cu / (Cu + Co)**, where **Cu** = cost of under-prep (lost margin on a stockout) and **Co** = cost of over-prep (wasted food). Higher relative cost of running out → order at a higher quantile. This is how a range becomes a single actionable number.
- **Rolling horizon.** We serve days +7..+13. Each morning the window rolls forward one day and every day inside it is re-forecast. A given calendar day therefore first appears at +13 and is re-forecast daily until +7 (and, in later versions, down to +1) — its band narrows over time as more is known (especially weather).
- **Backtesting (walk-forward).** Validate on the venue's own history before trusting a config: fit on past, predict forward, compare to what happened, out-of-sample.
- **Skill.** Error reduction versus a naive baseline (same-day-last-week). Our "is this actually good?" metric — the demand-forecasting analog of a risk-adjusted return. Report alongside wMAPE.
- **Factor attribution of error.** When actuals arrive, decompose the miss by factor ("we over-forecast 12%, mostly from the event term"). This drives both the owner-facing "why" and the recalibration.
- **Recalibration.** Update factor weights from realized error. Gentle/online update on normal drift; heavier retrain when a **drift detector** (e.g. windowed error monitoring / distribution test on residuals) fires.

---

## 6. Functional requirements

### 6.1 Input layer (three separate contracts)
- **Data connectors** — source item-level sales. Must sit behind one `SalesConnector` interface (`fetch() -> normalized rows`). v1: CSV/Excel upload + one real POS (Toast or Square). Loyalty/voucher data is an input here too.
- **Signal providers** — external context, one `SignalProvider` interface (`fetch(location, date_range) -> features`). Queried by restaurant location + date range; results fan out and attach as feature columns. v1 providers: events (Ticketmaster, later PredictHQ), holidays (Nager.Date), weather (Open-Meteo for dev; Visual Crossing for commercial). **Weather is down-weighted at the +7..+13 horizon** and re-weighted as the day nears.
- **Model registry** — forecasters behind one interface (`fit(history, features)`, `predict(features) -> {p10,p50,p90}`) so a factor model, a gradient-boosted quantile model, and a foundation model are interchangeable and comparable on the same backtest.

### 6.2 Feature store
A single store keyed by **(location, date, item)**. Connectors, signal providers, and loyalty data all write columns into it; models read from it. This is what makes the input layer extensible rather than a pile of special cases. The exact column schema is defined during the input-layer build phase and documented in the repo README; downstream phases must key off those exact names.

### 6.3 Forecast output
- Per item, per day in the +7..+13 window: `p10`, `p50`, `p90`, a recommended order/plan quantity (from q\*), and a factor-attribution list.
- A weekly view: per-item week total (from P50), the biggest night, the plan-for number, and a staffing hint.
- Owner-facing reasons in plain language derived from the top contributing factors.

### 6.4 Self-learning loop
- **Forecast log:** persist every forecast with its factor breakdown and the config version that produced it.
- **Actuals ingestion:** accept realized sales (via webhook/endpoint) keyed by (location, date, item).
- **Error attribution:** compute per-item and per-factor error.
- **Recalibration:** update weights (online for drift, retrain on detected shift) and append to a skill scorecard.

### 6.5 Owner app (consume)
Rolling +7..+13 view; per-dish range chart with narrowing bands; tap-a-day detail with P10/P50/P90 and a plan-for recommendation; plain-English "why"; a slow-day **voucher lever** (identify the week's low night and offer to send vouchers via the customer app — the loyalty differentiator); a whole-menu weekly table. No model controls.

### 6.6 Engineer studio (build/config)
Connect/toggle data connectors; toggle signal providers with weights; select+tune a registered model; set cost ratio (→ q\*) and horizon; run a backtest and view skill/wMAPE/drift; **publish** (gated by backtest) to push a new config/forecast to the owner app. No arbitrary model-code editing in the UI.

---

## 7. Technical architecture

```
CONNECTORS ─┐
SIGNALS ────┼─► FEATURE STORE ─► MODEL (from registry) ─► P10/P50/P90 + attribution per item/day
LOYALTY ────┘        ▲                                              │
                     │                                              ▼
      ENGINEER CONFIG (factors, model, cost ratio,            FORECAST LOG
       horizon=+7..+13) — published, backtest-gated                │
                     ▲                                              ▼
                     └──── recalibrate ◄── error attribution ◄── ACTUALS (webhook)
                                                                    │
                                                            OWNER APP (order/plan sheet)
```

**Stack:** Python 3.12, FastAPI (serve on port **8080**, required by Maritime), pandas for the feature store/backtests, pydantic for config, pytest for tests, Docker for packaging. Add heavier ML libs only when the model phase needs them.

**Suggested repo layout** (finalized during build):
```
airiwheels-agent/
  app/
    main.py                 # FastAPI: /health, /run, /actuals, /forecast/latest
    config.py               # loads config.yaml (pydantic)
    connectors/             # SalesConnector interface + implementations
    signals/                # SignalProvider interface + implementations
    features/               # feature store + synthetic dev data
    models/                 # model registry + factor model
    learning/               # forecast log, error attribution, recalibration
  config.yaml               # engineer-plane config (versioned)
  tests/
  Dockerfile                # python:3.12-slim, uvicorn on 8080
  README.md                 # incl. feature-store schema
  PRD.md                    # this document
```

---

## 8. Maritime deployment model

Maritime hosts AI agents on sleep/wake micro-VMs (cheap: ~$1–10/month), auto-detects the framework/container, and provides cron schedules, webhooks, an API endpoint, encrypted secrets, and per-agent isolation. Mapping:

- **Cron (daily, ~06:00):** wake the agent → pull sales + signals + loyalty → run the model → write P10/P50/P90 + attribution → publish the +7..+13 sheet to the owner app → sleep.
- **Webhook (`/actuals`):** end-of-day POS/loyalty events trigger error attribution + recalibration + skill-scorecard update.
- **API endpoint (`/forecast/latest`):** the owner app fetches the current sheet on demand.
- **Multi-tenant:** one isolated agent instance per restaurant; each keeps its own data and learned weights. Engineer config changes flow to all instances on redeploy.

The Maritime setup phase is **human-in-the-loop** (real account, GitHub repo, live API keys, external calls) — it must not run unattended, and the exact CLI/cron syntax should be taken from Maritime's current docs at build time, not from memory.

---

## 9. Data and external sources

- **Sales:** item-level POS transactions; ~12 months of clean history is the practical minimum for trustworthy forecasts. CSV upload for un-integrated venues.
- **Loyalty/voucher:** repeat-visit cadence, voucher redemptions, cohort trends — AiriWheels-only signals.
- **Weather:** Open-Meteo (free, non-commercial dev use; commercial needs a paid plan + attribution) / Visual Crossing (commercial-safe free tier, then usage-based).
- **Events:** Ticketmaster Discovery API (free tier) for concerts/sports/theatre; PredictHQ (demand intelligence) later.
- **Holidays:** Nager.Date (free, unlimited, 100+ countries).
- Confirm current API terms/pricing at integration time.

---

## 10. Success metrics

- **Primary (product):** demonstrated reduction in food cost / waste for a pilot venue (target: 1–3 pts of food cost); reduction in stockout/86 events.
- **Model:** beat the naive same-day-last-week baseline by ≥5 wMAPE points on pilot backtest; item-level wMAPE trending below ~15% after recalibration (>25% signals a data/granularity problem).
- **Loyalty edge:** voucher-lift and repeat-visit factors measurably improve accuracy or promo ROI within ~2 promo cycles (else de-emphasize in the pitch but keep the promotion-simulation feature).
- **System:** forecast refreshes on schedule; actuals ingestion + recalibration run without manual intervention; per-tenant cost stays in the $1–10/month range.

---

## 11. Scope tiers

**Hackathon v1 (build now):** clean interfaces with one concrete path down each — 2–3 connectors (one real POS + CSV), 3 signal providers (events, holidays, weather-demoted), 1 model (transparent factor model with quantiles), config.yaml as the engineer plane, the +7..+13 owner sheet, the self-learning loop, Dockerized FastAPI, deployed on Maritime.

**Later:** additional POS connectors and models (GBM quantiles, foundation model); a real Engineer-studio UI; near-term (+1..+6) prep view with weather re-weighted; automated ordering/labor; multi-location analytics; PredictHQ + commercial weather tiers.

---

## 12. Risks and caveats

- **Data quality, not the model, is the top failure cause.** Incomplete/dirty POS history breaks forecasts. Validate first.
- **Small-venue noise:** low-volume items give unstable per-item MAPE; use wMAPE at portfolio level and focus item-level forecasting on high-velocity items.
- **Agentic-AI reliability:** keep the self-learning loop simple, observable, and cheap; a bad auto-recalibration can silently degrade every forecast, so gate weight updates and log them.
- **Weather is unreliable at +7..+13** — down-weight it there by design.
- **Vendor accuracy claims are marketing** — always measure at the decision level on the customer's own data.
- **The finance analogy has limits:** demand is not a tradable asset (capacity limits, censored demand for 86'd items, menu changes break stationarity). Use the framing as a design/interpretability tool, not a literal market model.

---

## 13. Glossary

- **Actuals:** realized sales for a past day/item, used to grade and recalibrate.
- **Drift:** a change in demand behavior that makes past-fit weights stale; detected by monitoring error over time.
- **Factor attribution:** decomposition of a forecast (or a forecast error) into named factor contributions.
- **Financial-engineering philosophy:** treating demand forecasting with quant discipline — distributions, cost-based sizing, backtesting, skill scoring, recalibration.
- **Newsvendor / critical fractile (q\*):** the cost-optimal order quantile, q\* = Cu/(Cu+Co).
- **P10/P50/P90:** the 10th/50th/90th percentiles of the demand distribution (quiet / most-likely / busy).
- **Recalibrate:** update model weights from realized error.
- **Rolling horizon:** a forecast window that advances daily; each day is re-forecast repeatedly, tightening as it nears.
- **Skill:** error reduction vs a naive baseline; the "is it actually good" metric.
- **wMAPE:** weighted mean absolute percentage error; the portfolio-level accuracy metric preferred over per-item MAPE for low-volume items.

---

## 14. Build roadmap (phased; each phase ships green tests before the next)

1. **Scaffold** — runnable skeleton, config loader, `/health`.
2. **Input layer** — `SalesConnector` + CSV connector + feature store + synthetic dev data.
3. **Signal providers** — holidays / weather / events behind `SignalProvider`, mocked in tests.
4. **Forecast model** — factor model → P10/P50/P90 for +7..+13, behind the model registry, plus a walk-forward backtest (wMAPE, skill-vs-naive).
5. **Self-learning loop** — forecast log + actuals ingestion + per-factor error attribution + recalibration.
6. **Service + Docker** — FastAPI (`/run`, `/actuals`, `/forecast/latest`) containerized on 8080.
7. **Maritime deploy** — GitHub link, secrets, daily cron, actuals webhook, verify live. *(Human-in-the-loop; not unattended.)*
8. **Wire the prototype** — point the owner-app UI at the live endpoint.
