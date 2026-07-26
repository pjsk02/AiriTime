# Deployment runbook — Maritime

This is the deployment runbook for shipping the AiriWheels demand-forecast
service (FastAPI, Docker, port 8080 — see `Dockerfile`, `PRD.md` section 8)
to [Maritime](https://maritime.sh), the sleep/wake micro-VM host named in
PRD.md section 8 as the target platform.

**This document is a human-run checklist, not automation.** Nothing in this
file executes a live deploy on its own. Steps that require a real Maritime
account, API token, or live credentials are marked **HUMAN STEP**. Do not
script or unattended-run those steps (see PRD.md section 8: "The Maritime
setup phase is human-in-the-loop... it must not run unattended").

**Do not run the "Live deploy checklist" (last section) until `feat/service`
is merged to `main`.** The container that gets deployed is whatever's on
`main`'s `Dockerfile` + `app/`; deploying before the service endpoints
(`/run`, `/actuals`, `/forecast/latest`) land on `main` would deploy an
incomplete service.

Every Maritime-specific claim below cites the doc page it came from. Where
Maritime's docs don't spell out a detail (this happened for exact cron
syntax and webhook path-targeting — see the VERIFY notes), this file says so
explicitly instead of guessing.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Link the GitHub repo](#2-link-the-github-repo)
3. [Set secrets and environment variables](#3-set-secrets-and-environment-variables)
4. [Configure the daily cron (`POST /run`, ~06:00)](#4-configure-the-daily-cron-post-run-0600)
5. [Configure the actuals webhook (`POST /actuals`)](#5-configure-the-actuals-webhook-post-actuals)
6. [Verify the live deployment](#6-verify-the-live-deployment)
7. [Multi-tenant note: one agent per restaurant](#7-multi-tenant-note-one-agent-per-restaurant)
8. [Maritime doc sources cited in this runbook](#8-maritime-doc-sources-cited-in-this-runbook)
9. [RUN ONLY AFTER feat/service IS MERGED TO MAIN — live deploy checklist](#9-run-only-after-featservice-is-merged-to-main--live-deploy-checklist)

---

## 1. Prerequisites

- A Maritime account and an API token (`MARITIME_TOKEN`, format `mk_...`),
  used to authenticate the CLI/API non-interactively.
  **[VERIFY]** exact token-creation flow (dashboard page) — not covered on
  the doc pages fetched for this runbook; check the Maritime dashboard's
  account/token settings at deploy time.
  (Source: `MARITIME_TOKEN` named as the CLI/fleet auth token —
  https://maritime.sh/docs/fleet)
- The Maritime CLI installed and authenticated.
- This repo pushed to a GitHub remote you control, with `main` containing a
  working `Dockerfile` (already present, `python:3.12-slim`, `EXPOSE 8080`,
  `uvicorn app.main:app --host 0.0.0.0 --port 8080`).
- **HUMAN STEP** — all of the above (account creation, token generation,
  confirming GitHub remote access) requires your own Maritime account and
  GitHub credentials.

---

## 2. Link the GitHub repo

Maritime deploys from a Dockerfile-containing repo via its CLI; there is no
documented dashboard-driven OAuth "connect repo" flow — the linking happens
through `maritime create`/`maritime deploy` with an explicit `--repo` flag.

**HUMAN STEP** — run from your own machine, authenticated to your own
Maritime account:

```bash
maritime create airiwheels-agent \
  --repo https://github.com/<your-org>/<your-repo> \
  --port 8080
```

- `--repo` takes the GitHub URL directly; the repo must contain a
  Dockerfile for a custom (non-template) deploy.
  (Source: https://maritime.sh/docs/quickstart, https://maritime.sh/docs/cli)
- `--port 8080` — Maritime's documented CLI default is already 8080, so this
  flag is redundant with our Dockerfile but is included for explicitness/
  future-proofing if the default ever changes.
  (Source: `--port` flag doc, "Port your app listens on (exposed publicly;
  default 8080)" — https://maritime.sh/docs/cli)

To redeploy after pushing new commits to `main`:

```bash
maritime deploy airiwheels-agent --source github --repo https://github.com/<your-org>/<your-repo>
```

(Source: https://maritime.sh/docs/cli)

**[VERIFY]** Whether Maritime auto-detects the Dockerfile's `EXPOSE`/`CMD`
port or relies purely on the `--port` flag/default is not explicitly stated
in the docs fetched — only that "any repo with a Dockerfile can be deployed
as a public web agent: Maritime builds it, runs it serverlessly, and serves
it on a public no-login URL" (https://maritime.sh/docs/quickstart,
https://maritime.sh/docs/build). Since our Dockerfile already exposes 8080
(Maritime's documented default), this is low-risk either way, but confirm
the served port matches 8080 in step 6 below before relying on it.

---

## 3. Set secrets and environment variables

Maritime stores secrets encrypted in its own store, keyed to the agent —
**not** committed to the repo. Values you set here come from
`.env.example` (see that file for the full list; copy it, fill in real
values locally, and set them one at a time — never commit the filled-in
file).

**HUMAN STEP** — requires your live API keys / credentials. Example for one
var (repeat per variable in `.env.example` that has a real value for your
deployment):

```bash
maritime env set airiwheels-agent VISUAL_CROSSING_API_KEY=<your-real-key>
```

- Env vars are secret-by-default and encrypted; secret values are always
  returned masked in listings (key names stay visible, values don't).
  (Source: https://maritime.sh/docs/configuration,
  https://maritime.sh/docs/cli)
- To set a non-secret var (e.g. a log level) without encryption:
  `maritime env set airiwheels-agent LOG_LEVEL=info --no-secret`
  (Source: https://maritime.sh/docs/cli)
- List what's set (values masked): `maritime env list airiwheels-agent`
  (Source: https://maritime.sh/docs/cli)
- Apply changed env vars to a running agent without a full redeploy:
  `maritime env reload airiwheels-agent`
  (Source: https://maritime.sh/docs/cli)
- API equivalent for programmatic provisioning:
  `POST /api/v1/agents/{agent_id}/env` with an `is_secret` boolean field.
  (Source: https://maritime.sh/docs/api/provisioning)

---

## 4. Configure the daily cron (`POST /run`, ~06:00)

PRD.md section 8 calls for a daily ~06:00 cron that wakes the agent, runs
the forecast pipeline, and publishes the +7..+13 sheet — this maps directly
to calling `POST /run` (implemented in `app/main.py`; see
`app/service/pipeline.py::run_forecast_pipeline`).

Maritime's docs confirm cron schedules exist as a trigger type but are
**dashboard-configured**, and no page fetched during research spelled out
the exact schedule syntax (crontab-style vs. a custom format) or how a
cron trigger's target path/method is specified:

> "Cron schedules, webhooks, and email triggers... are configured through
> the dashboard."
> (Source: https://maritime.sh/docs/configuration; trigger listing also at
> https://maritime.sh/docs/channels)

The CLI only exposes a way to **list** existing triggers
(`maritime triggers airiwheels-agent`), not to create a cron trigger.
(Source: https://maritime.sh/docs/cli)

**[VERIFY]** exact cron syntax and target-path configuration — not
documented in text form as of this research. **HUMAN STEP:**

1. Open the Maritime dashboard for `airiwheels-agent`.
2. Find the Triggers / Schedules section.
3. Create a new cron trigger targeting `POST /run` on this agent, scheduled
   for approximately `06:00` daily (confirm the timezone the dashboard
   uses — likely UTC, but verify in the UI before relying on the exact
   hour).
4. Save, then confirm it appears in `maritime triggers airiwheels-agent`.

---

## 5. Configure the actuals webhook (`POST /actuals`)

PRD.md section 8 calls for a webhook that ingests end-of-day POS/loyalty
events and triggers `POST /actuals` (implemented in `app/main.py`; payload
shape below matches `ActualsStore.ingest`, see README.md "Self-learning
loop").

Maritime's docs describe **two different things** called "webhook," and
research could not confirm either one maps cleanly to "call POST /actuals
on my running container on a schedule/event from an external POS system":

- **Outbound SDK webhooks** — Maritime calls *your* external URL when
  agent lifecycle events happen (e.g. `agent.error`, `agent.deployed`),
  HMAC-SHA256 signed via `X-Maritime-Signature`. This is Maritime notifying
  you, not a way for an external system to reach your agent's `/actuals`.
  (Source: https://maritime.sh/docs/sdk/webhooks)
- **Inbound public webhook** — `POST /api/webhooks/{agent_id}` is a
  public, no-auth invoke URL for the agent (docs describe it as
  chat/invoke-style, treating the URL itself as the secret), not
  documented as forwarding to an arbitrary in-container path like
  `/actuals`.
  (Source: https://maritime.sh/docs/api)
- The dashboard-configured "webhook trigger" mentioned alongside cron
  (https://maritime.sh/docs/configuration) may be the right mechanism, but
  no fetched doc page confirms whether it can be pointed at a specific path
  such as `/actuals`, versus only invoking a generic agent entrypoint.

**[VERIFY]** whether a dashboard webhook trigger can target `/actuals`
specifically, and what request shape it forwards. **HUMAN STEP:**

1. Open the Maritime dashboard for `airiwheels-agent`, Triggers section.
2. Look for a "webhook" trigger type distinct from the outbound SDK
   webhooks above.
3. If it supports a target path, point it at `POST /actuals`; if it only
   exposes `https://.../api/webhooks/{agent_id}` as a generic inbound URL,
   confirm with Maritime support/docs whether path suffixes are honored,
   or whether the upstream POS/loyalty system should instead call
   `POST /actuals` directly against the agent's public URL
   (`https://<agent>.maritime.sh/actuals` — confirm exact base URL format
   from `maritime info airiwheels-agent`).
4. Confirm the payload your POS/loyalty system will send matches:

```json
{
  "rows": [
    {"location": "demo_location", "date": "2026-08-02", "item": "burger", "qty_sold": 41.0}
  ]
}
```

(This is the exact shape `app/main.py`'s `ActualsRequest`/`ActualRow`
models require — one row per `(location, date, item)`.)

---

## 6. Verify the live deployment

**HUMAN STEP** (reads real deployment state; safe/non-destructive, but
requires your live agent):

```bash
maritime status airiwheels-agent
maritime info airiwheels-agent
maritime logs airiwheels-agent -f
```

- `status` shows health/activity/deployment state; `info` shows full
  metadata including configured env keys and port; `logs -f` streams live
  logs (stop with the usual terminal interrupt, not a process kill — see
  this repo's `CLAUDE.md` process-safety note, which applies to any
  process you spawn locally to watch this, e.g. a piped log tail).
  (Source: https://maritime.sh/docs/cli)
- API equivalents: `GET /api/agents/{agent_id}`,
  `GET /api/agents/{agent_id}/logs`.
  (Source: https://maritime.sh/docs/api)
- Platform health check (no auth, sanity-checks Maritime itself):
  `GET https://api.maritime.sh/health`
  (Source: https://maritime.sh/docs/api)

Then confirm the actual service responds, using the agent's public URL
(get the exact hostname from `maritime info airiwheels-agent`):

```bash
curl https://<your-agent>.maritime.sh/health
# {"status":"ok"}
curl -X POST https://<your-agent>.maritime.sh/run
curl https://<your-agent>.maritime.sh/forecast/latest
```

If `/forecast/latest` 404s, call `POST /run` first — `app/main.py`
documents this as the expected 404 message before any forecast has been
generated.

---

## 7. Multi-tenant note: one isolated agent per restaurant

PRD.md section 8 requires one isolated agent instance per restaurant, each
keeping its own data and learned weights, with engineer-config changes
flowing to all instances on redeploy. Maritime documents exactly this
pattern under "Fleet":

- `maritime create swarm --template <template> --count <N> --idle 3600 --json`
  spins up `N` independently addressable agents (`swarm-1`...`swarm-N`),
  each manageable individually (`maritime logs swarm-3 -f`,
  `maritime delete swarm-3 -y`, etc.). Docs explicitly list "multi-tenant
  deployments (one agent per customer)" as a use case.
  (Source: https://maritime.sh/docs/fleet)
- Practical mapping for AiriWheels: **one Maritime agent per restaurant
  location**, e.g. `airiwheels-<restaurant-slug>`, each with its own env
  vars (its own POS credentials, its own `LOCATION` value — see
  `.env.example`) and its own persisted forecast log / weights store. A
  config change to the shared codebase (a new `main` commit) is rolled out
  by redeploying each agent (`maritime deploy <agent> ...`, per restaurant)
  — there is no single "redeploy all tenants" command documented; this is
  a per-agent loop today.
  (Source: https://maritime.sh/docs/fleet — hard cap of 50 agents per
  spin-up, billing requires N×$1 wallet balance upfront)
- Fleet has a documented cap of 50 agents per spin-up call. If AiriWheels
  grows past 50 restaurants, **[VERIFY]** whether multiple spin-up calls
  or a different provisioning path is needed — not covered in the docs
  fetched.
- API-first equivalent for provisioning one isolated agent per restaurant
  programmatically: `POST /api/v1/provision`, returning a unique
  `agent_id` per instance.
  (Source: https://maritime.sh/docs/api/provisioning)

---

## 8. Maritime doc sources cited in this runbook

- https://maritime.sh/
- https://maritime.sh/docs
- https://maritime.sh/docs/quickstart
- https://maritime.sh/docs/configuration
- https://maritime.sh/docs/cli
- https://maritime.sh/docs/build
- https://maritime.sh/docs/channels
- https://maritime.sh/docs/sdk/webhooks
- https://maritime.sh/docs/api
- https://maritime.sh/docs/api/provisioning
- https://maritime.sh/docs/fleet

---

## 9. RUN ONLY AFTER feat/service IS MERGED TO MAIN — live deploy checklist

Do not start this section until `feat/service`'s `/run`, `/actuals`, and
`/forecast/latest` endpoints (`app/main.py`, `app/service/pipeline.py`) are
merged into `main` — the container Maritime builds is whatever's on `main`.

**HUMAN STEP — the entire section below.** Run each command yourself, in
order, from your own authenticated Maritime CLI session. Do not automate
or schedule this section to run unattended.

1. Confirm `main` is up to date and green:
   ```bash
   git checkout main && git pull
   pytest
   ```
2. Confirm the container builds and serves `/health`, `/run`,
   `/forecast/latest` locally (mirrors README.md "Build and run the
   container"):
   ```bash
   docker build -t airiwheels-service .
   docker run --rm -p 8080:8080 airiwheels-service
   # in another terminal:
   curl http://localhost:8080/health
   curl -X POST http://localhost:8080/run
   curl http://localhost:8080/forecast/latest
   ```
   Stop the container afterward with `docker stop <container-id>` (the ID
   printed by `docker run`, or from `docker ps`) — not a broad
   `docker kill $(docker ps -q)` or similar, per this repo's process-safety
   rule in `CLAUDE.md`.
3. Link the repo and create the agent (Section 2 above).
4. Set every secret this deployment needs from `.env.example` (Section 3
   above) — real POS/API keys, not placeholders.
5. Configure the daily cron for `POST /run` (Section 4 above) via the
   dashboard; confirm the schedule and timezone.
6. Configure the actuals webhook/trigger for `POST /actuals` (Section 5
   above) via the dashboard; confirm with your POS/loyalty system what URL
   it will call.
7. Verify the live deployment end-to-end (Section 6 above): `status`,
   `info`, `logs`, then `curl` `/health`, `POST /run`, `/forecast/latest`
   against the real public URL.
8. Repeat steps 3–7 once per restaurant for multi-tenant rollout (Section
   7 above), each with its own agent name and its own env vars.
9. Point the owner-app UI at the live `/forecast/latest` endpoint (PRD.md
   section 14, roadmap phase 8) — out of scope for this runbook; tracked
   separately.
