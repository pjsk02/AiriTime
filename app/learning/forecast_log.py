"""Append-only log of every forecast ever produced (PRD.md section 6.4 "Forecast log").

Keyed conceptually by `(location, date, item, generated_for_date)` --
multiple entries legitimately exist for the SAME (location, date, item)
across different `generated_for_date` values, since the rolling +7..+13
horizon re-forecasts the same calendar day daily as it approaches (PRD.md
section 5). This is a log, not an upsert store; nothing is ever
overwritten (contrast with `app/learning/actuals.py::ActualsStore`, which
upserts).

Persisted as append-only JSON Lines, one forecast row per line, with
`p10/p50/p90` kept at FULL FLOAT PRECISION (not rounded). This is a
separate, internal, higher-precision record for the learning loop only --
it is not the Phase-4 `app/output/writer.py` `forecast_latest.json`
contract file, which stays untouched and still does its own int-rounding
independently for the owner-facing document.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

_COLUMNS = [
    "location",
    "date",
    "item",
    "generated_for_date",
    "weights_version",
    "model_name",
    "p10",
    "p50",
    "p90",
    "attribution",
]


class ForecastLog:
    """Append-only log of every forecast ever produced.

    Keyed (location, date, item, generated_for_date) -- multiple entries
    legitimately exist for the SAME (location, date, item) across
    different generated_for_date values, since the rolling +7..+13
    horizon re-forecasts the same calendar day daily as it approaches
    (PRD.md section 5) -- this is a log, not an upsert store; nothing is
    ever overwritten.
    """

    def __init__(self, path: str | Path) -> None:
        """Bind to `path` (created lazily on first `append`)."""
        self._path = Path(path)

    def append(
        self,
        predictions: pd.DataFrame,
        generated_for_date: date,
        weights_version: int,
        model_name: str,
    ) -> None:
        """Append one JSON line per row of `predictions`.

        Args:
            predictions: a `ForecastModel.predict`-shaped DataFrame
                (`location, date, item, p10, p50, p90, attribution`),
                straight from `predict()` at FULL FLOAT PRECISION -- do not
                round before calling this.
            generated_for_date: the forecast run's reference "today"
                (PRD.md section 8's daily cron "today").
            weights_version: the `WeightsStore` version whose weights were
                used to produce `predictions` (see
                `app/learning/weights_store.py`).
            model_name: the registered model name that produced
                `predictions` (e.g. `"factor_model_v1"`).
        """
        if predictions.empty:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for row in predictions.itertuples(index=False):
            record = {
                "location": row.location,
                "date": row.date.isoformat(),
                "item": row.item,
                "generated_for_date": generated_for_date.isoformat(),
                "weights_version": weights_version,
                "model_name": model_name,
                "p10": float(row.p10),
                "p50": float(row.p50),
                "p90": float(row.p90),
                "attribution": list(row.attribution),
            }
            lines.append(json.dumps(record))

        with self._path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def read(
        self,
        location: str | None = None,
        date_range: tuple[date, date] | None = None,
    ) -> pd.DataFrame:
        """Read back logged rows, optionally filtered.

        Args:
            location: if given, only rows for this location.
            date_range: if given, an inclusive `(start, end)` filter on the
                forecasted `date` column (NOT `generated_for_date`).

        Returns:
            A DataFrame with columns `location, date, item,
            generated_for_date, weights_version, model_name, p10, p50,
            p90, attribution` (attribution deserialized back into a Python
            list of dicts, not a JSON string). Empty (zero rows, correct
            columns) if the log file doesn't exist yet or nothing matches.
        """
        if not self._path.exists():
            return pd.DataFrame(columns=_COLUMNS)

        records = []
        text = self._path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["date"] = date.fromisoformat(payload["date"])
            payload["generated_for_date"] = date.fromisoformat(payload["generated_for_date"])
            records.append(payload)

        df = pd.DataFrame.from_records(records, columns=_COLUMNS)
        if df.empty:
            return df

        if location is not None:
            df = df[df["location"] == location]
        if date_range is not None:
            start, end = date_range
            df = df[(df["date"] >= start) & (df["date"] <= end)]

        return df.reset_index(drop=True)
