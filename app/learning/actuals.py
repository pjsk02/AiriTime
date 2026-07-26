"""Realized-sales ingestion for the self-learning loop (PRD.md section 6.4 "Actuals ingestion").

Unlike `app/learning/forecast_log.py::ForecastLog`, there is exactly one
true actual per `(location, date, item)`, so later ingests for the same
key legitimately overwrite (e.g. a corrected point-of-sale
reconciliation) -- matching `app/features/store.py::FeatureStore`'s own
upsert semantics.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

_KEY_COLUMNS = ["location", "date", "item"]
_COLUMNS = _KEY_COLUMNS + ["qty_sold"]


class ActualsStore:
    """Realized sales, upserted keyed (location, date, item).

    Unlike `ForecastLog`, there is exactly one true actual per (location,
    date, item), so later ingests for the same key legitimately overwrite
    (e.g. a corrected point-of-sale reconciliation), matching
    `FeatureStore`'s own upsert semantics (PRD.md section 6.4's actuals
    ingestion).
    """

    def __init__(self, path: str | Path) -> None:
        """Bind to `path` (created lazily on first `ingest`)."""
        self._path = Path(path)

    def ingest(self, rows: pd.DataFrame) -> None:
        """Upsert `rows` keyed by (location, date, item).

        Args:
            rows: `location, date, item, qty_sold` (the realized actual).
                Rewrites the whole backing file on each ingest -- acceptable
                at this scale, no need for a fancier storage engine.
        """
        if rows.empty:
            return

        rows = rows.copy()
        rows["date"] = pd.to_datetime(rows["date"]).dt.date
        rows = rows[_COLUMNS]

        existing = self._read_raw()
        if existing.empty:
            combined = rows
        else:
            # Existing rows first, new rows last: `drop_duplicates(keep="last")`
            # then keeps the NEW row whenever a key collides, while rows
            # from `existing` whose key isn't in `rows` are preserved
            # untouched -- a clean, unambiguous upsert.
            combined = pd.concat([existing, rows], ignore_index=True)
            combined = combined.drop_duplicates(subset=_KEY_COLUMNS, keep="last")
        combined = combined[_COLUMNS].reset_index(drop=True)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_raw(combined)

    def read(self, location: str, date_range: tuple[date, date]) -> pd.DataFrame:
        """Return rows for `location` within the inclusive `date_range`.

        Mirrors `FeatureStore.query`'s shape/semantics: sorted by (date,
        item); empty (zero rows, correct columns) if nothing matches or
        the store has never been ingested into.
        """
        df = self._read_raw()
        if df.empty:
            return df

        start, end = date_range
        mask = (df["location"] == location) & (df["date"] >= start) & (df["date"] <= end)
        return df.loc[mask].sort_values(["date", "item"]).reset_index(drop=True)

    def _read_raw(self) -> pd.DataFrame:
        if not self._path.exists():
            return pd.DataFrame(columns=_COLUMNS)
        text = self._path.read_text(encoding="utf-8")
        records = []
        for line in text.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["date"] = date.fromisoformat(payload["date"])
            records.append(payload)
        if not records:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.DataFrame.from_records(records, columns=_COLUMNS)

    def _write_raw(self, df: pd.DataFrame) -> None:
        lines = []
        for row in df.itertuples(index=False):
            lines.append(
                json.dumps(
                    {
                        "location": row.location,
                        "date": row.date.isoformat(),
                        "item": row.item,
                        "qty_sold": float(row.qty_sold),
                    }
                )
            )
        self._path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def join_forecast_actuals(forecast_rows: pd.DataFrame, actuals_rows: pd.DataFrame) -> pd.DataFrame:
    """Inner-join logged forecasts with realized actuals on (location, date, item).

    Args:
        forecast_rows: as from `ForecastLog.read()` -- needs at least
            `location, date, item, p10, p50, p90, attribution,
            generated_for_date, weights_version`.
        actuals_rows: `location, date, item, qty_sold`.

    Returns:
        The joined DataFrame with `qty_sold` renamed to `actual_qty` (to
        avoid confusion with any `qty_sold` that might already be present
        on `forecast_rows`), plus `error = actual_qty - p50` and
        `abs_error = abs(error)`. Rows with no actual yet (forecast for a
        date that hasn't happened) or no forecast (actual with nothing
        logged) are dropped (inner join) -- the correct semantics for a
        "graded forecasts" frame.
    """
    renamed_actuals = actuals_rows.rename(columns={"qty_sold": "actual_qty"})
    joined = forecast_rows.merge(renamed_actuals, on=["location", "date", "item"], how="inner")
    joined["error"] = joined["actual_qty"] - joined["p50"]
    joined["abs_error"] = joined["error"].abs()
    return joined
