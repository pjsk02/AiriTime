"""Append-only skill scorecard: one row per recalibration cycle (PRD.md section 6.4 "skill scorecard").

Never overwrites a prior entry -- same append-only JSON Lines pattern as
`app/learning/forecast_log.py` and `app/learning/weights_store.py`.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

_COLUMNS = ["date", "wmape", "skill_vs_naive", "weights_version"]


class Scorecard:
    """Append-only skill history: one row per recalibration cycle.

    (PRD.md section 6.4's "skill scorecard"). Never overwrites a prior
    entry.
    """

    def __init__(self, path: str | Path) -> None:
        """Bind to `path` (created lazily on first `append`)."""
        self._path = Path(path)

    def append(self, cycle_date: date, wmape: float, skill_vs_naive: float, weights_version: int) -> None:
        """Append one entry for a recalibration cycle.

        Args:
            cycle_date: the date this recalibration cycle ran.
            wmape: weighted MAPE for this cycle (see
                `app/models/backtest.py::walk_forward_backtest`).
            skill_vs_naive: skill vs the naive same-day-last-week baseline
                for this cycle.
            weights_version: the `WeightsStore` version active as of this
                cycle.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "date": cycle_date.isoformat(),
            "wmape": float(wmape),
            "skill_vs_naive": float(skill_vs_naive),
            "weights_version": weights_version,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read(self) -> pd.DataFrame:
        """Return all entries, columns `date, wmape, skill_vs_naive, weights_version`, sorted by date ascending.

        Returns:
            Empty (zero rows, correct columns) if the file doesn't exist yet.
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
            records.append(payload)

        if not records:
            return pd.DataFrame(columns=_COLUMNS)

        df = pd.DataFrame.from_records(records, columns=_COLUMNS)
        return df.sort_values("date").reset_index(drop=True)
