"""ForecastModel interface and the model registry (PRD.md sections 6.1, 6.3, 11, 14 phase 4).

The model registry is the third input-layer contract (alongside
`SalesConnector` in `app/connectors/base.py` and `SignalProvider` in
`app/signals/base.py`): forecasters are added as new files implementing
`ForecastModel` and registering themselves in `MODEL_REGISTRY`, without
editing anything downstream (the backtest, a future runner/CLI, the
output writer). `app/models/factor_model.py::FactorModel` is the v1
hackathon forecaster (PRD.md section 11); a GBM quantile model or a
foundation model (PRD.md section 11 "Later") would be interchangeable
drop-ins behind this same interface.
"""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class ForecastModel(ABC):
    """Interface every registered forecaster implements (PRD.md section 6.1, 6.3).

    Kept minimal and stable so a factor model, a GBM quantile model, and a
    foundation model (PRD.md section 11 "Later") are interchangeable and
    comparable on the same backtest (app/models/backtest.py).
    """

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> None:
        """Fit on past FeatureStore-shaped rows.

        Args:
            history: rows with at least `location, date, item, qty_sold`
                plus whatever signal columns are present (`is_holiday`,
                `holiday_name`, `temp_c`, `precip_mm`, `is_rain`,
                `event_count`, `event_impact`). Every date must strictly
                precede any date this model will later be asked to
                `predict()` -- callers (the backtest, the runner) are
                responsible for that; this method does not enforce it.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, future_features: pd.DataFrame, reference_today: date) -> pd.DataFrame:
        """Forecast p10/p50/p90 with attribution for future (date, item) rows.

        Args:
            future_features: rows with `location, date, item` plus the same
                signal columns as `fit`'s `history` (NO `qty_sold` -- that's
                what's being forecast). One row per (location, date, item)
                to forecast.
            reference_today: the forecast's reference "today" (PRD.md
                section 5's rolling horizon is measured from this date);
                used to compute each row's horizon offset
                `(row.date - reference_today).days` for horizon-dependent
                behavior (band widening, weather down-weighting).

        Returns:
            A DataFrame with columns `location, date, item, p10, p50, p90,
            attribution`, one row per input row, `p10 <= p50 <= p90`
            guaranteed by construction. `attribution` holds a Python list
            of dicts (NOT a DataFrame column of scalars -- an object column
            of lists), each dict shaped
            `{"factor": str, "direction": "up"|"down", "text": str, "contribution": float}`,
            ordered by descending `abs(contribution)` (biggest driver
            first) -- this feeds the owner-facing "why" (PRD.md section
            6.3).
        """
        raise NotImplementedError


MODEL_REGISTRY: dict[str, type[ForecastModel]] = {}
