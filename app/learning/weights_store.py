"""Append-only, versioned storage for recalibratable FactorModel weights (PRD.md section 6.4).

Every recalibration cycle (`app/learning/recalibrate.py`) proposes a new
`{"holiday_weight", "event_weight", "rain_weight"}` dict; `WeightsStore`
never mutates or deletes a prior version -- `put`/`rollback_to` always
APPEND a new version. This is what makes recalibration reversible (PRD.md
section 6.4's "gate weight updates and log them"): any past version stays
retrievable via `get(version)` forever, so a caller can reconstruct the
exact `FactorModel(weights=...)` that produced any historical forecast
(see `app/models/factor_model.py::FactorModel.__init__`'s `weights` param).

Persisted as append-only JSON Lines (one JSON object per line) -- simple,
human-auditable, no new dependency beyond the stdlib `json`/`pathlib`.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.models.factor_model import DEFAULT_WEIGHTS


@dataclass(frozen=True)
class WeightsRecord:
    """One immutable, versioned snapshot of the recalibratable weights.

    Attributes:
        version: 1-based, monotonically increasing version number.
        weights: `{"holiday_weight", "event_weight", "rain_weight"}` (a
            copy of the dict that was active as of this version).
        parent_version: the version this one was derived from; `None`
            only for the seed/version-1 record.
        reason: free-text audit note, e.g. `"seed"`, `"gentle
            recalibration"`, `"drift-triggered recalibration"`,
            `"rollback to v2"`.
    """

    version: int
    weights: dict[str, float]
    parent_version: int | None
    reason: str


class WeightsStore:
    """Append-only version history of recalibratable FactorModel weights.

    Never mutates or deletes a prior record -- `put`/`rollback_to` always
    APPEND a new version. This is what makes recalibration reversible
    (PRD.md section 6.4's "gate weight updates and log them"): any past
    version stays retrievable via `get(version)` forever, so a caller can
    reconstruct the exact `FactorModel(weights=...)` that produced any
    historical forecast.
    """

    def __init__(self, path: str | Path, initial_weights: dict[str, float] | None = None) -> None:
        """Load history from `path`, seeding version 1 if the file doesn't exist yet.

        Args:
            path: JSON Lines file backing this store.
            initial_weights: seed weights for version 1, used only when
                `path` does not already exist. Defaults to
                `app.models.factor_model.DEFAULT_WEIGHTS` when `None`.
                Ignored (history is loaded instead) if `path` already has
                records.
        """
        self._path = Path(path)
        self._records: list[WeightsRecord] = []

        if self._path.exists():
            self._records = self._read_all()
        else:
            seed_weights = dict(initial_weights) if initial_weights is not None else dict(DEFAULT_WEIGHTS)
            seed = WeightsRecord(version=1, weights=seed_weights, parent_version=None, reason="seed")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._append_record(seed)
            self._records = [seed]

    def _read_all(self) -> list[WeightsRecord]:
        records: list[WeightsRecord] = []
        text = self._path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(
                WeightsRecord(
                    version=payload["version"],
                    weights=dict(payload["weights"]),
                    parent_version=payload["parent_version"],
                    reason=payload["reason"],
                )
            )
        return records

    def _append_record(self, record: WeightsRecord) -> None:
        line = json.dumps(asdict(record))
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def _defensive_copy(record: WeightsRecord) -> WeightsRecord:
        """Return `record` with a freshly copied `.weights` dict.

        `WeightsRecord` is a frozen dataclass, but freezing only stops
        rebinding its fields -- it does not stop a caller from mutating
        the *dict object* held in `.weights` in place (e.g.
        `store.get(1).weights["holiday_weight"] = 999`). Every read-path
        method (`get`, `latest`, `history`) must hand out a copy, never a
        reference into `self._records`, or such a mutation would silently
        corrupt that version's permanent history -- directly breaking the
        "any past version stays retrievable forever" guarantee this store
        exists to provide.
        """
        return WeightsRecord(
            version=record.version,
            weights=dict(record.weights),
            parent_version=record.parent_version,
            reason=record.reason,
        )

    def latest(self) -> WeightsRecord:
        """Return the highest-version record (a defensive copy)."""
        return self._defensive_copy(self._records[-1])

    def get(self, version: int) -> WeightsRecord:
        """Return the record for exactly `version` (a defensive copy).

        Raises:
            KeyError: no record with that version exists.
        """
        for record in self._records:
            if record.version == version:
                return self._defensive_copy(record)
        raise KeyError(f"WeightsStore.get: no record for version={version!r}")

    def put(self, weights: dict[str, float], reason: str) -> WeightsRecord:
        """Append a new record: `version = latest().version + 1`, `parent_version = latest().version`.

        Returns:
            The newly appended `WeightsRecord`.
        """
        parent = self.latest()
        record = WeightsRecord(
            version=parent.version + 1,
            weights=dict(weights),
            parent_version=parent.version,
            reason=reason,
        )
        self._append_record(record)
        self._records.append(record)
        return record

    def rollback_to(self, version: int) -> WeightsRecord:
        """Append a NEW record whose weights are a copy of `get(version).weights`.

        Advances the version counter but reproduces old weights exactly
        (`reason=f"rollback to v{version}"`). Note `get(version)` alone is
        already sufficient to reconstruct/reuse an old version directly
        without calling this -- this method is for when a caller wants the
        rollback itself recorded as a new, auditable event in history.
        """
        target = self.get(version)
        return self.put(dict(target.weights), reason=f"rollback to v{version}")

    def history(self) -> list[WeightsRecord]:
        """Return all records, version 1..latest, in order (each a defensive copy)."""
        return [self._defensive_copy(record) for record in self._records]
