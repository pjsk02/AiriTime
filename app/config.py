"""Engineer-plane config loader.

Loads `config.yaml` (repo root) into a validated `AgentConfig` pydantic
model. Field names here must match `config.yaml` keys exactly.

This is Phase 1 scaffolding only: fields are placeholders for the rolling
horizon (+7..+13), the model registry selection, factor toggles, and the
newsvendor cost ratio (q* = Cu / (Cu + Co), see PRD.md section 5). Nothing
here is wired into forecasting yet — that starts in later roadmap phases.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class FactorToggles(BaseModel):
    """Which demand factors are switched on for the factor model.

    All placeholders for Phase 1 — no factor logic exists yet.
    """

    day_of_week: bool = True
    holidays: bool = True
    events: bool = True
    weather: bool = False
    loyalty: bool = False


class AgentConfig(BaseModel):
    """Engineer-plane config (PRD section 6.6 / section 5).

    - horizon_start / horizon_end: rolling horizon window, days from
      "today" (PRD section 5: we serve +7..+13).
    - model_name: which registered model to use (PRD section 6.1); a
      placeholder string until the model registry exists (Phase 4).
    - factors: toggles for which named factors feed the factor model.
    - cost_ratio: Cu (cost of under-prep) over Co (cost of over-prep),
      used to derive the newsvendor critical fractile
      q* = Cu / (Cu + Co) (PRD section 5). Stored as the two raw costs
      so q* can be computed rather than hardcoded.
    """

    horizon_start: int = Field(default=7, description="Rolling horizon start, days from today")
    horizon_end: int = Field(default=13, description="Rolling horizon end, days from today")
    model_name: str = "factor_model_v1"
    factors: FactorToggles = Field(default_factory=FactorToggles)
    cost_underprep: float = Field(default=2.0, description="Cu: cost of a stockout / lost margin")
    cost_overprep: float = Field(default=1.0, description="Co: cost of wasted/over-prepped food")

    @property
    def critical_fractile(self) -> float:
        """q* = Cu / (Cu + Co), the newsvendor-optimal order quantile."""
        return self.cost_underprep / (self.cost_underprep + self.cost_overprep)


def load_config(path: str = "config.yaml") -> AgentConfig:
    """Read `path` as YAML and construct an `AgentConfig`.

    Kept deliberately simple: callers decide how to handle a missing or
    invalid file. `/health` does not depend on this succeeding.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    return AgentConfig(**data)
