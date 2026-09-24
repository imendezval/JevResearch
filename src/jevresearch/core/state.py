"""Serializable campaign state presented to candidate generation and control."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchState:
    session_id: int
    attempted: int
    budget: int
    best_trial_id: int | None
    best_objective: float | None
    incumbent_config: dict[str, Any] | None
    history: tuple[dict[str, Any], ...]
