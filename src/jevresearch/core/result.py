"""The persisted outcome of one attempted experiment."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ExperimentResult:
    status: Literal["completed", "failed", "interrupted"]
    objective: float | None
    metrics: dict[str, Any]
    duration: float
    error_type: str | None = None
    error_message: str | None = None
    artifacts: tuple[str, ...] = ()
