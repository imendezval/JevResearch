"""A complete proposed experiment offered to a controller."""

from dataclasses import dataclass
from typing import Any

from .experiment import ExperimentSpec


@dataclass(frozen=True)
class Candidate:
    id: str
    operator: str
    parameters: dict[str, Any]
    parent_id: int | None
    config: dict[str, Any]
    spec: ExperimentSpec
