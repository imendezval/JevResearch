"""Serializable Phase 1 contracts and identity rules."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _freeze(value):
    if isinstance(value, dict):
        return FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class FrozenDict(dict):
    """JSON-compatible mapping whose contents cannot be changed after creation."""

    def __init__(self, values=(), **kwargs):
        raw = dict(values, **kwargs)
        dict.__init__(self, ((key, _freeze(value)) for key, value in raw.items()))

    def _immutable(self, *args, **kwargs):
        raise TypeError("ExperimentSpec config is immutable")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _immutable


@dataclass(frozen=True)
class ExperimentSpec:
    task: str
    protocol: str
    config: dict[str, Any]
    data_split: str
    eval_budget: int
    seed: int
    source_digest: str
    parent_id: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "config", FrozenDict(self.config))

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))

    @property
    def config_key(self) -> str:
        return digest({"task": self.task, "protocol": self.protocol,
                       "config": self.config, "data_split": self.data_split,
                       "eval_budget": self.eval_budget})


@dataclass(frozen=True)
class Candidate:
    id: str
    operator: str
    parameters: dict[str, Any]
    parent_id: int | None
    config: dict[str, Any]
    spec: ExperimentSpec


@dataclass(frozen=True)
class SearchState:
    session_id: int
    attempted: int
    budget: int
    best_trial_id: int | None
    best_objective: float | None
    incumbent_config: dict[str, Any] | None
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ExperimentResult:
    status: Literal["completed", "failed", "interrupted"]
    objective: float | None
    metrics: dict[str, Any]
    duration: float
    error_type: str | None = None
    error_message: str | None = None
    artifacts: tuple[str, ...] = ()


class Task(Protocol):
    name: str
    protocol: str
    data_split: str
    eval_budget: int
    objective_direction: Literal["max", "min"]

    def baseline(self) -> dict[str, Any]: ...
    def validate(self, config: dict[str, Any]) -> None: ...
    def proposals(self, state: SearchState) -> list[tuple[str, dict[str, Any], dict[str, Any]]]: ...
    def evaluate(self, spec: ExperimentSpec) -> ExperimentResult: ...


class Controller(Protocol):
    kind: str

    def select(self, state: SearchState, candidates: tuple[Candidate, ...],
               rng_state: str) -> tuple[str, str]: ...


def valid_objective(result: ExperimentResult) -> bool:
    return (result.status == "completed" and isinstance(result.objective, (float, int))
            and not isinstance(result.objective, bool) and math.isfinite(result.objective))


def better(value: float, incumbent: float | None, direction: str) -> bool:
    return incumbent is None or (value > incumbent if direction == "max" else value < incumbent)
