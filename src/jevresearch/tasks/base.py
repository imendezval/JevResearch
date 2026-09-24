"""Contract implemented by each search task."""

from typing import Any, Literal, Protocol
from pathlib import Path

from ..core.experiment import ExperimentSpec
from ..core.result import ExperimentResult
from ..core.state import SearchState


class Task(Protocol):
    name: str
    protocol: str
    data_split: str
    eval_budget: int
    objective_direction: Literal["max", "min"]

    def baseline(self) -> dict[str, Any]: ...
    def validate(self, config: dict[str, Any]) -> None: ...
    def proposals(self, state: SearchState) -> list[tuple[str, dict[str, Any], dict[str, Any]]]: ...


class InlineTask(Task, Protocol):
    def evaluate(self, spec: ExperimentSpec) -> ExperimentResult: ...


class ProcessTask(Task, Protocol):
    def worker_payload(self, spec: ExperimentSpec) -> dict[str, Any]: ...
    def worker_command(self, input_path: Path, result_path: Path) -> list[str]: ...
