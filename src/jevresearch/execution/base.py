"""Trial execution contract independent of task and controller."""

from typing import Protocol

from ..core.experiment import ExperimentSpec
from ..core.result import ExperimentResult
from ..tasks.base import Task


class Executor(Protocol):
    kind: str

    def execute(self, task: Task, spec: ExperimentSpec, trial_id: int) -> ExperimentResult: ...
