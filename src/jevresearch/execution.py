"""Execution interface; vision adds a supervised process implementation."""

from .core import ExperimentResult, ExperimentSpec, Task


class InlineExecutor:
    kind = "inline"

    def execute(self, task: Task, spec: ExperimentSpec, trial_id: int) -> ExperimentResult:
        return task.evaluate(spec)
