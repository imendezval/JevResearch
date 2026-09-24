"""Execution interface; vision adds a supervised process implementation."""

from ..core import ExperimentResult, ExperimentSpec
from ..tasks.base import InlineTask


class InlineExecutor:
    kind = "inline"

    def execute(self, task: InlineTask, spec: ExperimentSpec, trial_id: int) -> ExperimentResult:
        return task.evaluate(spec)
