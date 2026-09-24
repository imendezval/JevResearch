"""Public contracts for task-independent search."""

from .candidate import Candidate
from .experiment import ExperimentSpec, FrozenDict, canonical, digest
from .objective import better, valid_objective
from .result import ExperimentResult
from .state import SearchState


def __getattr__(name):
    if name == "Controller":
        from ..controllers.base import Controller
        return Controller
    if name == "Task":
        from ..tasks.base import Task
        return Task
    raise AttributeError(name)

__all__ = ["Candidate", "Controller", "ExperimentResult", "ExperimentSpec",
           "FrozenDict", "SearchState", "Task", "better", "canonical", "digest",
           "valid_objective"]
