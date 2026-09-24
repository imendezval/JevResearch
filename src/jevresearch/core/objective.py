"""Generic objective validation and comparison."""

import math

from .result import ExperimentResult


def valid_objective(result: ExperimentResult) -> bool:
    return (result.status == "completed" and isinstance(result.objective, (float, int))
            and not isinstance(result.objective, bool) and math.isfinite(result.objective))


def better(value: float, incumbent: float | None, direction: str) -> bool:
    return incumbent is None or (value > incumbent if direction == "max" else value < incumbent)
