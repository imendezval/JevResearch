"""Small deterministic two-dimensional search task."""

from __future__ import annotations

import time
from typing import Any

from .core import ExperimentResult, ExperimentSpec, SearchState


class SyntheticTask:
    name = "synthetic_grid"
    protocol = "v1"
    data_split = "fixed_synthetic_v1"
    eval_budget = 1
    objective_direction = "max"

    def __init__(self, fail_seed: int | None = None):
        self.fail_seed = fail_seed

    def baseline(self) -> dict[str, int]:
        return {"x": 0, "y": 0}

    def validate(self, config: dict[str, Any]) -> None:
        if set(config) != {"x", "y"} or any(type(v) is not int or v < -3 or v > 3 for v in config.values()):
            raise ValueError("synthetic config requires integer x,y in [-3,3]")

    def proposals(self, state: SearchState):
        base = state.incumbent_config or self.baseline()
        out = []
        for axis in ("x", "y"):
            for step in (-1, 1):
                config = dict(base)
                config[axis] += step
                out.append((f"adjust_{axis}", {"step": step}, config))
        return out

    def evaluate(self, spec: ExperimentSpec) -> ExperimentResult:
        start = time.monotonic()
        self.validate(spec.config)
        if spec.seed == self.fail_seed:
            raise RuntimeError("injected synthetic failure")
        x, y = spec.config["x"], spec.config["y"]
        objective = -float((x - 2) ** 2 + (y + 1) ** 2)
        return ExperimentResult("completed", objective, {"distance_squared": -objective},
                                time.monotonic() - start)
