"""Instantiate and deduplicate task-defined operators."""

from dataclasses import asdict

from .core import Candidate, ExperimentSpec, SearchState, Task, digest


def trial_seed(seed: int, number: int) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(f"{seed}:{number}".encode()).digest()[:4], "big")


def make_spec(task: Task, config: dict, seed: int, source_digest: str,
              parent: int | None) -> ExperimentSpec:
    task.validate(config)
    return ExperimentSpec(task.name, task.protocol, dict(config), task.data_split,
                          task.eval_budget, seed, source_digest, parent)


class CandidateGenerator:
    def generate(self, task: Task, state: SearchState, seed: int,
                 source_digest: str) -> tuple[Candidate, ...]:
        seen = {entry["config_key"] for entry in state.history}
        candidates = []
        for operator, parameters, config in task.proposals(state):
            try:
                spec = make_spec(task, config, trial_seed(seed, state.attempted),
                                 source_digest, state.best_trial_id)
            except ValueError:
                continue
            if spec.config_key in seen:
                continue
            seen.add(spec.config_key)
            cid = digest({"operator": operator, "parameters": parameters,
                          "spec": asdict(spec)})[:16]
            candidates.append(Candidate(cid, operator, parameters, state.best_trial_id,
                                        dict(config), spec))
        return tuple(candidates)
