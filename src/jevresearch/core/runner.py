"""Generic durable experiment loop."""

from __future__ import annotations

import json
import time
from dataclasses import asdict

from .candidate_generator import CandidateGenerator, make_spec, trial_seed
from ..controllers.random import initial_rng
from ..controllers.base import Controller
from ..execution.local import InlineExecutor
from ..source import identity
from ..storage.history import InvariantError, Store
from ..tasks.base import Task
from . import Candidate, ExperimentResult, ExperimentSpec, SearchState, canonical, valid_objective


def _candidate(raw: dict) -> Candidate:
    return Candidate(raw["id"], raw["operator"], raw["parameters"], raw["parent_id"],
                     raw["config"], ExperimentSpec(**raw["spec"]))


def _result(task: Task, spec: ExperimentSpec, executor, trial_id: int) -> ExperimentResult:
    start = time.monotonic()
    try:
        result = executor.execute(task, spec, trial_id)
    except Exception as exc:
        return ExperimentResult("failed", None, {}, time.monotonic() - start,
                                type(exc).__name__, "task evaluation raised an exception")
    if (not isinstance(result, ExperimentResult)
            or (result.status == "completed" and not valid_objective(result))
            or result.status not in ("completed", "failed", "interrupted")
            or spec.task != task.name or spec.protocol != task.protocol):
        return ExperimentResult("failed", None, {}, time.monotonic() - start,
                                "InvalidResult", "missing, nonfinite, failed, or incompatible objective")
    try:
        canonical(asdict(result))
    except (TypeError, ValueError):
        return ExperimentResult("failed", None, {}, time.monotonic() - start,
                                "InvalidResult", "result contains non-JSON or nonfinite values")
    return result


class Runner:
    def __init__(self, store: Store, task: Task, controller: Controller,
                 executor=None, generator=None):
        self.store, self.task, self.controller = store, task, controller
        self.executor = executor or InlineExecutor()
        self.generator = generator or CandidateGenerator()

    def start(self, budget: int, seed: int) -> int:
        if budget < 1:
            raise ValueError("budget must be >= 1")
        if self.task.objective_direction not in ("min", "max"):
            raise ValueError("objective direction must be min or max")
        if self.task.eval_budget < 1:
            raise ValueError("evaluation budget must be >= 1")
        source = identity(self.task)
        baseline = make_spec(self.task, self.task.baseline(), trial_seed(seed, 0), source["digest"], None)
        settings = {"task": self.task.name, "protocol": self.task.protocol,
                    "data_split": self.task.data_split, "eval_budget": self.task.eval_budget,
                    "direction": self.task.objective_direction, "controller": self.controller.kind,
                    "seed": seed, "budget": budget, "seed_schedule": "sha256(seed:index)",
                    "executor": self.executor.kind,
                    "execution_details": getattr(self.executor, "details", lambda: {})(),
                    "controller_details": getattr(self.controller, "details", lambda: {})(),
                    "candidate_limit": self.generator.max_candidates,
                    "task_details": getattr(self.task, "details", lambda: {})()}
        return self.store.create(settings, source, initial_rng(seed), baseline)

    def _compatible(self, sid: int):
        row = self.store.session(sid)
        settings, source = json.loads(row["settings"]), json.loads(row["source"])
        expected = (self.task.name, self.task.protocol, self.task.data_split,
                    self.task.eval_budget, self.task.objective_direction, self.controller.kind)
        actual = tuple(settings[k] for k in ("task", "protocol", "data_split", "eval_budget", "direction", "controller"))
        if (actual != expected or source["digest"] != identity(self.task)["digest"]
                or settings.get("executor", "inline") != self.executor.kind
                or settings.get("execution_details", {}) != getattr(self.executor, "details", lambda: {})()
                or settings.get("controller_details", {}) != getattr(self.controller, "details", lambda: {})()
                or settings.get("candidate_limit", 8) != self.generator.max_candidates
                or settings.get("task_details", {}) != getattr(self.task, "details", lambda: {})()):
            raise InvariantError("session task/protocol/controller or executable source changed; start a new session")
        return settings, source

    def _audited_select(self, sid: int, offer, candidates: tuple[Candidate, ...],
                        state: SearchState, direction: str, calls: int) -> int:
        previous = self.store.validated_decision(offer["id"])
        if previous is not None:
            match = [c for c in candidates if c.id == previous["selected_id"]]
            if len(match) != 1:
                raise InvariantError("validated decision refers to an absent candidate")
            self.store.select(sid, offer["id"], match[0], offer["rng_before"], previous["id"])
            return calls
        if calls >= self.controller.max_calls:
            self.store.pause(sid, "logical API call limit reached; resume to continue")
            return calls
        prepared = self.store.first_decision_request(offer["id"])
        if prepared is None:
            prepared = self.controller.prepare(state, candidates, direction)
        attempt_id = self.store.begin_decision(sid, offer["id"], prepared,
                                               self.controller.details())
        calls += 1
        start = time.monotonic()
        try:
            raw = self.controller.invoke(prepared)
        except Exception as exc:
            unknown = getattr(exc, "outcome_unknown", True)
            code = getattr(exc, "category", "transport_error")
            self.store.fail_decision(attempt_id, "unknown" if unknown else "failed",
                                     code, time.monotonic() - start)
            self.store.pause(sid, f"controller {code}; resume to retry saved offer")
            return calls
        latency = time.monotonic() - start
        try:
            validated = self.controller.validate(prepared, raw)
        except Exception:
            self.store.fail_decision(attempt_id, "failed", "invalid_response", latency)
            self.store.pause(sid, "controller invalid_response; resume to retry saved offer")
            return calls
        match = [c for c in candidates if c.id == validated.selected_id]
        if len(match) != 1:
            self.store.fail_decision(attempt_id, "failed", "invalid_candidate", latency)
            self.store.pause(sid, "controller invalid_candidate; resume to retry saved offer")
            return calls
        self.store.validate_decision(attempt_id, validated.response, validated.selected_id, latency)
        self.store.select(sid, offer["id"], match[0], offer["rng_before"], attempt_id)
        return calls

    def run(self, sid: int, max_new_trials: int | None = None) -> SearchState:
        settings, source = self._compatible(sid)
        if max_new_trials is not None and max_new_trials < 0:
            raise ValueError("max_new_trials must be nonnegative")
        audited = hasattr(self.controller, "prepare")
        if audited:
            self.store.mark_unknown_started(sid)
        self.store.resume(sid)
        new = 0
        calls = 0
        while True:
            state = self.store.state(sid)
            rows = self.store.trials(sid)
            last = rows[-1]
            if last["status"] == "running":
                self.store.interrupt(last["id"])
                state = self.store.state(sid)
                if last["number"] == 0:
                    self.store.stop(sid, "baseline failed")
                    return state
                continue
            if last["status"] == "pending":
                if max_new_trials is not None and new >= max_new_trials:
                    return state
                self.store.running(last["id"])
                spec = ExperimentSpec(**json.loads(last["spec"]))
                result = _result(self.task, spec, self.executor, last["id"])
                self.store.finish(last["id"], result)
                new += 1
                if last["number"] == 0 and result.status != "completed":
                    self.store.stop(sid, "baseline failed")
                    return self.store.state(sid)
                continue
            if self.store.session(sid)["status"] == "stopped":
                return state
            if state.attempted >= settings["budget"]:
                self.store.stop(sid, "budget exhausted")
                return state
            if max_new_trials is not None and new >= max_new_trials:
                return state
            offer = self.store.outstanding(sid)
            if offer is None:
                candidates = self.generator.generate(self.task, state, settings["seed"], source["digest"])
                if not candidates:
                    self.store.stop(sid, "no novel candidates")
                    return state
                oid = self.store.save_offer(sid, state, candidates, self.store.session(sid)["rng_state"])
                offer = self.store.offer(sid, oid)
            candidates = tuple(_candidate(c) for c in json.loads(offer["candidates"]))
            snapshot = SearchState(**{**json.loads(offer["state"]),
                                      "history": tuple(json.loads(offer["state"])["history"])})
            if audited:
                calls = self._audited_select(sid, offer, candidates, snapshot,
                                             settings["direction"], calls)
                if self.store.session(sid)["status"] == "paused":
                    return self.store.state(sid)
                continue
            selected_id, next_rng = self.controller.select(snapshot, candidates, offer["rng_before"])
            match = [c for c in candidates if c.id == selected_id]
            if len(match) != 1:
                raise InvariantError("controller selected an ID outside the saved offer")
            self.store.select(sid, offer["id"], match[0], next_rng)
