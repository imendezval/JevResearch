"""Replay Optuna ask/tell from the authoritative JevResearch trial ledger.

No Optuna sidecar is used: a crash before saving an offer has no durable sampler
effect, and a saved offer is reused before any new ask. Pinned Optuna and seed
make replay deterministic; mismatches fail visibly instead of resetting search.
"""

from __future__ import annotations

from dataclasses import asdict

from ..core.candidate_generator import make_spec, trial_seed
from ..storage.history import InvariantError


class OptunaProposer:
    def __init__(self, strategy: str, domain, rejection_limit: int):
        if strategy not in ("tpe", "cmaes") or (strategy == "cmaes" and domain.mixed):
            raise ValueError("CMA-ES requires the numeric domain; unknown Optuna strategy")
        self.strategy, self.domain = strategy, domain
        self.rejection_limit = rejection_limit
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("install the classical extra: pip install '.[classical]'") from exc
        if optuna.__version__ != "5.0.0":
            raise RuntimeError("Optuna 5.0.0 is required for deterministic replay")
        self.optuna = optuna

    def details(self):
        return {"sampler": self.strategy, "sampler_seed": "session_seed",
                "sampler_options": {"n_startup_trials": 10} if self.strategy == "tpe"
                else {"n_startup_trials": 1, "popsize": 4, "representation": "log-distribution"},
                "sampler_version": self.optuna.__version__, "rejection_limit": self.rejection_limit}

    def _study(self, seed):
        optuna = self.optuna
        if self.strategy == "tpe":
            sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
        else:
            sampler = optuna.samplers.CmaEsSampler(seed=seed, n_startup_trials=1, popsize=4)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        return optuna.create_study(direction="maximize", sampler=sampler)

    def _draw(self, study, task, seen, source_digest, seed, index):
        rejected = 0
        for _ in range(self.rejection_limit):
            trial = study.ask()
            params = self.domain.suggest(trial)
            try:
                config = self.domain.config(task, params)
                spec = make_spec(task, config, trial_seed(seed, index), source_digest, None)
            except ValueError:
                study.tell(trial, state=self.optuna.trial.TrialState.FAIL)
                rejected += 1
                continue
            if spec.config_key in seen:
                study.tell(trial, state=self.optuna.trial.TrialState.FAIL)
                rejected += 1
                continue
            return trial, params, config, spec, rejected
        return None

    def propose(self, task, history, seed, index, source_digest):
        if not history or history[0]["status"] != "completed":
            raise InvariantError("classical sampler requires a completed baseline")
        study = self._study(seed)
        baseline = history[0]
        params = self.domain.parameters(baseline["spec"]["config"])
        if self.domain.config(task, params) != baseline["spec"]["config"]:
            raise InvariantError("baseline is outside proposal domain")
        study.add_trial(self.optuna.trial.create_trial(
            params=params, distributions=self.domain.distributions(params),
            value=baseline["result"]["objective"]))
        seen = {baseline["config_key"]}
        for row in history[1:]:
            drawn = self._draw(study, task, seen, source_digest, seed, row["number"])
            if drawn is None:
                raise InvariantError("saved proposal cannot be replayed")
            trial, _, _, spec, rejected = drawn
            metadata = row["proposal"] or {}
            if (spec.config_key != row["config_key"] or trial.number != metadata.get("optuna_trial_number")
                    or rejected != metadata.get("rejections")):
                raise InvariantError("Optuna replay differs from saved trial; refuse to resume")
            if row["status"] not in ("completed", "failed", "interrupted"):
                raise InvariantError("unresolved sampler trial before next proposal")
            result = row["result"]
            if row["status"] == "completed" and result and result["objective"] is not None:
                study.tell(trial, result["objective"])
            else:
                study.tell(trial, state=self.optuna.trial.TrialState.FAIL)
            seen.add(row["config_key"])
        completed = sum(t.state == self.optuna.trial.TrialState.COMPLETE for t in study.trials)
        drawn = self._draw(study, task, seen, source_digest, seed, index)
        if drawn is None:
            return None
        trial, params, config, spec, rejected = drawn
        phase = ("startup" if completed < 10 else "adaptive") if self.strategy == "tpe" else (
            "population" if completed < 5 else "adaptive")
        return config, spec, {"domain_params": params, "optuna_trial_number": trial.number,
                              "rejections": rejected, "completed_observations": completed,
                              "phase": phase}


class TpePoolProposer(OptunaProposer):
    """Versioned batch policy; declined suggestions never become observations."""

    policy = "tpe-pool-v1"

    def __init__(self, domain, rejection_limit: int, width: int):
        if not domain.mixed or not 2 <= width <= 8:
            raise ValueError("TPE pool requires the mixed domain and width in [2,8]")
        super().__init__("tpe", domain, rejection_limit)
        self.width = width

    def details(self):
        return {"sampler": "tpe", "sampler_seed": "session_seed",
                "sampler_options": {"n_startup_trials": 10, "constant_liar": True},
                "sampler_version": self.optuna.__version__, "pool_policy": self.policy,
                "pool_width": self.width, "rejection_limit_per_slot": self.rejection_limit,
                "rejection_policy": "invalid_or_duplicate=FAIL; full_pool_or_stop",
                "declined_policy": "declined_untrained=FAIL; slot_order_then_selected"}

    def _study(self, seed):
        sampler = self.optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=10, constant_liar=True)
        self.optuna.logging.set_verbosity(self.optuna.logging.ERROR)
        return self.optuna.create_study(direction="maximize", sampler=sampler)

    def _pool(self, study, task, seen, source_digest, seed, index, build_candidate):
        completed = sum(t.state == self.optuna.trial.TrialState.COMPLETE for t in study.trials)
        phase = "startup" if completed < 10 else "adaptive"
        drawn = []
        for slot in range(self.width):
            rejections = []
            for _ in range(self.rejection_limit):
                trial = study.ask()
                params = self.domain.suggest(trial)
                try:
                    config = self.domain.config(task, params)
                    spec = make_spec(task, config, trial_seed(seed, index), source_digest, None)
                except ValueError:
                    rejections.append({"trial_number": trial.number, "domain_params": params,
                                       "reason": "invalid"})
                    study.tell(trial, state=self.optuna.trial.TrialState.FAIL)
                    continue
                if spec.config_key in seen:
                    rejections.append({"trial_number": trial.number, "domain_params": params,
                                       "config_key": spec.config_key, "reason": "duplicate"})
                    study.tell(trial, state=self.optuna.trial.TrialState.FAIL)
                    continue
                seen.add(spec.config_key)
                metadata = {"domain_params": params, "optuna_trial_number": trial.number,
                            "pool_slot": slot, "rejections_before_slot": rejections,
                            "completed_observations": completed, "phase": phase,
                            "pool_policy": self.policy}
                drawn.append((trial, build_candidate(config, spec, metadata)))
                break
            else:
                return None
        return drawn

    def propose_pool(self, task, history, seed, index, source_digest, build_candidate):
        if not history or history[0]["status"] != "completed":
            raise InvariantError("TPE pool requires a completed baseline")
        study = self._study(seed)
        baseline = history[0]
        params = self.domain.parameters(baseline["spec"]["config"])
        if self.domain.config(task, params) != baseline["spec"]["config"]:
            raise InvariantError("baseline is outside proposal domain")
        study.add_trial(self.optuna.trial.create_trial(
            params=params, distributions=self.domain.distributions(params),
            value=baseline["result"]["objective"]))
        seen = {baseline["config_key"]}
        for row in history[1:]:
            saved = row.get("offer")
            if not saved or row["status"] not in ("completed", "failed", "interrupted"):
                raise InvariantError("unresolved or missing saved TPE pool")
            drawn = self._pool(study, task, seen, source_digest, seed, row["number"], build_candidate)
            if drawn is None or [asdict(candidate) for _, candidate in drawn] != saved["candidates"]:
                raise InvariantError("TPE pool replay differs from saved offer")
            selected = [(trial, candidate) for trial, candidate in drawn
                        if candidate.id == row["candidate_id"]]
            if (len(selected) != 1 or saved["selected_id"] != row["candidate_id"]
                    or row["spec"] != asdict(selected[0][1].spec)
                    or row["config_key"] != selected[0][1].spec.config_key):
                raise InvariantError("TPE pool replay differs from saved offer or trial")
            for trial, candidate in drawn:
                if candidate.id != row["candidate_id"]:
                    study.tell(trial, state=self.optuna.trial.TrialState.FAIL)
            result = row["result"]
            if row["status"] == "completed" and result and result["objective"] is not None:
                study.tell(selected[0][0], result["objective"])
            else:
                study.tell(selected[0][0], state=self.optuna.trial.TrialState.FAIL)
        drawn = self._pool(study, task, seen, source_digest, seed, index, build_candidate)
        return tuple(candidate for _, candidate in drawn) if drawn is not None else ()
