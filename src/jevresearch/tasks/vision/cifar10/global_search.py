"""Task-specific global proposal generation; selection remains generic."""

from __future__ import annotations

import random
from dataclasses import asdict

from ....core.candidate import Candidate
from ....core.candidate_generator import CandidateGenerator, make_spec, trial_seed
from ....core.experiment import digest
from .domain import CifarDomain


def cifar_generator(strategy: str, domain: str | None = None, candidate_limit: int = 8):
    if strategy == "local-move":
        return CandidateGenerator(candidate_limit)
    return GlobalCandidateGenerator(domain, strategy, candidate_limit)


class GlobalCandidateGenerator:
    rejection_limit = 32

    def __init__(self, domain_name: str, strategy: str, candidate_limit: int = 8):
        self.domain = CifarDomain(domain_name)
        if strategy not in ("global-random", "global-pool", "tpe", "tpe-pool", "cmaes"):
            raise ValueError("unknown global proposal strategy")
        if strategy == "cmaes" and self.domain.mixed:
            raise ValueError("CMA-ES cannot use the categorical mixed domain")
        if not 1 <= candidate_limit <= 8:
            raise ValueError("global candidate limit must be in [1,8]")
        if strategy == "tpe-pool" and (not self.domain.mixed or candidate_limit < 2):
            raise ValueError("TPE pool requires the mixed domain and width in [2,8]")
        self.strategy = strategy
        self.max_candidates = candidate_limit if strategy in ("global-pool", "tpe-pool") else 1
        self.proposer = None
        if strategy in ("tpe", "cmaes"):
            from ....optimizers.optuna import OptunaProposer
            self.proposer = OptunaProposer(strategy, self.domain, self.rejection_limit)
        elif strategy == "tpe-pool":
            from ....optimizers.optuna import TpePoolProposer
            self.proposer = TpePoolProposer(self.domain, self.rejection_limit, candidate_limit)
        self._exhaustion_reason = "global proposal resample limit reached"

    def details(self):
        details = {"proposal_strategy": self.strategy, "proposal_domain": self.domain.name,
                   "domain_definition": self.domain.definition(),
                   "domain_fingerprint": self.domain.fingerprint}
        if self.proposer is not None:
            details.update(self.proposer.details())
        else:
            details.update({"sampler": "python-random", "sampler_version": "python-random-v1",
                            "rejection_limit": self.rejection_limit,
                            "sampler_seed": "sha256(search_seed:attempted_trials)",
                            "sampler_options": {"categorical": "uniform", "positive_float": "log-uniform",
                                                "max_draws": self.rejection_limit}})
        return details

    @property
    def exhaustion_reason(self):
        return self._exhaustion_reason

    def generate(self, task, state, seed, source_digest, history=()):
        if self.strategy == "tpe-pool":
            return self.proposer.propose_pool(task, history, seed, state.attempted,
                                              source_digest, self._candidate)
        if self.proposer is not None:
            proposed = self.proposer.propose(task, history, seed, state.attempted, source_digest)
            if proposed is None:
                return ()
            config, spec, metadata = proposed
            choices = [(config, spec, metadata)]
        else:
            rng = random.Random(trial_seed(seed, state.attempted))
            seen = {row["config_key"] for row in history}
            choices = []
            for draw in range(self.rejection_limit):
                params = self.domain.random_parameters(rng)
                try:
                    config = self.domain.config(task, params)
                    spec = make_spec(task, config, trial_seed(seed, state.attempted),
                                     source_digest, None)
                except ValueError:
                    continue
                if spec.config_key in seen:
                    continue
                seen.add(spec.config_key)
                choices.append((config, spec, {"domain_params": params,
                                               "draw_index": draw, "phase": "global"}))
                if len(choices) == self.max_candidates:
                    break
        return tuple(self._candidate(config, spec, metadata) for config, spec, metadata in choices)

    def _candidate(self, config, spec, metadata):
        parameters = {"proposal_strategy": self.strategy, "domain": self.domain.name,
                      "domain_fingerprint": self.domain.fingerprint, **metadata}
        cid = digest({"operator": "sample-global", "parameters": parameters,
                      "spec": asdict(spec)})[:16]
        return Candidate(cid, "sample-global", parameters, None, config, spec)
