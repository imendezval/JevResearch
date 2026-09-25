"""Strict, versioned configuration for a local paired study."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .experiment import digest

_ID = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,31}$")
_FIELDS = {"version", "task", "protocol", "data_dir", "output_root", "split_seed", "epochs",
           "batch_size", "device", "trial_timeout_seconds", "active_time_budget_seconds",
           "trial_budget", "candidate_limit", "checkpoint_policy", "compute_hourly_usd",
           "arms", "seeds"}
_REQUIRED = {"version", "task", "protocol", "data_dir", "output_root", "arms", "seeds",
             "trial_budget", "active_time_budget_seconds"}
_DEFAULTS = {"split_seed": 1729, "epochs": 1, "batch_size": 256, "device": "cpu",
             "trial_timeout_seconds": 900.0, "candidate_limit": 8,
             "checkpoint_policy": "none", "compute_hourly_usd": None}


@dataclass(frozen=True)
class StudyArm:
    id: str
    controller: str
    model: str | None = None
    api_timeout: float | None = None
    sdk_retries: int | None = None
    max_api_calls: int | None = None
    proposal_strategy: str = "local-move"
    proposal_domain: str | None = None


@dataclass(frozen=True)
class StudySpec:
    version: int
    task: str
    protocol: str
    data_dir: str
    output_root: str
    split_seed: int
    epochs: int
    batch_size: int
    device: str
    trial_timeout_seconds: float
    active_time_budget_seconds: float
    trial_budget: int
    candidate_limit: int
    checkpoint_policy: str
    compute_hourly_usd: float | None
    arms: tuple[StudyArm, ...]
    seeds: tuple[int, ...]

    @classmethod
    def load(cls, path: str | Path, *, data_dir=None, output_root=None):
        raw = json.loads(Path(path).read_text())
        if type(raw) is not dict or set(raw) - _FIELDS or _REQUIRED - set(raw):
            raise ValueError("unknown or missing study specification fields")
        values = {**_DEFAULTS, **raw}
        overrides = {}
        for key, override in (("data_dir", data_dir), ("output_root", output_root)):
            if override is not None:
                overrides[key] = str(Path(override).resolve())
                values[key] = override
            if type(values[key]) is not str or not values[key]:
                raise ValueError(f"invalid {key}")
            values[key] = str(Path(values[key]).resolve())
        if type(values["version"]) is not int or values["version"] != 1:
            raise ValueError("study version must be 1")
        if values["task"] not in ("cifar10", "cifar10_fixture"):
            raise ValueError("unsupported study task")
        expected_protocol = "fixture-v1" if values["task"] == "cifar10_fixture" else "official-train-v1"
        if values["protocol"] != expected_protocol:
            raise ValueError("study protocol does not match task")
        if values["device"] not in ("cpu", "cuda") or values["checkpoint_policy"] not in ("none", "best"):
            raise ValueError("invalid device or checkpoint policy")
        for key, maximum in (("split_seed", None), ("epochs", None), ("batch_size", None),
                             ("trial_budget", None), ("candidate_limit", 8)):
            value = values[key]
            minimum = 0 if key == "split_seed" else 1
            if type(value) is not int or value < minimum or (maximum and value > maximum):
                raise ValueError(f"invalid {key}")
        for key in ("trial_timeout_seconds", "active_time_budget_seconds"):
            value = values[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {key}")
            values[key] = float(value)
        rate = values["compute_hourly_usd"]
        if rate is not None and (type(rate) not in (int, float) or not math.isfinite(rate) or rate < 0):
            raise ValueError("invalid compute_hourly_usd")
        if rate is not None:
            values["compute_hourly_usd"] = float(rate)
        arms = values["arms"]
        if type(arms) is not list or not arms:
            raise ValueError("arms must be a nonempty list")
        parsed = []
        for arm in arms:
            if type(arm) is not dict or set(arm) - set(StudyArm.__dataclass_fields__) or not {"id", "controller"} <= set(arm):
                raise ValueError("invalid arm fields")
            item = StudyArm(**arm)
            if (type(item.id) is not str or not _ID.fullmatch(item.id)
                    or type(item.controller) is not str
                    or item.controller not in ("random", "jev", "single")):
                raise ValueError("invalid arm id or controller")
            if item.proposal_strategy not in ("local-move", "global-random", "global-pool",
                                              "tpe", "tpe-pool", "cmaes"):
                raise ValueError("invalid proposal strategy")
            if item.proposal_strategy == "local-move":
                if item.proposal_domain is not None:
                    raise ValueError("local moves cannot have a global domain")
            elif item.proposal_domain not in ("cifar-mixed-v1", "cifar-sgd-numeric-v1"):
                raise ValueError("global proposal domain is required")
            if item.proposal_strategy == "cmaes" and item.proposal_domain != "cifar-sgd-numeric-v1":
                raise ValueError("CMA-ES requires numeric domain")
            if item.proposal_strategy == "tpe-pool" and (
                    item.proposal_domain != "cifar-mixed-v1" or values["candidate_limit"] < 2):
                raise ValueError("TPE pool requires mixed domain and candidate_limit in [2,8]")
            if item.proposal_strategy in ("global-random", "tpe", "cmaes"):
                if item.controller != "single":
                    raise ValueError("single-proposal strategies require the single selector")
            elif item.controller == "single":
                raise ValueError("local moves and global pools require random or Jev selection")
            if item.controller != "jev" and set(arm) - {"id", "controller", "proposal_strategy", "proposal_domain"}:
                raise ValueError("non-Jev arm cannot have Jev settings")
            if item.controller == "jev":
                item = replace(item, model="jev-1.13.0" if item.model is None else item.model,
                               api_timeout=10.0 if item.api_timeout is None else item.api_timeout,
                               sdk_retries=1 if item.sdk_retries is None else item.sdk_retries,
                               max_api_calls=1 if item.max_api_calls is None else item.max_api_calls)
                if (type(item.model) is not str
                        or not re.fullmatch(r"jev-\d+\.\d+\.\d+", item.model)
                        or type(item.api_timeout) not in (int, float) or not math.isfinite(item.api_timeout)
                        or item.api_timeout <= 0 or type(item.sdk_retries) is not int
                        or item.sdk_retries not in (0, 1)
                        or type(item.max_api_calls) is not int or item.max_api_calls < 1):
                    raise ValueError("invalid Jev arm settings")
            parsed.append(item)
        if len({a.id for a in parsed}) != len(parsed):
            raise ValueError("duplicate arm ID")
        seeds = values["seeds"]
        if type(seeds) is not list or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("seeds must be nonnegative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("duplicate search seed")
        values["arms"], values["seeds"] = tuple(parsed), tuple(seeds)
        return cls(**values), overrides

    def normalized(self):
        return asdict(self)

    def fingerprint(self, source_digest: str) -> str:
        return digest({"spec": self.normalized(), "source_digest": source_digest})
