"""Immutable experiment specifications and stable JSON identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _freeze(value):
    if isinstance(value, dict):
        return FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class FrozenDict(dict):
    """JSON-compatible mapping whose contents cannot be changed after creation."""

    def __init__(self, values=(), **kwargs):
        raw = dict(values, **kwargs)
        dict.__init__(self, ((key, _freeze(value)) for key, value in raw.items()))

    def _immutable(self, *args, **kwargs):
        raise TypeError("ExperimentSpec config is immutable")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _immutable


@dataclass(frozen=True)
class ExperimentSpec:
    task: str
    protocol: str
    config: dict[str, Any]
    data_split: str
    eval_budget: int
    seed: int
    source_digest: str
    parent_id: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "config", FrozenDict(self.config))

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))

    @property
    def config_key(self) -> str:
        return digest({"task": self.task, "protocol": self.protocol,
                       "config": self.config, "data_split": self.data_split,
                       "eval_budget": self.eval_budget})
