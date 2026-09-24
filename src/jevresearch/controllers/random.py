"""Seeded random selection with serializable RNG state."""

import json
import random

from ..core import Candidate, SearchState, canonical


def initial_rng(seed: int) -> str:
    return canonical(random.Random(seed).getstate())


def _tuple(value):
    return tuple(_tuple(v) for v in value) if isinstance(value, list) else value


class RandomController:
    kind = "random"

    def select(self, state: SearchState, candidates: tuple[Candidate, ...], rng_state: str) -> tuple[str, str]:
        if not candidates:
            raise ValueError("empty candidate offer")
        rng = random.Random()
        rng.setstate(_tuple(json.loads(rng_state)))
        choice = rng.choice(candidates)
        return choice.id, canonical(rng.getstate())
