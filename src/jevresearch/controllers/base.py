"""Selection-only controller contract."""

from typing import Protocol

from ..core.candidate import Candidate
from ..core.state import SearchState


class Controller(Protocol):
    kind: str

    def select(self, state: SearchState, candidates: tuple[Candidate, ...],
               rng_state: str) -> tuple[str, str]: ...
