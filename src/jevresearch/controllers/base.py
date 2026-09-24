"""Explicit local and audited controller contracts."""

from typing import Any, Literal, Protocol

from ..core.candidate import Candidate
from ..core.state import SearchState


class LocalController(Protocol):
    kind: str
    selection_mode: Literal["local"]

    def select(self, state: SearchState, candidates: tuple[Candidate, ...],
               rng_state: str) -> tuple[str, str]: ...


class AuditedDecision(Protocol):
    selected_id: str
    response: dict[str, Any]


class AuditedController(Protocol):
    kind: str
    selection_mode: Literal["audited"]
    max_calls: int

    def details(self) -> dict[str, Any]: ...
    def prepare(self, state: SearchState, candidates: tuple[Candidate, ...],
                direction: str) -> dict[str, Any]: ...
    def invoke(self, prepared: dict[str, Any]) -> dict[str, Any]: ...
    def validate(self, prepared: dict[str, Any], raw: Any) -> AuditedDecision: ...


Controller = LocalController | AuditedController
