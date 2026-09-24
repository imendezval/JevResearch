"""A bounded operator returns a new complete configuration."""

from typing import Any, Protocol


class ConfigOperator(Protocol):
    name: str

    def parameters(self) -> dict[str, Any]: ...
    def apply(self, config: dict[str, Any]) -> dict[str, Any]: ...
