from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class HeuristicRule(Protocol):
    name: str

    def apply(self, candidate_row: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class RuleResult:
    forbidden: bool = False
    dominated: bool = False
    utility_adjustment: float = 0.0

