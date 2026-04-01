from __future__ import annotations

from typing import Any, Protocol


class Optimizer(Protocol):
    name: str

    def solve(self, candidate_rows: list[dict[str, Any]], budget: float) -> dict[str, Any]:
        ...

