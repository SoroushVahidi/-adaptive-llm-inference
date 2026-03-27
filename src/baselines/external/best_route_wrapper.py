"""Placeholder wrapper for the BEST-Route baseline.

Paper: "Adaptive LLM Routing with Test-Time Optimal Compute"
Official code: https://github.com/best-route/best-route

This wrapper is a stub.  Once the official repo is cloned into
``external/best_route/.repo``, flesh out the integration here.
"""

from __future__ import annotations

from pathlib import Path

from src.baselines.base import BaselineResult
from src.baselines.external.base import ExternalBaseline
from src.models.base import Model

_REPO_DIR = Path(__file__).resolve().parents[3] / "external" / "best_route" / ".repo"


class BESTRouteBaseline(ExternalBaseline):
    """Wrapper around the official BEST-Route implementation."""

    def __init__(self, model: Model | None = None) -> None:
        super().__init__(model)

    @property
    def name(self) -> str:
        return "best_route"

    def _check_installation(self) -> bool:
        return _REPO_DIR.exists()

    def solve(
        self, query_id: str, question: str, ground_truth: str, n_samples: int
    ) -> BaselineResult:
        if not self.installed:
            raise RuntimeError(
                f"BEST-Route is not installed. Clone the official repo into {_REPO_DIR}. "
                "See external/best_route/README.md for instructions."
            )
        raise NotImplementedError(
            "BEST-Route integration is not yet implemented. "
            "Implement the bridge between official BEST-Route code and our Baseline interface."
        )
