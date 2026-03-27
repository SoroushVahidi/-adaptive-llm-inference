"""Placeholder wrapper for the TALE baseline.

Paper: "Token-Budget-Aware LLM Reasoning"
Official code: https://github.com/ChenWu98/TALE

This wrapper is a stub.  Once the official repo is cloned into
``external/tale/.repo``, flesh out the integration here.
"""

from __future__ import annotations

from pathlib import Path

from src.baselines.base import BaselineResult
from src.baselines.external.base import ExternalBaseline
from src.models.base import Model

_REPO_DIR = Path(__file__).resolve().parents[3] / "external" / "tale" / ".repo"


class TALEBaseline(ExternalBaseline):
    """Wrapper around the official TALE implementation."""

    def __init__(self, model: Model | None = None) -> None:
        super().__init__(model)

    @property
    def name(self) -> str:
        return "tale"

    def _check_installation(self) -> bool:
        return _REPO_DIR.exists()

    def solve(
        self, query_id: str, question: str, ground_truth: str, n_samples: int
    ) -> BaselineResult:
        if not self.installed:
            raise RuntimeError(
                f"TALE is not installed. Clone the official repo into {_REPO_DIR}. "
                "See external/tale/README.md for instructions."
            )
        raise NotImplementedError(
            "TALE integration is not yet implemented. "
            "Implement the bridge between official TALE code and our Baseline interface."
        )
