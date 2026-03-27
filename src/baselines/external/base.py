"""Base class for external baseline wrappers.

External baselines wrap official code from published papers.  They live under
``external/<name>/`` and are adapted through thin wrappers here so they conform
to the same ``Baseline`` interface used by native baselines.
"""

from __future__ import annotations

from abc import abstractmethod

from src.baselines.base import Baseline, BaselineResult
from src.models.base import Model


class ExternalBaseline(Baseline):
    """Thin adapter for baselines backed by official author code.

    Subclasses should override ``_check_installation`` and ``solve``.
    The model may or may not be used — some external baselines bring their own
    model invocation.
    """

    def __init__(self, model: Model | None = None) -> None:
        if model is not None:
            super().__init__(model)
        self._installed = False

    def _check_installation(self) -> bool:
        """Return True if the external dependency is installed and importable."""
        return False

    @property
    def installed(self) -> bool:
        if not self._installed:
            self._installed = self._check_installation()
        return self._installed

    @abstractmethod
    def solve(
        self, query_id: str, question: str, ground_truth: str, n_samples: int
    ) -> BaselineResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
