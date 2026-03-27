"""Abstract model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Model(ABC):
    """Base class for answer-generation models.

    Subclasses must implement ``generate`` which, given a question string,
    returns a single candidate answer string.  Calling ``generate`` multiple
    times with the same question may return different answers (stochastic
    sampling).
    """

    @abstractmethod
    def generate(self, question: str) -> str:
        """Return one candidate answer for *question*."""

    def generate_n(self, question: str, n: int) -> list[str]:
        """Return *n* independent candidate answers for *question*."""
        return [self.generate(question) for _ in range(n)]
