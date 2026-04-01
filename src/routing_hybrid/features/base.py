from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class FeatureExtractor(Protocol):
    name: str

    def transform(self, candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ...


@dataclass
class SimpleFeatureExtractor:
    name: str
    fn: Any

    def transform(self, candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.fn(candidate_rows)

