"""Per-query experiment logger."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.baselines.base import BaselineResult
from src.evaluation.metrics import compute_accuracy


class ExperimentLogger:
    """Collects per-query results and writes them to disk."""

    def __init__(self) -> None:
        self.results: list[BaselineResult] = []

    def log(self, result: BaselineResult) -> None:
        """Append a single query result."""
        self.results.append(result)

    def summary(self) -> dict:
        """Return aggregate statistics."""
        return compute_accuracy(self.results)

    def save(self, path: str | Path) -> None:
        """Write all per-query logs and summary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "summary": self.summary(),
            "per_query": [asdict(r) for r in self.results],
        }
        path.write_text(json.dumps(payload, indent=2))
