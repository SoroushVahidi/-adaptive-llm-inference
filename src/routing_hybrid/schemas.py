from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CandidateRow:
    prompt_id: str
    regime: str
    question: str
    split: str
    action_name: str
    action_family: str
    action_cost: float
    correctness_label: int
    answer_format: str
    utility_labels: dict[str, float]
    metadata: dict[str, Any]


@dataclass
class OptimizationResult:
    chosen_by_prompt: dict[str, str]
    objective_value: float
    total_cost: float
    budget: float
    optimizer_name: str

