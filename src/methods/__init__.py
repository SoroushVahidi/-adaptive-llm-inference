"""Lightweight method prototypes for adaptive test-time compute."""

from src.methods.mode_then_budget import ModeThenBudgetConfig, run_mode_then_budget
from src.methods.selective_escalation import (
    SelectiveEscalationConfig,
    compute_escalation_signals,
    run_selective_escalation,
    score_escalation,
)

__all__ = [
    "ModeThenBudgetConfig",
    "SelectiveEscalationConfig",
    "compute_escalation_signals",
    "run_mode_then_budget",
    "run_selective_escalation",
    "score_escalation",
]
