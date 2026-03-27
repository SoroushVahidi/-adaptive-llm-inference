"""Lightweight method prototypes for adaptive test-time compute."""

from src.methods.selective_escalation import (
    SelectiveEscalationConfig,
    compute_escalation_signals,
    run_selective_escalation,
    score_escalation,
)

__all__ = [
    "SelectiveEscalationConfig",
    "compute_escalation_signals",
    "run_selective_escalation",
    "score_escalation",
]
