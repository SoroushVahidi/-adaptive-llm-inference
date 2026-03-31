"""Tests for external baseline wrapper infrastructure."""

import pytest

from src.baselines.external.best_route_wrapper import (
    BESTRouteAdaptedBaseline,
    BESTRouteBaseline,
)
from src.baselines.external.tale_wrapper import TALEBaseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DeterministicModel:
    """Deterministic fake model for testing BESTRouteAdaptedBaseline.

    Always returns the same answer string so routing decisions and outputs
    are fully reproducible.
    """

    def __init__(self, answer: str = "42") -> None:
        self._answer = answer

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return f"Final answer: {self._answer}"

    def generate_n(self, prompt: str, n: int) -> list[str]:  # noqa: ARG002
        return [f"Final answer: {self._answer}"] * n


class _HighScoreModel(_DeterministicModel):
    """Model that produces a first-pass output guaranteed to trigger escalation.

    Returns output with multiple uncertainty phrases and no clear parse
    success, which drives the confidence_proxy close to 0 and therefore
    the routing score close to 2 (well above any reasonable threshold).
    """

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        if "review" in prompt.lower() or "carefully review" in prompt.lower():
            # Revision call → return numeric answer
            return f"Final answer: {self._answer}"
        # First-pass call → ambiguous/uncertain output to force escalation
        return (
            "I'm not sure about this. It might be A or maybe B. "
            "I cannot determine the exact answer. Possibly the answer is uncertain."
        )


# ---------------------------------------------------------------------------
# BESTRouteBaseline (official-code wrapper, always blocked without .repo)
# ---------------------------------------------------------------------------


def test_tale_not_installed():
    baseline = TALEBaseline()
    assert baseline.name == "tale"
    assert not baseline.installed


def test_best_route_not_installed():
    baseline = BESTRouteBaseline()
    assert baseline.name == "best_route"
    assert not baseline.installed


def test_tale_solve_raises_without_install():
    baseline = TALEBaseline()
    with pytest.raises(RuntimeError, match="TALE is not installed"):
        baseline.solve("q1", "What is 2+2?", "4", 1)


def test_best_route_solve_raises_without_install():
    baseline = BESTRouteBaseline()
    with pytest.raises(RuntimeError, match="BEST-Route official code is not installed"):
        baseline.solve("q1", "What is 2+2?", "4", 1)


# ---------------------------------------------------------------------------
# BESTRouteAdaptedBaseline — basic interface
# ---------------------------------------------------------------------------


def test_adapted_name():
    model = _DeterministicModel()
    baseline = BESTRouteAdaptedBaseline(model)
    assert baseline.name == "best_route_adapted"


def test_adapted_default_threshold():
    model = _DeterministicModel()
    baseline = BESTRouteAdaptedBaseline(model)
    assert baseline.threshold == BESTRouteAdaptedBaseline.DEFAULT_THRESHOLD


def test_adapted_custom_threshold():
    model = _DeterministicModel()
    baseline = BESTRouteAdaptedBaseline(model, threshold=1.2)
    assert baseline.threshold == 1.2


def test_adapted_invalid_threshold_raises():
    model = _DeterministicModel()
    with pytest.raises(ValueError, match="threshold must be in"):
        BESTRouteAdaptedBaseline(model, threshold=3.0)


# ---------------------------------------------------------------------------
# BESTRouteAdaptedBaseline — budget enforcement (n_samples < 2)
# ---------------------------------------------------------------------------


def test_adapted_budget_too_low_always_uses_cheap():
    """When n_samples < 2, cheap action must always be used (no escalation)."""
    model = _DeterministicModel("7")
    # Use threshold=0.0 which would escalate everything if budget allowed
    baseline = BESTRouteAdaptedBaseline(model, threshold=0.0)
    result = baseline.solve("q1", "What is 3+4?", "7", n_samples=1)
    assert result.samples_used == 1
    assert result.metadata["action"] == "reasoning_greedy"
    assert result.metadata.get("budget_exceeded") is True


# ---------------------------------------------------------------------------
# BESTRouteAdaptedBaseline — routing decisions (deterministic)
# ---------------------------------------------------------------------------


def test_adapted_no_escalation_with_high_threshold():
    """With threshold=2.0 (max possible score), cheap action is always used."""
    model = _DeterministicModel("42")
    baseline = BESTRouteAdaptedBaseline(model, threshold=2.0)
    result = baseline.solve("q1", "What is 6×7?", "42", n_samples=2)
    assert result.samples_used == 1
    assert result.metadata["action"] == "reasoning_greedy"
    assert result.metadata["bubble_mode"] is True


def test_adapted_escalation_when_score_high():
    """Uncertain first-pass output should trigger escalation to revision."""
    model = _HighScoreModel("99")
    # Low threshold (0.0) ensures escalation whenever budget allows
    baseline = BESTRouteAdaptedBaseline(model, threshold=0.0)
    result = baseline.solve(
        "q1",
        "What is 100 - 1?",
        "99",
        n_samples=2,
    )
    assert result.samples_used == 2
    assert result.metadata["action"] == "direct_plus_revise"
    assert len(result.candidates) == 2


# ---------------------------------------------------------------------------
# BESTRouteAdaptedBaseline — result structure
# ---------------------------------------------------------------------------


def test_adapted_result_fields_present():
    """BaselineResult fields must all be populated."""
    model = _DeterministicModel("5")
    baseline = BESTRouteAdaptedBaseline(model, threshold=2.0)  # never escalate
    result = baseline.solve("q2", "2+3?", "5", n_samples=2)
    assert result.query_id == "q2"
    assert result.question == "2+3?"
    assert result.ground_truth == "5"
    assert isinstance(result.candidates, list)
    assert isinstance(result.correct, bool)
    assert isinstance(result.samples_used, int)
    assert "routing_score" in result.metadata
    assert "adaptation" in result.metadata
    assert result.metadata["adaptation"] == "binary_best_route"


def test_adapted_deterministic_same_seed():
    """Two calls with same input must produce identical results."""
    model = _DeterministicModel("12")
    baseline = BESTRouteAdaptedBaseline(model, threshold=1.5)
    r1 = baseline.solve("q1", "3 × 4?", "12", n_samples=2)
    r2 = baseline.solve("q1", "3 × 4?", "12", n_samples=2)
    assert r1.final_answer == r2.final_answer
    assert r1.samples_used == r2.samples_used
    assert r1.metadata["action"] == r2.metadata["action"]


# ---------------------------------------------------------------------------
# BESTRouteAdaptedBaseline — config parsing smoke test
# ---------------------------------------------------------------------------


def test_config_file_parseable():
    """The best_route_adapted YAML config must be parseable without errors."""
    import yaml
    from pathlib import Path

    cfg_path = (
        Path(__file__).resolve().parents[1] / "configs" / "best_route_adapted.yaml"
    )
    assert cfg_path.exists(), f"Config not found: {cfg_path}"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    assert cfg["baseline"] == "best_route_adapted"
    assert "threshold" in cfg
    assert isinstance(cfg["threshold"], float)
    assert "model" in cfg
    assert "datasets" in cfg
