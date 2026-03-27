from src.methods.mode_then_budget import (
    ModeThenBudgetConfig,
    compute_direct_mode_signals,
    run_mode_then_budget,
    score_mode_switch,
)


def test_direct_mode_signals_detect_basic_issues() -> None:
    signals = compute_direct_mode_signals(
        direct_output="",
        direct_answer="",
        reasoning_probe_output="Answer: 12",
        reasoning_probe_answer="12",
    )

    assert signals["parse_failure"] is True
    assert signals["malformed_output"] is True
    assert signals["direct_reasoning_disagreement"] is True


def test_score_mode_switch_combines_weights() -> None:
    config = ModeThenBudgetConfig(
        total_budget=10,
        weight_parse_failure=2.0,
        weight_malformed_output=1.0,
        weight_low_confidence_format=0.5,
        weight_direct_reasoning_disagreement=1.5,
        min_switch_score=2.0,
    )
    score = score_mode_switch(
        {
            "parse_failure": True,
            "malformed_output": False,
            "low_confidence_format": True,
            "direct_reasoning_disagreement": True,
        },
        config,
    )

    assert score == 4.0


def test_run_mode_then_budget_returns_expected_structure() -> None:
    class _FakeDirectModel:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str) -> str:
            values = ["Answer: 4", "", "Answer: 7", "Answer: 8"]
            value = values[self.calls % len(values)]
            self.calls += 1
            return value

    class _FakeReasoningModel:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str) -> str:
            values = [
                "Final answer: 4",
                "Final answer: 5",
                "Final answer: 7",
                "Final answer: 8",
            ]
            value = values[self.calls % len(values)]
            self.calls += 1
            return value

        def generate_n(self, prompt: str, n: int) -> list[str]:
            values = ["Final answer: 4", "Final answer: 4", "Final answer: 4"]
            self.calls += n
            return values[:n]

    config = ModeThenBudgetConfig(total_budget=6, min_switch_score=1.0)
    result = run_mode_then_budget(
        direct_model=_FakeDirectModel(),
        reasoning_model=_FakeReasoningModel(),
        questions=["2+2", "5+0"],
        total_budget=6,
        config=config,
    )

    assert len(result["rows"]) == 2
    assert result["total_samples_used"] >= 2
    assert result["queries_switched_to_reasoning"] >= 0
