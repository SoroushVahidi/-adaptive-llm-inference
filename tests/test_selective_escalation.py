from src.methods.selective_escalation import (
    SelectiveEscalationConfig,
    compute_escalation_signals,
    run_selective_escalation,
    score_escalation,
)


def test_compute_escalation_signals_detects_basic_indicators() -> None:
    signals = compute_escalation_signals(
        first_output="I am not sure.",
        second_output="Answer: 12",
    )

    assert signals["parse_failure"] is True
    assert signals["disagreement_2sample"] is True
    assert signals["malformed_output"] is False
    assert signals["low_confidence_format"] is False


def test_score_escalation_combines_signal_weights() -> None:
    config = SelectiveEscalationConfig(
        total_budget=10,
        weight_parse_failure=2.0,
        weight_disagreement_2sample=1.5,
        weight_malformed_output=0.5,
        weight_low_confidence_format=0.25,
    )
    signals = {
        "parse_failure": True,
        "disagreement_2sample": True,
        "malformed_output": False,
        "low_confidence_format": True,
    }
    score = score_escalation(signals, config)

    assert score == 3.75


def test_run_selective_escalation_returns_expected_structure() -> None:
    class _FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str) -> str:  # noqa: ARG002
            responses = [
                "The answer is 4",
                "The answer is unknown",
                "The answer is 7",
                "The answer is 8",
            ]
            value = responses[self.calls % len(responses)]
            self.calls += 1
            return value

        def generate_n(self, prompt: str, n: int) -> list[str]:  # noqa: ARG002
            base = [
                ["The answer is 4", "The answer is 4"],
                ["The answer is 5", "The answer is 5"],
                ["The answer is 7", "The answer is 7"],
                ["The answer is 8", "The answer is 8"],
            ]
            value = base[(self.calls // 2) % len(base)]
            self.calls += n
            return value[:n]

    queries = ["2+2", "5+0"]
    model = _FakeModel()
    config = SelectiveEscalationConfig(
        total_budget=6,
        escalation_target_k=3,
        use_second_sample_for_disagreement=True,
        weight_parse_failure=2.0,
        weight_disagreement_2sample=1.0,
        weight_malformed_output=1.0,
        weight_low_confidence_format=1.0,
    )
    result = run_selective_escalation(
        model=model,
        questions=queries,
        total_budget=4,
        config=config,
    )

    assert len(result["diagnostics"]) == 2
    assert result["total_samples_used"] >= 2
    assert result["queries_escalated"] >= 0
