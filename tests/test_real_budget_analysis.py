import pytest

from src.datasets.gsm8k import Query
from src.evaluation import real_budget_analysis
from src.evaluation.real_budget_analysis import flatten_real_budget_runs_for_csv, run_real_budget_sweep


def test_real_budget_sweep_runs_multiple_budgets_with_dummy_model() -> None:
    queries = [
        Query(id="q1", question="1+1?", answer="2"),
        Query(id="q2", question="2+2?", answer="4"),
        Query(id="q3", question="3+3?", answer="6"),
        Query(id="q4", question="4+4?", answer="8"),
        Query(id="q5", question="5+5?", answer="10"),
        Query(id="q6", question="6+6?", answer="12"),
        Query(id="q7", question="7+7?", answer="14"),
        Query(id="q8", question="8+8?", answer="16"),
    ]
    config = {
        "dataset": {
            "name": "gsm8k",
            "split": "test",
            "max_samples": 8,
        },
        "model": {
            "type": "dummy",
            "correct_prob": 0.3,
            "seed": 7,
        },
        "baselines": ["greedy", "best_of_n", "self_consistency"],
        "budgets": [8, 16],
        "n_samples": 3,
        "output_dir": "outputs/test_real_budget_sweep",
    }

    original_loader = real_budget_analysis.load_real_queries
    real_budget_analysis.load_real_queries = lambda _: queries
    try:
        result = run_real_budget_sweep(config)
    finally:
        real_budget_analysis.load_real_queries = original_loader

    assert result["budgets"] == [8, 16]
    assert set(result["available_baselines"]) == {"greedy", "best_of_n", "self_consistency"}
    assert len(result["runs"]) == 6
    assert len(result["comparisons"]) == 2
    for run in result["runs"]:
        assert run["total_queries"] == 8
        assert 0.0 <= run["accuracy"] <= 1.0
        assert run["total_compute_used"] >= 0
        assert run["average_compute_per_query"] >= 0.0


def test_flatten_real_budget_runs_for_csv_is_well_formed() -> None:
    rows = flatten_real_budget_runs_for_csv(
        [
            {
                "budget": 10,
                "baseline": "greedy",
                "total_queries": 5,
                "accuracy": 0.4,
                "total_compute_used": 5,
                "average_compute_per_query": 1.0,
            }
        ]
    )

    assert rows == [
        {
            "budget": 10,
            "baseline": "greedy",
            "total_queries": 5,
            "accuracy": 0.4,
            "total_compute_used": 5,
            "average_compute_per_query": 1.0,
        }
    ]


def test_real_budget_sweep_rejects_unavailable_model_type() -> None:
    config = {
        "dataset": {"name": "gsm8k", "split": "test", "max_samples": 4},
        "model": {"type": "api_model"},
        "baselines": ["greedy"],
        "budgets": [4],
    }

    original_loader = real_budget_analysis.load_real_queries
    real_budget_analysis.load_real_queries = lambda _: [
        Query(id="q1", question="1+1?", answer="2")
    ]
    try:
        with pytest.raises(ValueError, match="Unsupported model type"):
            run_real_budget_sweep(config)
    finally:
        real_budget_analysis.load_real_queries = original_loader
