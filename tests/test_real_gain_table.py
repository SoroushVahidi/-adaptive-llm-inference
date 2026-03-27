from src.evaluation.real_gain_table import build_budget_comparison, summarize_gain_table


def test_budget_comparison_oracle_beats_uniform_when_headroom_exists() -> None:
    gain_rows = [
        {
            "question_id": "q1",
            "gold_answer": "7",
            "empirical_success_k1": 0.2,
            "empirical_success_k2": 0.9,
            "empirical_success_k3": 0.9,
            "marginal_gain_1_to_2": 0.7,
            "marginal_gain_2_to_3": 0.0,
        },
        {
            "question_id": "q2",
            "gold_answer": "9",
            "empirical_success_k1": 0.6,
            "empirical_success_k2": 0.65,
            "empirical_success_k3": 0.66,
            "marginal_gain_1_to_2": 0.05,
            "marginal_gain_2_to_3": 0.01,
        },
        {
            "question_id": "q3",
            "gold_answer": "3",
            "empirical_success_k1": 0.7,
            "empirical_success_k2": 0.71,
            "empirical_success_k3": 0.72,
            "marginal_gain_1_to_2": 0.01,
            "marginal_gain_2_to_3": 0.01,
        },
    ]

    comparison = build_budget_comparison(gain_rows, budgets=[3, 4, 5, 6])

    assert len(comparison) == 4
    row_budget_4 = next(row for row in comparison if row["budget"] == 4)
    assert row_budget_4["oracle_expected_solved"] > row_budget_4["uniform_expected_solved"]
    assert row_budget_4["absolute_gap"] > 0.0


def test_summarize_gain_table_reports_headroom_statistics() -> None:
    gain_rows = [
        {
            "question_id": "q1",
            "gold_answer": "7",
            "empirical_success_k1": 0.2,
            "empirical_success_k2": 0.5,
            "empirical_success_k3": 0.6,
            "marginal_gain_1_to_2": 0.3,
            "marginal_gain_2_to_3": 0.1,
        },
        {
            "question_id": "q2",
            "gold_answer": "9",
            "empirical_success_k1": 0.6,
            "empirical_success_k2": 0.6,
            "empirical_success_k3": 0.6,
            "marginal_gain_1_to_2": 0.0,
            "marginal_gain_2_to_3": 0.0,
        },
        {
            "question_id": "q3",
            "gold_answer": "3",
            "empirical_success_k1": 0.2,
            "empirical_success_k2": 0.6,
            "empirical_success_k3": 0.65,
            "marginal_gain_1_to_2": 0.4,
            "marginal_gain_2_to_3": 0.05,
        },
    ]
    budget_rows = build_budget_comparison(gain_rows, budgets=[3, 4, 5, 6, 7, 8, 9])

    summary = summarize_gain_table(gain_rows, budget_rows)

    assert summary["fraction_positive_gain"] == 2 / 3
    assert summary["fraction_no_gain"] == 1 / 3
    assert summary["budget_where_oracle_beats_uniform_most"] in {4, 5}
