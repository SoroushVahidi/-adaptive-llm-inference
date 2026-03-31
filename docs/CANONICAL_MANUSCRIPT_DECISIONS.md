# Canonical Manuscript Decisions

Date: 2026-03-31  
Evidence basis: committed outputs only; no new API experiments.

## 1) Canonical Regimes (Main Paper)

Main-paper regimes are fixed to:

1. `gsm8k_random_100`
2. `hard_gsm8k_100`
3. `hard_gsm8k_b2`
4. `math500_100`

Exploratory/supplementary only:
- `aime2024` (small-pass exploratory)
- `gpqa_diamond` (build artifacts present; policy-eval not integrated into main-paper set in this pass)

## 2) Canonical Policy Story

- Main comparison includes: `reasoning_greedy`, `adaptive_policy_v5`, `adaptive_policy_v6`, `adaptive_policy_v7`, `direct_plus_revise`, `oracle`.
- Canonical primary adaptive policy for headline comparisons: **`adaptive_policy_v5`**.

Rationale (repository-grounded):
- v5 attains the highest adaptive accuracy (or ties highest) on the four canonical regimes.
- When ties occur (e.g., GSM8K), v6 remains reported as lower-cost operating point.
- v6/v7 are retained in main comparison tables to avoid hiding stronger/weaker variants.

## 3) Main vs Supplementary Placement

Main paper:
- `outputs/paper_tables_final/main_results_summary.csv`
- `outputs/paper_tables_final/cross_regime_summary.csv`
- `outputs/paper_tables_final/policy_comparison_main.csv`
- `outputs/paper_tables_final/oracle_headroom_main.csv`
- `outputs/paper_tables_final/routing_outcome_breakdown_main.csv`
- `outputs/paper_tables_final/budget_curve_main_points.csv`
- `outputs/paper_figures_final/cross_regime_accuracy_cost.png`
- `outputs/paper_figures_final/routing_headroom_barplot.png`
- `outputs/paper_figures_final/routing_outcome_stacked_bar.png`
- `outputs/paper_figures_final/oracle_gap_barplot.png`
- `outputs/paper_figures_final/budget_curve_main.png`
- `outputs/paper_figures_final/adaptive_efficiency_scatter.png`
- `outputs/paper_figures_final/graphic_abstract.png` + `.pdf`

Supplementary/appendix:
- `outputs/paper_tables_final/baseline_comparison_appendix.csv`
- `outputs/paper_tables_final/statistical_support_main.csv`
- `outputs/paper_figures_final/threshold_tradeoff_curve.png`

## 4) Canonical Source-of-Truth Rule

- Authoritative final manuscript assets are only the `*_final` directories:
  - `outputs/paper_tables_final/`
  - `outputs/paper_figures_final/`
- Older `paper_tables`, `paper_tables_cleaned`, `paper_tables_enhanced`,
  `paper_figures`, and `paper_figures_enhanced` are historical/intermediate outputs.

## 5) Results Classified as Exploratory

- AIME small-pass outputs under `outputs/small_pass/` and `outputs/paper_tables_small_pass/`.
- GPQA-Diamond artifacts under `outputs/real_gpqa_routing_dataset/` until full policy-eval integration into the manuscript table stack.
