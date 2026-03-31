# Manuscript Writing Aid

Practical section-to-artifact guide for drafting/revision, grounded in canonical final assets only.

Canonical asset roots:
- `outputs/paper_tables_final/`
- `outputs/paper_figures_final/`

## Section-to-Asset Map

| Manuscript section | Primary assets to cite | Best supporting signal |
|---|---|---|
| Introduction motivation | `cross_regime_summary.csv`, `routing_headroom_barplot.png` | Regime-dependent revise-helpful rates and nonzero oracle headroom |
| Methodology intuition | `graphic_abstract.png`, `policy_comparison_main.csv` | Cheap vs revise decision framing with adaptive routing |
| Cross-regime headroom | `oracle_headroom_main.csv`, `oracle_gap_barplot.png` | Quantified gap from adaptive/cheap to oracle |
| Policy comparison | `main_results_summary.csv`, `policy_comparison_main.csv`, `cross_regime_accuracy_cost.png` | Accuracy-cost tradeoffs across cheap/adaptive/always-revise/oracle |
| Budget-aware analysis | `budget_curve_main_points.csv`, `budget_curve_main.png` | Behavior at canonical cost points (1.0, 1.1, 1.2, 2.0) |
| Discussion / limitations | `baseline_comparison_appendix.csv`, `statistical_support_main.csv`, `FINAL_REPO_STATUS.md` | n-scope boundaries, baseline caveats, confidence intervals/paired tests |

## Safe sentence templates (artifact-grounded)

### Introduction motivation
- "Across the four canonical regimes, revise-helpful prevalence is regime-dependent, indicating that selective escalation is a routing problem rather than a one-size-fits-all policy (`outputs/paper_tables_final/cross_regime_summary.csv`)."
- "Oracle analysis shows nonzero headroom over deployable policies on harder regimes, motivating adaptive routing (`outputs/paper_tables_final/oracle_headroom_main.csv`)."

### Methodology intuition
- "Our deployment framing compares a cheap single-pass route against a more expensive revise route, with a lightweight adaptive policy making per-query escalation decisions (`outputs/paper_figures_final/graphic_abstract.png`)."
- "We report v5/v6/v7 transparently while using v5 as the canonical headline adaptive comparator (`docs/CANONICAL_MANUSCRIPT_DECISIONS.md`)."

### Cross-regime headroom
- "Headroom is quantified as the gap between oracle and adaptive policy accuracy, and varies across regimes (`outputs/paper_tables_final/oracle_headroom_main.csv`)."
- "Routing outcome decomposition shows where revise-only wins are concentrated (`outputs/paper_tables_final/routing_outcome_breakdown_main.csv`)."

### Policy comparison
- "Adaptive routing improves or matches cheap-route accuracy while remaining less costly than always-revise on the canonical regimes (`outputs/paper_tables_final/main_results_summary.csv`)."
- "Cross-regime plots show this as an explicit accuracy-cost tradeoff rather than a single-metric gain (`outputs/paper_figures_final/cross_regime_accuracy_cost.png`)."

### Budget-aware analysis
- "At canonical budget points, accuracy changes are reported from committed budget-curve artifacts without retuning claims (`outputs/paper_tables_final/budget_curve_main_points.csv`)."
- "The multi-panel budget figure provides the main operational tradeoff view across all four regimes (`outputs/paper_figures_final/budget_curve_main.png`)."

### Discussion / limitations
- "Baseline rollups with sample-size mismatch are appendix-only and should not be treated as main-table equivalents (`outputs/paper_tables_final/baseline_comparison_appendix.csv`)."
- "Confidence intervals and paired tests are provided as supporting statistics and should be interpreted with n=100 regime scope in mind (`outputs/paper_tables_final/statistical_support_main.csv`)."
