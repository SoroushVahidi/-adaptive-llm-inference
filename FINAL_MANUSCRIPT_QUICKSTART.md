# Final Manuscript Quickstart

This is the single best entry point for using the repository as a paper companion package.

## Final Scope (Main Paper)

- Model/evidence family: committed `gpt-4o-mini` routing artifacts
- Main regimes:
  - `gsm8k_random_100`
  - `hard_gsm8k_100`
  - `hard_gsm8k_b2`
  - `math500_100`
- Canonical adaptive policy for headline comparison: `adaptive_policy_v5`
- Transparency policy: `adaptive_policy_v6` and `adaptive_policy_v7` remain in main comparison tables

## Canonical Artifact Directories

- Final tables: `outputs/paper_tables_final/`
- Final figures: `outputs/paper_figures_final/`
- Graphic abstract:
  - `outputs/paper_figures_final/graphic_abstract.png`
  - `outputs/paper_figures_final/graphic_abstract.pdf`
  - `outputs/paper_figures_final/graphic_abstract_caption.txt`

## Table-by-Table Guide

- `main_results_summary.csv` — main regime summary (cheap, adaptive, always-revise, oracle)
- `cross_regime_summary.csv` — normalized four-regime comparison
- `policy_comparison_main.csv` — per-policy comparison rows used in manuscript text/plots
- `oracle_headroom_main.csv` — oracle-gap/headroom view
- `routing_outcome_breakdown_main.csv` — RG-only / DPR-only decomposition
- `budget_curve_main_points.csv` — canonical budget points (`1.0`, `1.1`, `1.2`, `2.0`)
- `statistical_support_main.csv` — bootstrap CIs and paired tests
- `baseline_comparison_appendix.csv` — appendix-only baseline table

## Figure-by-Figure Guide

- `cross_regime_accuracy_cost.png` — compact cross-regime accuracy/cost comparison
- `routing_headroom_barplot.png` — headroom by regime
- `routing_outcome_stacked_bar.png` — outcome decomposition
- `oracle_gap_barplot.png` — oracle gap reduction view
- `budget_curve_main.png` — budget curves across the four main regimes
- `adaptive_efficiency_scatter.png` — adaptive vs baseline cost/accuracy positioning
- `threshold_tradeoff_curve.png` — supplementary threshold tradeoff plot

## Regenerate Final Assets

```bash
python3 scripts/generate_final_manuscript_artifacts.py
```

This command performs a deterministic export from already committed artifacts and writes:
- final tables and figures
- graphic abstract
- `outputs/paper_tables_final/FINAL_ARTIFACT_EXPORT_REPORT.md`

## Explicitly Not Main-Paper in This Package

- AIME small-pass outputs (`outputs/small_pass/`, `outputs/paper_tables_small_pass/`)
- GPQA-Diamond artifacts (`outputs/real_gpqa_routing_dataset/`) in this manuscript pass
- Historical/intermediate directories:
  - `outputs/paper_tables/`
  - `outputs/paper_tables_cleaned/`
  - `outputs/paper_tables_enhanced/`
  - `outputs/paper_figures/`
  - `outputs/paper_figures_enhanced/`

## Canonical Decision Docs

- `docs/CANONICAL_MANUSCRIPT_DECISIONS.md`
- `docs/FINAL_CONSISTENCY_AUDIT.md`
- `docs/FINAL_FIGURE_PROVENANCE.md`
- `docs/FINAL_REPO_STATUS.md`
