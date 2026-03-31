# Final Figure Provenance

This maps each `outputs/paper_figures_final/*` artifact to exact committed inputs and generation code.

Generator script:
- `scripts/generate_final_manuscript_artifacts.py`

## Figure input mapping

### `cross_regime_accuracy_cost.png`
- Inputs:
  - `outputs/real_policy_eval/summary.json`
  - `outputs/real_hard_gsm8k_policy_eval/summary.json`
  - `outputs/real_hard_gsm8k_b2_policy_eval/summary.json`
  - `outputs/real_math500_policy_eval/summary.json`
  - `outputs/oracle_routing_eval/*.json`

### `routing_headroom_barplot.png`
- Inputs:
  - `outputs/paper_tables/oracle_headroom_table.csv`

### `routing_outcome_stacked_bar.png`
- Inputs:
  - `outputs/paper_tables/routing_outcome_breakdown.csv`

### `oracle_gap_barplot.png`
- Inputs:
  - Policy summaries listed above
  - Oracle summaries listed above

### `budget_curve_main.png`
- Inputs:
  - `outputs/budget_sweep/gsm8k_random100_budget_curve.csv`
  - `outputs/budget_sweep/hard_gsm8k_100_budget_curve.csv`
  - `outputs/budget_sweep/hard_gsm8k_b2_budget_curve.csv`
  - `outputs/budget_sweep/math500_100_budget_curve.csv`

### `threshold_tradeoff_curve.png`
- Inputs:
  - `outputs/baselines/confidence_threshold/confidence_threshold_sweep.csv`

### `adaptive_efficiency_scatter.png`
- Inputs:
  - Policy summaries listed above

### `graphic_abstract.png` / `graphic_abstract.pdf` / `graphic_abstract_caption.txt`
- Inputs:
  - Policy summaries listed above
  - Oracle summaries listed above
  - Canonical decisions encoded in `docs/CANONICAL_MANUSCRIPT_DECISIONS.md`

## Reproduction command

```bash
python3 scripts/generate_final_manuscript_artifacts.py
```
