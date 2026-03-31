# Final Repo Status

Date: 2026-03-31

## What is complete for the paper package

- Canonical decisions documented:
  - `docs/CANONICAL_MANUSCRIPT_DECISIONS.md`
  - `docs/FINAL_CONSISTENCY_AUDIT.md`
- Deterministic final exporter available:
  - `scripts/generate_final_manuscript_artifacts.py`
- Final canonical assets generated and organized:
  - `outputs/paper_tables_final/*`
  - `outputs/paper_figures_final/*`

## Supplementary-only material

- `outputs/paper_tables_final/baseline_comparison_appendix.csv`
- `outputs/paper_tables_final/statistical_support_main.csv`
- `outputs/paper_figures_final/threshold_tradeoff_curve.png`
- AIME small-pass outputs and GPQA artifacts remain outside main-paper canonical set in this pass.

## Remaining human manuscript-writing decisions (not repo blockers)

- Whether to emphasize v5 alone in headline prose, or co-emphasize v6 as a lower-cost tied operating point in easy regimes.
- Whether to include GPQA-Diamond as appendix/supplementary in this submission cycle.

## Main evidence-base limitations

- Main evidence is based on four n=100 regimes with one model family.
- Some baseline rollups are sample-size-mismatched relative to main n=100 results and are therefore appendix-only.
- Oracle is an upper bound and not a deployable route.

## Safe claims supported by repository artifacts

- Adaptive routing improves or matches cheap baseline accuracy with lower cost than always-revise across the four canonical regimes.
- Regime difficulty structure matters (revise-helpful rates differ materially by regime).
- Oracle analysis demonstrates measurable remaining routing headroom on hard regimes.
- Budget-aware curves support accuracy-cost tradeoff framing within canonical regime scope.
