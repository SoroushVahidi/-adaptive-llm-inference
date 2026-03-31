# Final Manuscript Asset Index

Authoritative index for final manuscript-support assets.
Canonical output families only:
- `outputs/paper_tables_final/`
- `outputs/paper_figures_final/`

| Path | Type | Purpose | Placement | Canonical/Historical | Regeneration source |
|---|---|---|---|---|---|
| `FINAL_MANUSCRIPT_QUICKSTART.md` | supporting doc | first-stop manuscript guide | main | canonical | manual doc |
| `docs/CANONICAL_MANUSCRIPT_DECISIONS.md` | supporting doc | canonical policy/regime decisions | main | canonical | manual doc |
| `docs/FINAL_CONSISTENCY_AUDIT.md` | supporting doc | consistency issues + resolutions | main | canonical | manual doc |
| `docs/FINAL_FIGURE_PROVENANCE.md` | supporting doc | final figure input provenance | main | canonical | manual doc |
| `docs/FINAL_REPO_STATUS.md` | supporting doc | submission-readiness status and limitations | main | canonical | manual doc |
| `docs/FINAL_COHERENCE_CHECK.md` | supporting doc | lightweight final coherence verification | main | canonical | manual doc |
| `docs/MANUSCRIPT_WRITING_AID.md` | supporting doc | section-by-section writing guide | main | canonical | manual doc |
| `outputs/paper_tables_final/main_results_summary.csv` | main table | high-level results summary | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_tables_final/cross_regime_summary.csv` | main table | normalized cross-regime comparison | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_tables_final/policy_comparison_main.csv` | main table | per-policy comparison rows | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_tables_final/oracle_headroom_main.csv` | main table | oracle headroom quantification | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_tables_final/routing_outcome_breakdown_main.csv` | main table | outcome decomposition | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_tables_final/budget_curve_main_points.csv` | main table | canonical budget points | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_tables_final/statistical_support_main.csv` | appendix table | CIs + paired tests support | appendix/supp | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_tables_final/baseline_comparison_appendix.csv` | appendix table | baseline comparison with caveats | appendix/supp | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_tables_final/README.md` | supporting doc | table-file usage notes | main | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_tables_final/FINAL_ARTIFACT_EXPORT_REPORT.md` | supporting doc | export summary report | main | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_figures_final/cross_regime_accuracy_cost.png` | main figure | cross-regime accuracy/cost view | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_figures_final/routing_headroom_barplot.png` | main figure | headroom by regime | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_figures_final/routing_outcome_stacked_bar.png` | main figure | routing outcomes by regime | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_figures_final/oracle_gap_barplot.png` | main figure | oracle-gap comparison | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_figures_final/budget_curve_main.png` | main figure | budget-accuracy curves | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_figures_final/adaptive_efficiency_scatter.png` | main figure | adaptive efficiency scatter | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_figures_final/threshold_tradeoff_curve.png` | appendix figure | confidence-threshold tradeoff | appendix/supp | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_figures_final/graphic_abstract.png` | graphic abstract | journal-ready visual abstract | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_figures_final/graphic_abstract.pdf` | graphic abstract | vector-friendly abstract export | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |
| `outputs/paper_figures_final/graphic_abstract_caption.txt` | supporting doc | abstract caption text | main paper | canonical | `scripts/generate_final_manuscript_artifacts.py` |

Historical/background-only families (not canonical for manuscript citation):
- `outputs/paper_tables/*`
- `outputs/paper_tables_cleaned/*`
- `outputs/paper_tables_enhanced/*`
- `outputs/paper_figures/*`
- `outputs/paper_figures_enhanced/*`
