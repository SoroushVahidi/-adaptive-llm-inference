# Final Manuscript Table and Figure Index

**Purpose:** Definitive list of all publication-ready tables and figures, their file paths, section
assignments, and main-paper vs appendix status.

**Cleaned assets root:**
- Tables → `outputs/paper_tables_cleaned/`
- Figures → `outputs/paper_figures_cleaned/`

**Original (unpolished) exports remain at:**
- `outputs/paper_tables/`
- `outputs/paper_figures/`

Do not cite original paths in the paper draft. Use the `_cleaned` paths below.

---

## A. Final Table List

### Main-Paper Tables

| Table # | Title | File | Paper section |
|---------|-------|------|---------------|
| **Table 1** | Main results summary — reasoning, revise, best adaptive policy, oracle, revise-helpful rate | `outputs/paper_tables_cleaned/main_results_summary.csv` | §4 Main Results |
| **Table 2** | Policy evaluation — accuracy, avg cost, revise rate, oracle gap by method and regime | `outputs/paper_tables_cleaned/policy_eval_comparison.csv` | §4 Main Results |
| **Table 3** | Cross-regime comparison — 4 datasets (incl. hard_gsm8k_b2 added) | `outputs/paper_tables_cleaned/final_cross_regime_summary_fixed.csv` | §4.2 Cross-Regime |
| **Table 4** | Oracle routing upper bounds — accuracy, cost, revise rate per dataset | `outputs/paper_tables_cleaned/oracle_routing_eval.csv` | §4.3 Oracle Ceiling |
| **Table 5** | Budget curves — accuracy vs target cost across all datasets | `outputs/paper_tables_cleaned/budget_curves_all_datasets.csv` | §4.4 Budget Analysis |

### Appendix-Only Tables

| Table # | Title | File | Reason for appendix |
|---------|-------|------|---------------------|
| **Table A1** | Baseline strategy rollup (n=15–30; mismatched sample size) | `outputs/paper_tables_cleaned/baselines_appendix.csv` | n=30/15 does not match main n=100 slices; cannot compare directly |
| **Table A2** | Regime-level routing model viability | `outputs/paper_tables_cleaned/cross_regime_summary.csv` | Auxiliary detail; 3-regime version without B2 |

---

## B. Final Figure List

### Main-Paper Figures

| Figure # | Title | File | Paper section | Notes |
|----------|-------|------|---------------|-------|
| **Figure 1** | 2×2 composite: accuracy vs cost, all four regimes | `outputs/paper_figures_cleaned/real_routing/accuracy_vs_cost_2x2_composite.png` | §4 Main Results | Primary overview figure; consistent legend across panels |
| **Figure 1a** | Accuracy vs cost: Hard GSM8K-100 | `outputs/paper_figures_cleaned/real_routing/accuracy_vs_cost_hard_gsm8k_100.png` | §4 Main Results | Standalone panel if 2×2 is too small for venue |
| **Figure 1b** | Accuracy vs cost: Hard GSM8K-B2 | `outputs/paper_figures_cleaned/real_routing/accuracy_vs_cost_hard_gsm8k_b2.png` | §4 Main Results | Replication of hard-regime story |
| **Figure 2a** | Budget curve: Hard GSM8K-B2 | `outputs/paper_figures_cleaned/next_stage/budget_curve_hard_gsm8k_b2.png` | §4.4 Budget Analysis | Best single budget curve (largest accuracy lift) |
| **Figure 2b** | Budget curve: Hard GSM8K-100 | `outputs/paper_figures_cleaned/next_stage/budget_curve_hard_gsm8k_100.png` | §4.4 Budget Analysis | Paired hard-regime budget figure |

### Supporting Figures (include if space permits; otherwise appendix)

| Figure # | Title | File | Suggested placement |
|----------|-------|------|---------------------|
| **Figure S1** | Accuracy vs cost: MATH500-100 | `outputs/paper_figures_cleaned/real_routing/accuracy_vs_cost_math500_100.png` | Appendix or §4.2 supporting |
| **Figure S2** | Budget curve: MATH500-100 | `outputs/paper_figures_cleaned/next_stage/budget_curve_math500_100.png` | Appendix or §4.2 supporting |

### Appendix-Only Figures

| Figure # | Title | File | Reason for appendix |
|----------|-------|------|---------------------|
| **Figure A1** | Accuracy vs cost: GSM8K random-100 (easy / control) | `outputs/paper_figures_cleaned/real_routing/accuracy_vs_cost_gsm8k_random_100.png` | Control regime; shows convergence — supports but doesn't advance main claim |
| **Figure A2** | Budget curve: GSM8K random-100 | `outputs/paper_figures_cleaned/next_stage/budget_curve_gsm8k_random_100.png` | Flat curve; control only |
| **Figure A3** | Cascade curve: Hard GSM8K-100 | `outputs/paper_figures_cleaned/next_stage/cascade_curve_hard_gsm8k_100.png` | Threshold sensitivity detail |
| **Figure A4** | Cascade curve: Hard GSM8K-B2 | `outputs/paper_figures_cleaned/next_stage/cascade_curve_hard_gsm8k_b2.png` | Threshold sensitivity detail |
| **Figure A5** | Cascade curve: MATH500-100 | `outputs/paper_figures_cleaned/next_stage/cascade_curve_math500_100.png` | Threshold sensitivity detail |
| **Figure A6** | Cascade curve: GSM8K random-100 | `outputs/paper_figures_cleaned/next_stage/cascade_curve_gsm8k_random_100.png` | Control / appendix only |

### Blocked Figures (do not reference in paper until inputs exist)

| Group | Blocking input | How to unblock |
|-------|---------------|----------------|
| Simulated MCKP allocation (3 PNGs) | `outputs/simulated_sweep/budget_sweep_comparisons.csv` | `python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml` |

---

## C. Section ↔ Asset Cross-Reference

| Paper section | Tables | Figures |
|---------------|--------|---------|
| §1 Introduction | — | — |
| §2 Background / Related Work | — | — |
| §3 Method | — | — |
| **§4 Main Results** | Table 1, Table 2 | Figure 1 (2×2 composite), Figure 1a, Figure 1b |
| **§4.2 Cross-Regime Analysis** | Table 3 | Figure S1 (optional) |
| **§4.3 Oracle Ceiling** | Table 4 | — |
| **§4.4 Budget Analysis** | Table 5 | Figure 2a, Figure 2b |
| §5 Discussion / Limitations | — | — |
| **Appendix A: Baselines** | Table A1 | — |
| **Appendix B: Routing Viability** | Table A2 | — |
| **Appendix C: Additional Figures** | — | Figure A1–A6, Figure S1–S2 |

---

## D. Main Paper vs Appendix Summary

### Main paper (submit with manuscript body)

**Tables:** Table 1, Table 2, Table 3, Table 4, Table 5
**Figures:** Figure 1 (or 1a+1b), Figure 2a, Figure 2b

### Appendix / supplementary

**Tables:** Table A1 (baselines, with explicit n=30/15 caveat), Table A2
**Figures:** Figure A1–A6, Figure S1, Figure S2

---

## E. Fixes Applied in This Pass

| Issue | Fix | Affected files |
|-------|-----|----------------|
| Inconsistent dataset naming (`gsm8k_random100` vs `gsm8k_easy100`) | Normalised to `gsm8k_random_100` throughout | All `_cleaned` CSVs |
| `eval_summary` column contained internal pipeline names | Replaced with human-readable `dataset` column | `policy_eval_comparison.csv` |
| `route` column name | Renamed to `method` | `policy_eval_comparison.csv` |
| `achieved_avg_cost` and `dataset_key` column names | Renamed to `avg_cost`, `dataset` | `budget_curves_all_datasets.csv` |
| Oracle table had opaque column names (`accuracy`, `avg_cost`) | Prefixed as `oracle_acc`, `oracle_avg_cost`, `oracle_revise_rate` | `oracle_routing_eval.csv` |
| `oracle_gap` missing from policy eval table | Added as `oracle_acc − method_accuracy` per row | `policy_eval_comparison.csv` |
| `hard_gsm8k_b2` absent from final cross-regime summary | Added from upstream artifacts (policy eval + oracle + RTR JSONs) | `final_cross_regime_summary_fixed.csv` |
| AIME row with empty best-policy columns in cross-regime summary | Dropped from cleaned version | `final_cross_regime_summary_fixed.csv` |
| `reasoning_csv_used` housekeeping column in cross-regime table | Removed from cleaned version | `final_cross_regime_summary_fixed.csv` |
| Figure axis labels inconsistent (`Average cost` / `avg cost` etc.) | Standardised to `Average Cost` / `Accuracy` | All `_cleaned` PNGs |
| Figure legend used raw Python variable names | Replaced with friendly labels (`Reasoning only`, `Always revise`, etc.) | All scatter PNGs |
| Baseline strategy CSVs had per-dataset column with short internal name | Normalised to full `dataset` name; added `note` with sample size warning | `baselines_appendix.csv` |
| Row ordering within datasets was arbitrary | Sorted: baseline → adaptive v5/v6/v7 | `policy_eval_comparison.csv` |
| Dataset row ordering across tables was inconsistent | Sorted: hard_gsm8k_100 → hard_gsm8k_b2 → math500_100 → gsm8k_random_100 | All cleaned tables |
| No composite overview figure existed | Added 2×2 panel figure with shared legend | `accuracy_vs_cost_2x2_composite.png` |

---

## F. Remaining Minor Inconsistencies

| Issue | Status | Notes |
|-------|--------|-------|
| `main_results_summary.csv` values display as `0.9` not `0.900` | Cosmetic only — Python CSV writer drops trailing zeros; values are correct | Fix in LaTeX with `\num{0.900}` or format in pandas before printing |
| `cross_regime_summary.csv` (3-regime) does not include hard_gsm8k_b2 | Intentional — that table mirrors the original 3-regime source; use `final_cross_regime_summary_fixed.csv` for the 4-dataset version | — |
| Baselines in `baselines_appendix.csv` at n=30/15 — no 100-query version | Requires re-running `scripts/run_strong_baselines.py`; blocked by API cost | Explicitly noted as appendix-only; do not cite in §4 |
| `budget_curves_all_datasets.csv` contains target_avg_cost column (planned cost) that differs slightly from achieved avg_cost | Expected — budget curve by design reports both; the `avg_cost` column is the achieved value | Note in figure caption that x-axis shows achieved cost |
| Simulated MCKP figures absent | Blocked — sweep not run per instructions | Cite as future work or add as post-review supplement |
