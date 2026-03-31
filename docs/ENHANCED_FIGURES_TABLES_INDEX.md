# Enhanced Figures and Tables Index

This document inventories every artifact created by
`scripts/generate_enhanced_artifacts.py` and the supporting module
`src/paper_artifacts/exports_enhanced.py`.

All numbers come directly from committed evaluation artifacts.
No results were invented or extrapolated.

---

## Tables

### A1 – Oracle-Gap Table  *(main-paper candidate)*

| Field | Value |
|---|---|
| **File** | `outputs/paper_tables_enhanced/oracle_gap_table.csv` |
| **Location class** | Main paper |
| **Columns** | `regime`, `cheap_acc`, `revise_acc`, `best_adaptive_acc`, `best_adaptive_policy`, `oracle_acc`, `oracle_gap`, `revise_helpful_rate` |
| **Source files** | `outputs/real_*_policy_eval/summary.json`, `outputs/oracle_routing_eval/*.json` |
| **Purpose** | Shows how much recoverable routing headroom exists per regime and how much the best policy leaves on the table relative to the oracle upper bound. |
| **Caveats** | `oracle_gap` is `oracle_acc − best_adaptive_acc`; "best adaptive" is the v5/v6/v7 with highest accuracy. |

**Quick view**

| Regime | Cheap | Best Adaptive | Oracle | Oracle Gap | Revise-Helpful |
|---|---|---|---|---|---|
| GSM8K-Rand | 0.90 | 0.92 (v5) | 0.92 | 0.00 | 2% |
| MATH-500 | 0.64 | 0.66 (v5) | 0.70 | 0.04 | 6% |
| Hard-GSM8K-B2 | 0.83 | 0.91 (v5) | 0.92 | 0.01 | 9% |
| Hard-GSM8K | 0.79 | 0.86 (v5) | 0.91 | 0.05 | 12% |

---

### A2 – Cost-Efficiency Gain Table  *(main-paper candidate)*

| Field | Value |
|---|---|
| **File** | `outputs/paper_tables_enhanced/cost_efficiency_gain_table.csv` |
| **Location class** | Main paper |
| **Columns** | `regime`, `best_adaptive_policy`, `best_adaptive_acc_gain`, `best_adaptive_avg_cost`, `cost_increase_over_cheap`, `acc_per_unit_cost`, `matches_always_revise` |
| **Source files** | `outputs/real_*_policy_eval/summary.json` |
| **Purpose** | Makes the deployment story explicit: accuracy gain, cost overhead, and efficiency ratio. |
| **Caveats** | Cost baseline is 1× (cheap greedy); `acc_per_unit_cost = acc_gain / cost_increase`. When `cost_increase = 0`, ratio is undefined (NaN). |

---

### A3 – Policy Ranking Table  *(main-paper candidate)*

| Field | Value |
|---|---|
| **File** | `outputs/paper_tables_enhanced/policy_ranking_table.csv` |
| **Location class** | Main paper |
| **Columns** | `regime`, `policy`, `accuracy`, `avg_cost`, `revise_rate` |
| **Source files** | `outputs/real_*_policy_eval/summary.json`, `outputs/oracle_routing_eval/*.json`, `outputs/baselines/confidence_threshold/confidence_threshold_summary.json` |
| **Purpose** | Single comprehensive comparison across all policies and regimes, sorted by accuracy descending then cost ascending. Includes oracle and confidence-threshold baseline. |
| **Caveats** | Confidence-threshold uses best operating-point per regime (from `confidence_threshold_summary.json`). |

---

### A4 – AIME Supplementary Table  *(supplementary/appendix)*

| Field | Value |
|---|---|
| **File** | `outputs/paper_tables_enhanced/aime_supplementary_table.csv` |
| **Location class** | Supplementary / Appendix |
| **Columns** | `policy`, `accuracy`, `avg_cost`, `revise_rate`, `notes` |
| **Source files** | `outputs/small_pass/aime_policy_comparison.csv`, `outputs/small_pass/aime_summary.json` |
| **Purpose** | Shows the AIME-2024 limit case where revise-helpful rate = 0% and all routing policies degenerate to the same accuracy as the cheap baseline. |
| **Caveats** | n = 30 (small sample). This is the "routing degenerates" limit case, suitable only for appendix discussion — do not use as a main-paper result. |

---

## Figures

### B1 – Oracle-Gap Bar Chart  *(main-paper candidate)*

| Field | Value |
|---|---|
| **File** | `outputs/paper_figures_enhanced/oracle_gap_bar_chart.png` / `.pdf` |
| **Location class** | Main paper |
| **What it shows** | Grouped bar chart with three bars per regime: Cheap (greedy), Best Adaptive, Oracle. Oracle-gap annotations show the Δ between oracle and best adaptive. |
| **Source files** | `outputs/real_*_policy_eval/summary.json`, `outputs/oracle_routing_eval/*.json` |
| **Purpose** | Visually communicates remaining recoverable headroom by regime. |
| **Caveats** | Best adaptive is the single best v5/v6/v7 per regime (not a tuned combination). |

---

### B2 – Revise-Helpful vs. Best-Policy Gain Scatter  *(main-paper candidate)*

| Field | Value |
|---|---|
| **File** | `outputs/paper_figures_enhanced/revise_helpful_vs_gain_scatter.png` / `.pdf` |
| **Location class** | Main paper |
| **What it shows** | Scatter plot (4 points = 4 regimes): x = revise-helpful rate, y = best adaptive accuracy gain over cheap baseline. Linear trend line added. |
| **Source files** | `outputs/real_*_policy_eval/summary.json` |
| **Purpose** | Supports the manuscript's central claim that routing value depends on workload structure (higher revise-helpful rate → more gain from routing). |
| **Caveats** | Only 4 data points; trend line is illustrative, not statistically validated. |

---

### B3 – Cost–Accuracy Pareto Plot  *(main-paper candidate)*

| Field | Value |
|---|---|
| **File** | `outputs/paper_figures_enhanced/cost_accuracy_pareto.png` / `.pdf` |
| **Location class** | Main paper |
| **What it shows** | 4-panel figure (one per regime), each showing a scatter of cost vs. accuracy for all policies: cheap, always-revise, adaptive v5/v6/v7, confidence-threshold, oracle. |
| **Source files** | `outputs/real_*_policy_eval/summary.json`, `outputs/oracle_routing_eval/*.json`, `outputs/baselines/confidence_threshold/confidence_threshold_summary.json` |
| **Purpose** | Strengthens the efficiency interpretation — shows which policies achieve accuracy gains without excessive cost. |
| **Caveats** | Oracle is plotted at its measured cost (not cost=1); this is the minimum achievable cost for perfect routing. |

---

### B4 – Policy Revise-Rate Comparison  *(supplementary)*

| Field | Value |
|---|---|
| **File** | `outputs/paper_figures_enhanced/policy_revise_rate_comparison.png` / `.pdf` |
| **Location class** | Supplementary |
| **What it shows** | Grouped bar chart showing the revise rate for each adaptive policy (v5/v6/v7) and confidence-threshold baseline per regime. The true revise-helpful rate is shown as a horizontal marker. |
| **Source files** | `outputs/real_*_policy_eval/summary.json`, `outputs/baselines/confidence_threshold/confidence_threshold_summary.json` |
| **Purpose** | Makes operational behaviour easy to understand — shows which policies over-escalate or under-escalate relative to the oracle rate. |
| **Caveats** | The revise-helpful rate marker is the oracle ideal; policies deviating from it either waste compute or miss recoverable cases. |

---

### B5 – Confidence Baseline Comparison  *(supplementary/appendix)*

| Field | Value |
|---|---|
| **File** | `outputs/paper_figures_enhanced/confidence_baseline_comparison.png` / `.pdf` |
| **Location class** | Supplementary |
| **What it shows** | Grouped bar chart: Cheap vs. Always-Revise vs. Best Adaptive vs. Confidence-Threshold per regime. |
| **Source files** | `outputs/real_*_policy_eval/summary.json`, `outputs/baselines/confidence_threshold/confidence_threshold_summary.json` |
| **Purpose** | Shows that confidence-threshold routing is a strong and interpretable baseline that performs comparably to adaptive policies in some regimes. |
| **Caveats** | Confidence-threshold uses its optimal operating point per regime (threshold chosen ex-post); in practice threshold selection requires validation data. |

---

### B6 – AIME Limit-Case Figure  *(appendix)*

| Field | Value |
|---|---|
| **File** | `outputs/paper_figures_enhanced/aime_limit_case.png` / `.pdf` |
| **Location class** | Appendix |
| **What it shows** | Two panels: (1) bar chart of accuracy for all policies on AIME-2024, (2) cost vs. accuracy scatter for AIME-2024. All policies converge to the same accuracy (13.3%) because revise-helpful rate = 0%. |
| **Source files** | `outputs/small_pass/aime_policy_comparison.csv`, `outputs/small_pass/aime_summary.json` |
| **Purpose** | Demonstrates the theoretical limit case where adaptive routing provides no benefit. |
| **Caveats** | n = 30; all policies achieve 13.3% (4/30 correct). Even the oracle cannot improve because revision never helps on AIME-2024 with this model. |

---

## Graphic Abstract

### C1 – Graphic Abstract  *(main paper, journal submission)*

| Field | Value |
|---|---|
| **Files** | `outputs/graphic_abstract/graphic_abstract.png`, `outputs/graphic_abstract/graphic_abstract.pdf` |
| **Location class** | Main paper (journal front matter) |
| **Notes** | `outputs/graphic_abstract/graphic_abstract_notes.md` |
| **What it shows** | 4-panel figure: (1) Query → cheap reasoning, (2) Route decision node, (3) Two paths (keep vs revise), (4) Key grounded findings. |
| **Source files** | `outputs/real_*_policy_eval/summary.json`, `outputs/oracle_routing_eval/*.json`, `outputs/small_pass/aime_summary.json` |
| **Purpose** | Journal-ready one-page visual summary for Knowledge-Based Systems submission. |
| **Caveats** | Mini-bars in Panel 3 use rounded accuracy figures from paper tables. All key numbers are directly from committed evaluation artifacts. |

---

## Artifacts Not Created (Honesty Report)

The following requested artifacts were **not created** because the data is
insufficient or would require unsupported extrapolation:

| Artifact | Reason |
|---|---|
| GPQA-Diamond supplementary table | `real_gpqa_diamond_routing_dataset_enriched.csv` exists but no `outputs/real_gpqa_*_policy_eval/` summary JSON was found; policy evaluation was not run on GPQA at full scale. Creating a table from incomplete data would be misleading. |
| Hard-GSM8K-B2 dedicated baseline table | The `outputs/baselines/` directory contains GSM8K, Hard-GSM8K, and MATH-500 baseline summaries (n=15–30) but no Hard-GSM8K-B2 baseline. A separate table would be incomplete. |

---

## Generation Script

```bash
# Generate all enhanced artifacts:
python3 scripts/generate_enhanced_artifacts.py --repo-root .

# Generate only tables:
python3 scripts/generate_enhanced_artifacts.py --only tables

# Generate only figures and graphic abstract:
python3 scripts/generate_enhanced_artifacts.py --only figures,graphic_abstract
```

## Source Module

`src/paper_artifacts/exports_enhanced.py`

Contains all table-export and figure-generation functions, each documented
with their source inputs, output columns, and honesty notes.

---

## Recommendation Summary

### Recommend for Main Paper
1. **B2 – Revise-Helpful vs. Best-Policy Gain Scatter** — directly visualises the paper's central claim
2. **A1 – Oracle-Gap Table** — provides quantitative headroom analysis in a single table
3. **B3 – Cost–Accuracy Pareto Plot** — comprehensive 4-regime efficiency comparison

### Recommend as Supplementary
- A3 – Policy Ranking Table (comprehensive, useful for reviewers)
- B4 – Policy Revise-Rate Comparison (operational insight)
- B5 – Confidence Baseline Comparison (validates confidence-threshold baseline)

### Recommend as Appendix Only
- A4 – AIME Supplementary Table (limit case, small n=30)
- B6 – AIME Limit-Case Figure (limit case degeneration)

### Main Paper Front Matter
- C1 – Graphic Abstract
