# Paper Export Run Status

**Run date:** 2026-03-30
**Commands executed (from repo root):**

```bash
python3 scripts/generate_paper_tables.py
python3 scripts/generate_paper_figures.py
```

Both scripts exited with code 0 (partial-export mode; no `--strict` flag).
matplotlib 3.10.8 was used for figure rendering.

---

## A. Generated Files

### Tables (`outputs/paper_tables/`)

| File | Contents | Source artifacts |
|------|----------|-----------------|
| `baselines/baselines_gsm8k_strategies.csv` | Strategy accuracy/cost rollup (n=30) | `outputs/baselines/gsm8k_baseline_summary.json` |
| `baselines/baselines_hard_gsm8k_strategies.csv` | Strategy accuracy/cost rollup (n=30) | `outputs/baselines/hard_gsm8k_baseline_summary.json` |
| `baselines/baselines_math500_strategies.csv` | Strategy accuracy/cost rollup (n=15) | `outputs/baselines/math500_baseline_summary.json` |
| `cross_regime/cross_regime_summary.csv` | 3-regime rollup (adaptive v6/v7 accuracy, cost, learned_router_viability) | `outputs/cross_regime_comparison/cross_regime_summary.csv` |
| `cross_regime/final_cross_regime_summary.csv` | 4-dataset rollup incl. oracle, RTR, best policy (AIME row has empty policy columns) | `outputs/cross_regime_comparison/final_cross_regime_summary.csv` |
| `oracle_routing/oracle_routing_eval_summaries.csv` | Oracle upper-bound accuracy/cost/revise_rate for 4 dataset keys | `outputs/oracle_routing_eval/*_oracle_summary.json` |
| `real_routing/real_policy_eval_comparison_long.csv` | Long-form accuracy/avg_cost/revise_rate per route × 4 regimes (n=100 each) | `outputs/real_*_policy_eval/summary.json` |
| `next_stage/next_stage_budget_curves_all_datasets.csv` | Merged budget curves (all 4 dataset keys) | `outputs/next_stage_eval/*/budget_curve.csv` |
| `export_manifest.json` | Machine-readable list of written paths and blockers | — |

**Total: 8 data files + 1 manifest = 9 paths written.**

### Figures (`outputs/paper_figures/`)

**Next-stage budget curves (4 PNGs):**
- `next_stage/next_stage_budget_curve_gsm8k_random100.png`
- `next_stage/next_stage_budget_curve_hard_gsm8k_100.png`
- `next_stage/next_stage_budget_curve_hard_gsm8k_b2.png`
- `next_stage/next_stage_budget_curve_math500_100.png`

**Next-stage cascade curves (4 PNGs):**
- `next_stage/next_stage_cascade_curve_gsm8k_random100.png`
- `next_stage/next_stage_cascade_curve_hard_gsm8k_100.png`
- `next_stage/next_stage_cascade_curve_hard_gsm8k_b2.png`
- `next_stage/next_stage_cascade_curve_math500_100.png`

**Real policy accuracy-vs-cost scatter (4 PNGs, one per regime):**
- `real_routing/real_policy_accuracy_vs_cost_real_policy_eval.png`
- `real_routing/real_policy_accuracy_vs_cost_real_math500_policy_eval.png`
- `real_routing/real_policy_accuracy_vs_cost_real_hard_gsm8k_policy_eval.png`
- `real_routing/real_policy_accuracy_vs_cost_real_hard_gsm8k_b2_policy_eval.png`

**Total: 12 PNGs + 1 manifest = 13 paths written.**

---

## B. Remaining Blocked Tables / Figures

### Tables (2 blocked exporters)

| Exporter | Missing input | How to unblock |
|----------|--------------|---------------|
| `simulated_sweep` | `outputs/simulated_sweep/budget_sweep_comparisons.csv` (and `noise_sensitivity_comparisons.csv`) | `python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml` |
| `oracle_subset` | `outputs/oracle_subset_eval/summary.json` | `python3 scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml` |

### Figures (1 blocked exporter)

| Exporter | Missing input | How to unblock |
|----------|--------------|---------------|
| `simulated_sweep` | `outputs/simulated_sweep/budget_sweep_comparisons.csv` | Same simulated sweep run as above |

---

## C. Correspondence with Available Result Artifacts

| Result artifact family | Paper export produced | Notes |
|------------------------|----------------------|-------|
| `outputs/real_*_policy_eval/` | ✅ `real_routing/real_policy_eval_comparison_long.csv` + 4 scatter PNGs | All 4 regimes (n=100 each); comparison rows fully exported |
| `outputs/cross_regime_comparison/` | ✅ `cross_regime/cross_regime_summary.csv` + `final_cross_regime_summary.csv` | AIME row in final summary has empty `best_policy_*` columns — noted in inventory |
| `outputs/oracle_routing_eval/` | ✅ `oracle_routing/oracle_routing_eval_summaries.csv` | All 4 dataset keys present |
| `outputs/next_stage_eval/` | ✅ `next_stage/next_stage_budget_curves_all_datasets.csv` + 4 budget PNGs + 4 cascade PNGs | All 4 dataset keys present |
| `outputs/baselines/` | ✅ 3 strategy CSVs | ⚠️ n=15–30 only; not aligned with 100-query policy-eval slices |
| `outputs/simulated_sweep/` | ⛔ Blocked | Sweep not yet run |
| `outputs/oracle_subset_eval/` | ⛔ Blocked | Subset eval not yet run |
