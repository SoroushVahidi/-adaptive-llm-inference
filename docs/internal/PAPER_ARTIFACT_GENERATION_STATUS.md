# Paper artifact generation status

This document tracks what manuscript tables and figures the repository can **materialize from existing outputs** using:

- `scripts/generate_paper_tables.py`
- `scripts/generate_paper_figures.py`

These tools **never invent** metrics: they only read artifacts produced by experiment scripts. If inputs are missing or a run is `BLOCKED`, the exporter records a precise blocker message and (unless `--strict`) continues with other exports.

Default output locations:

- Tables: `outputs/paper_tables/`
- Figures: `outputs/paper_figures/` (requires **matplotlib**)

Each run also writes `export_manifest.json` in the output directory listing paths written and any blockers.

---

## Artifact formats this repo actually uses

| Pipeline | Typical outputs | Produced by |
|----------|-----------------|-------------|
| Simulated allocation sweep | `budget_sweep_comparisons.csv`, `noise_sensitivity_comparisons.csv`, raw runs JSON/CSV | `scripts/run_simulated_sweep.py` |
| Simulated allocation (single budget) | `outputs/simulated_allocation_results.json` (or config `output`) | `scripts/run_simulated_allocation.py` |
| Oracle subset (GSM8K strategies) | `summary.json`, `summary.csv`, `per_query_matrix.csv`, `oracle_assignments.csv` | `scripts/run_oracle_subset_eval.py` |
| Real routing dataset build | `*_run_summary.json`, per-query CSV, `provider_metadata.json` | `scripts/run_build_*_routing_dataset.py` |
| Real policy eval | `summary.json` with `comparison` list (accuracy, avg_cost, revise_rate) | `scripts/run_real_policy_eval.py` |
| Learned routing model | `summary.json`, `routing_simulation.csv`, metrics CSVs | `scripts/run_real_routing_model_eval.py` |
| Strong baselines | `*_baseline_summary.json`, optional `final_baseline_summary.csv`, ladder JSON | `scripts/run_strong_baselines.py` |
| Cross-regime rollup | `cross_regime_summary.csv` / `.json`, optional `final_cross_regime_summary.csv` | `scripts/run_cross_regime_comparison.py`, `scripts/run_final_cross_regime_summary.py` |
| Next-stage / oracle routing | `outputs/oracle_routing_eval/*_oracle_summary.json`, `outputs/next_stage_eval/<dataset>/budget_curve.csv`, `cascade_curve.csv`, `outputs/budget_sweep/*_budget_curve.csv` | `scripts/run_next_stage_postprocess.py` |

---

## Tables: supported vs blocked

### Already generatable (when inputs exist)

| Paper-oriented name | Exporter key | Output path (under `outputs/paper_tables/`) | Required inputs |
|---------------------|--------------|---------------------------------------------|-----------------|
| Simulated sweep budget summary | `simulated_sweep` | `simulated_sweep/simulated_sweep_budget_summary.csv` | `outputs/simulated_sweep/budget_sweep_comparisons.csv` |
| Simulated sweep noise summary | `simulated_sweep` | `simulated_sweep/simulated_sweep_noise_summary.csv` | `outputs/simulated_sweep/noise_sensitivity_comparisons.csv` |
| Baseline strategy rollup | `baselines` | `baselines/baselines_<dataset>_strategies.csv` | `outputs/baselines/*_baseline_summary.json` |
| Cross-regime summary | `cross_regime` | `cross_regime/cross_regime_summary.csv` | `outputs/cross_regime_comparison/cross_regime_summary.csv` **or** `.json` |
| Final cross-regime summary | `final_cross_regime` | `cross_regime/final_cross_regime_summary.csv` | `outputs/cross_regime_comparison/final_cross_regime_summary.csv` |
| Oracle routing eval (merged) | `oracle_routing` | `oracle_routing/oracle_routing_eval_summaries.csv` | `outputs/oracle_routing_eval/*_oracle_summary.json` |
| Oracle subset | `oracle_subset` | `oracle_subset/oracle_subset_strategy_accuracy.csv` (and/or headline CSV) | `outputs/oracle_subset_eval/summary.json` with `run_status` ≠ `BLOCKED`; optional `summary.csv` |
| Real policy comparison (long) | `real_policy_comparison` | `real_routing/real_policy_eval_comparison_long.csv` | At least one of: `outputs/real_policy_eval/summary.json`, `outputs/real_math500_policy_eval/summary.json`, `outputs/real_hard_gsm8k_policy_eval/summary.json`, `outputs/real_hard_gsm8k_b2_policy_eval/summary.json` |
| Next-stage budget curves (merged) | `next_stage_budget_curves` | `next_stage/next_stage_budget_curves_all_datasets.csv` | `outputs/next_stage_eval/*/budget_curve.csv` |

### Blocked in a fresh clone (typical)

| Artifact | Why blocked | Exact inputs needed |
|----------|-------------|---------------------|
| Simulated sweep tables/figures | `outputs/simulated_sweep/` not populated | `outputs/simulated_sweep/budget_sweep_comparisons.csv`, `noise_sensitivity_comparisons.csv` |
| Baseline rollup CSV mirror | No `outputs/baselines/` or no `*_baseline_summary.json` | Files from `scripts/run_strong_baselines.py` |
| Final cross-regime CSV | `run_final_cross_regime_summary.py` not run | `data/real_*_routing_dataset*.csv` + policy summaries; produces `outputs/cross_regime_comparison/final_cross_regime_summary.csv` |
| Oracle subset exports | Subset eval never completed successfully | `outputs/oracle_subset_eval/summary.json` (and ideally `summary.csv`) from `scripts/run_oracle_subset_eval.py` |
| Strong baselines `final_baseline_summary.csv` | Not generated by the paper export script (only JSON rollups today) | Run `run_strong_baselines.py`; use or copy `outputs/baselines/final_baseline_summary.csv` manually if needed for the paper |

### Not covered by export scripts (by design / different format)

| Content | Notes |
|---------|--------|
| Single-shot `outputs/simulated_allocation_results.json` | Different layout than sweep; add a dedicated exporter if the paper needs it. |
| `scripts/run_experiment.py` JSON | Schema is per-config; not standardized for paper tables. |
| Per-query routing CSVs | Large; point readers at `outputs/real_routing_dataset/` etc. |
| `outputs/real_routing_model/routing_simulation.csv` | Could add a small summary exporter later. |

---

## Figures: supported vs blocked

| Paper-oriented name | Exporter key | Output (under `outputs/paper_figures/`) | Required inputs |
|---------------------|--------------|----------------------------------------|-----------------|
| Simulated sweep (budget + noise) | `simulated_sweep` | `simulated_sweep/*.png` | Same CSV pair as simulated sweep tables |
| Next-stage budget curves | `next_stage_budget` | `next_stage/next_stage_budget_curve_<dataset_key>.png` | `outputs/next_stage_eval/<dataset_key>/budget_curve.csv` |
| Next-stage cascade curves | `next_stage_cascade` | `next_stage/next_stage_cascade_curve_<dataset_key>.png` | `outputs/next_stage_eval/<dataset_key>/cascade_curve.csv` |
| Real policy accuracy vs cost | `real_policy` | `real_routing/real_policy_accuracy_vs_cost_<slug>.png` | Policy `summary.json` files listed in the table section above |

**matplotlib** is included in the `dev` optional dependency set (`pip install -e ".[dev]"`) or can be installed alone (`pip install matplotlib`).

---

## Commands to regenerate everything (once artifacts exist)

From the repository root:

```bash
# 1) Simulated sweep + summaries (feeds sweep tables/figures)
python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml
python3 scripts/summarize_simulated_results.py --input-dir outputs/simulated_sweep

# 2) Oracle subset (optional; needs API / model access as per script)
python3 scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml

# 3) Strong baselines (dummy or real config)
python3 scripts/run_strong_baselines.py --config configs/strong_baselines_dummy.yaml

# 4) Real routing stack (example pieces; adjust configs/paths as in docs)
python3 scripts/run_build_real_routing_dataset.py
python3 scripts/run_real_policy_eval.py --output-dir outputs/real_policy_eval
python3 scripts/run_cross_regime_comparison.py
python3 scripts/run_final_cross_regime_summary.py

# 5) Next-stage curves (per dataset; example pattern)
python3 scripts/run_next_stage_postprocess.py \
  --dataset-key gsm8k_random100 \
  --routing-csv data/real_gsm8k_routing_dataset.csv \
  --policy-summary-json outputs/real_policy_eval/summary.json
# Repeat for other dataset keys / CSVs as in docs/NEXT_STAGE_EXPERIMENT_RESULTS.md

# 6) Paper exports
python3 scripts/generate_paper_tables.py
python3 scripts/generate_paper_figures.py
```

Use `--strict` if you want a non-zero exit when any subsection is blocked:

```bash
python3 scripts/generate_paper_tables.py --strict
python3 scripts/generate_paper_figures.py --strict
```

Export only selected pieces:

```bash
python3 scripts/generate_paper_tables.py --only baselines cross_regime real_policy_comparison
python3 scripts/generate_paper_figures.py --only next_stage_budget real_policy
```

---

## Validation

- `ruff check src/paper_artifacts/ scripts/generate_paper_tables.py scripts/generate_paper_figures.py`
- `pytest` (project test suite; no new tests required for this infrastructure unless you add them)
