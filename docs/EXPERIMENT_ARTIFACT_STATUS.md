# Experiment artifact status (paper pipeline)

This file records **repository-grounded** experiment outputs that feed:

- `python3 scripts/generate_paper_tables.py --strict`
- `python3 scripts/generate_paper_figures.py --strict`

Last verification: both commands exited **0** after the recovery pass documented below.

---

## Latest verification pass (structured report)

**A. Artifacts successfully generated in this pass**

This workspace already contained the upstream files from a prior recovery run. **No experiment scripts were re-executed** in this pass (nothing was missing). The only actions taken were: confirm files on disk, sanity-check oracle `summary.json`, reinstall dev deps, regenerate paper exports.

**B. Artifacts still missing**

None relative to `generate_paper_tables.py` / `generate_paper_figures.py` strict requirements in this environment.

**C. Blocker per missing artifact**

Not applicable here. On a fresh clone, see the “Failed / blocked artifacts” table below.

**D. Commands to generate each missing artifact later**

See “Minimal reproduction (full paper artifact chain)” and `docs/PAPER_ARTIFACT_GENERATION_STATUS.md`.

**E. Paper tables fully generatable with `--strict`?**

**Yes** — `python3 scripts/generate_paper_tables.py --strict` exited **0** (13 paths under `outputs/paper_tables/`, including `export_manifest.json`).

**F. Paper figures fully generatable with `--strict`?**

**Yes** — `python3 scripts/generate_paper_figures.py --strict` exited **0** (16 paths under `outputs/paper_figures/`, including `export_manifest.json`).

**Note on `outputs/recent_baselines/`:** the paper exporters use **`outputs/baselines/*_baseline_summary.json`** (`run_strong_baselines.py`), not `outputs/recent_baselines/`. To populate recent baselines separately, use `scripts/run_recent_baselines_experiment.py` (optional for the current export scripts).

---

## Required inputs → producing scripts

| Artifact path(s) | Producer | Upstream dependencies |
|------------------|----------|------------------------|
| `outputs/simulated_sweep/budget_sweep_comparisons.csv`, `noise_sensitivity_comparisons.csv`, `simulated_sweep_results.json` | `python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml` | None (synthetic) |
| `outputs/baselines/*_baseline_summary.json` | `python3 scripts/run_strong_baselines.py --config configs/strong_baselines_dummy.yaml` (or `_real.yaml` + API) | Datasets per config (dummy uses local paths) |
| `outputs/cross_regime_comparison/cross_regime_summary.{csv,json}` | `python3 scripts/run_cross_regime_comparison.py` | Build + policy + router summaries under `outputs/` (see script) |
| `outputs/cross_regime_comparison/final_cross_regime_summary.csv` | `python3 scripts/run_final_cross_regime_summary.py` | `data/real_*_routing_dataset*.csv` + policy `summary.json` files |
| `outputs/oracle_routing_eval/*_oracle_summary.json`, `outputs/next_stage_eval/<key>/budget_curve.csv`, `cascade_curve.csv` | `python3 scripts/run_next_stage_postprocess.py` (per dataset key) | Routing CSV + optional policy summary |
| `outputs/oracle_subset_eval/summary.json` (+ `summary.csv`, matrices) | `python3 scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml` | **OPENAI_API_KEY**, network, bundled GSM8K sample in config |
| `outputs/real_*_policy_eval/summary.json` | `python3 scripts/run_real_policy_eval.py` | Prior real routing dataset build + API costs |

**Note:** The paper exporters read **`outputs/baselines/`**, not `outputs/recent_baselines/`. The status doc `PAPER_ARTIFACT_GENERATION_STATUS.md` table refers to strong baselines under `outputs/baselines/`.

---

## Completed artifacts (this environment)

After the recovery pass, the following exist and match exporter expectations:

| Area | Paths |
|------|--------|
| Simulated sweep | `outputs/simulated_sweep/budget_sweep_comparisons.csv`, `noise_sensitivity_comparisons.csv`, `budget_sweep_runs.csv`, `noise_sensitivity_runs.csv`, `simulated_sweep_results.json` |
| Strong baselines | `outputs/baselines/*_baseline_summary.json` plus additional ladder/router files from dummy run |
| Cross-regime | `outputs/cross_regime_comparison/cross_regime_summary.csv`, `cross_regime_summary.json`, `final_cross_regime_summary.csv` |
| Oracle routing + next-stage | `outputs/oracle_routing_eval/*_oracle_summary.json`, `outputs/next_stage_eval/*/budget_curve.csv`, `outputs/next_stage_eval/*/cascade_curve.csv`, `outputs/budget_sweep/*_budget_curve.csv` |
| Real policy summaries | `outputs/real_policy_eval/summary.json`, `outputs/real_math500_policy_eval/summary.json`, `outputs/real_hard_gsm8k_policy_eval/summary.json`, `outputs/real_hard_gsm8k_b2_policy_eval/summary.json` |
| Oracle subset | `outputs/oracle_subset_eval/summary.json`, `summary.csv`, `per_query_matrix.csv`, `oracle_assignments.csv`, `pairwise_win_matrix.csv` |
| Paper exports | `outputs/paper_tables/**`, `outputs/paper_figures/**` (from `generate_paper_*` scripts) |

Oracle subset run also refreshed:

- `docs/EXPERIMENT_LOG_ORACLE_SUBSET.md`
- `docs/RESULTS_ORACLE_SUBSET.md`

---

## Failed / blocked artifacts

**None** for the current `generate_paper_tables.py` / `generate_paper_figures.py` strict runs in this workspace after recovery.

If a clone is missing pieces, typical blockers:

| Blocker | Requirement | Unblock command |
|---------|-------------|-----------------|
| Simulated sweep missing | No API | `python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml` |
| Baselines missing | Dummy OK | `python3 scripts/run_strong_baselines.py --config configs/strong_baselines_dummy.yaml` |
| Oracle subset missing / BLOCKED | `OPENAI_API_KEY`, internet | `export OPENAI_API_KEY=...` then `python3 scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml` |
| Cross-regime missing | Prior real-stack outputs | `python3 scripts/run_cross_regime_comparison.py` |
| Final cross-regime missing | Routing CSVs on disk | `python3 scripts/run_final_cross_regime_summary.py` |
| Figures blocked | matplotlib | `pip install -e ".[dev]"` (includes matplotlib) |

---

## Minimal reproduction (full paper artifact chain)

From repo root, after `pip install -e ".[dev]"`:

```bash
# A — Synthetic (no API)
python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml

# B — Baselines (dummy backend)
python3 scripts/run_strong_baselines.py --config configs/strong_baselines_dummy.yaml

# C — Real stack (only if rebuilding routing/policy; large / API-heavy)
# python3 scripts/run_build_real_routing_dataset.py
# python3 scripts/run_real_policy_eval.py --output-dir outputs/real_policy_eval
# … (see docs/REAL_GSM8K_ROUTING_STUDY.md and related)

python3 scripts/run_cross_regime_comparison.py
python3 scripts/run_final_cross_regime_summary.py

# D — Oracle subset (requires OPENAI_API_KEY)
python3 scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml

# E — Next-stage curves (per dataset; example)
python3 scripts/run_next_stage_postprocess.py \
  --dataset-key gsm8k_random100 \
  --routing-csv data/real_gsm8k_routing_dataset.csv \
  --policy-summary-json outputs/real_policy_eval/summary.json
# Repeat for other keys as in docs/PAPER_ARTIFACT_GENERATION_STATUS.md

# Paper exports
python3 scripts/generate_paper_tables.py --strict
python3 scripts/generate_paper_figures.py --strict
```

---

## Validation checklist

After generating artifacts:

1. `outputs/simulated_sweep/budget_sweep_comparisons.csv` and `noise_sensitivity_comparisons.csv` exist and parse as CSV.
2. `outputs/oracle_subset_eval/summary.json` exists and does **not** contain `"run_status": "BLOCKED"`.
3. `python3 scripts/generate_paper_tables.py --strict` → exit 0.
4. `python3 scripts/generate_paper_figures.py --strict` → exit 0.
