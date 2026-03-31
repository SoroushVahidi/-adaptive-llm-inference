# Manuscript asset map — binary real-routing track

> OUTDATED / SUPERSEDED (2026-03-31): superseded by
> `FINAL_MANUSCRIPT_QUICKSTART.md` and `FINAL_MANUSCRIPT_ASSET_INDEX.md`.
> This file remains useful for background provenance only.

**Grounding:** Paths and numbers below refer to files present in this repository snapshot as of authoring. Export manifests are authoritative for **which paper exports succeeded** in the last non-strict run (`outputs/paper_tables/export_manifest.json`, `outputs/paper_figures/export_manifest.json`).

**Scope:** Binary real routing (reasoning vs revise, adaptive policies v5–v7, fixed baselines in policy eval). Multi-action is out of scope here.

---

## A. Usable tables right now (from `outputs/paper_tables/`)

Listed in `export_manifest.json` under `written` (8 data files + manifest counts as 9th written path in generator output).

| Path | Role |
|------|------|
| `outputs/paper_tables/real_routing/real_policy_eval_comparison_long.csv` | Long-form accuracy, `avg_cost`, `revise_rate` per route for GSM8K random 100, MATH500, hard GSM8K, hard GSM8K B2 (from policy eval summaries). |
| `outputs/paper_tables/cross_regime/final_cross_regime_summary.csv` | Per-dataset rollup: reasoning/revise/oracle accuracy, best policy name/cost/accuracy, `revise_helpful_rate`, RTR accuracy where enriched CSVs exist. |
| `outputs/paper_tables/cross_regime/cross_regime_summary.csv` | Regime comparison columns including adaptive v6/v7 accuracy and cost, `learned_router_viability` label. |
| `outputs/paper_tables/oracle_routing/oracle_routing_eval_summaries.csv` | Flattened oracle routing summaries (accuracy, cost, revise rate, n) with `source_file` pointer to JSON under `outputs/oracle_routing_eval/`. |
| `outputs/paper_tables/next_stage/next_stage_budget_curves_all_datasets.csv` | Merged budget-curve rows from `outputs/next_stage_eval/<key>/budget_curve.csv` with `dataset_key`. |
| `outputs/paper_tables/baselines/baselines_gsm8k_strategies.csv` | Rolled-up strategies from `outputs/baselines/gsm8k_baseline_summary.json`. |
| `outputs/paper_tables/baselines/baselines_hard_gsm8k_strategies.csv` | Same for `hard_gsm8k_baseline_summary.json`. |
| `outputs/paper_tables/baselines/baselines_math500_strategies.csv` | Same for `math500_baseline_summary.json`. |

**Blocked exporters (no table files for these keys in manifest):**

- `simulated_sweep` — missing `outputs/simulated_sweep/budget_sweep_comparisons.csv`
- `oracle_subset` — missing `outputs/oracle_subset_eval/summary.json`

---

## B. Usable figures right now (from `outputs/paper_figures/`)

Listed in `export_manifest.json` under `written` (12 PNG paths).

**Real routing — accuracy vs cost (one PNG per regime):**

| Path |
|------|
| `outputs/paper_figures/real_routing/real_policy_accuracy_vs_cost_real_policy_eval.png` |
| `outputs/paper_figures/real_routing/real_policy_accuracy_vs_cost_real_math500_policy_eval.png` |
| `outputs/paper_figures/real_routing/real_policy_accuracy_vs_cost_real_hard_gsm8k_policy_eval.png` |
| `outputs/paper_figures/real_routing/real_policy_accuracy_vs_cost_real_hard_gsm8k_b2_policy_eval.png` |

**Next-stage — budget and cascade curves (per dataset key):**

| Dataset keys (from filenames) |
|-------------------------------|
| `gsm8k_random100`, `hard_gsm8k_100`, `hard_gsm8k_b2`, `math500_100` |

Files: `outputs/paper_figures/next_stage/next_stage_budget_curve_<key>.png`, `next_stage_cascade_curve_<key>.png`.

**Blocked:** `simulated_sweep` figures (same missing sweep CSV as tables).

---

## C. Result claims already supported (binary track, conservative)

These are **supported by artifacts named above** plus the upstream JSON/CSVs they summarize. Wording should stay aligned with **n = 100** where the CSV row says 100, and should not over-claim significance.

| Claim shape | Supporting artifacts |
|-------------|----------------------|
| On fixed 100-query slices, **fixed routes** (reasoning-only vs always revise) and **adaptive policies** (v5–v7) have documented **accuracy and average cost proxy** in tabular form. | `real_policy_eval_comparison_long.csv`, `outputs/real_*_policy_eval/summary.json` |
| **Cross-regime:** reasoning vs revise accuracy, oracle upper bound, best listed policy, and revise-helpful prevalence differ by dataset row. | `final_cross_regime_summary.csv`, `cross_regime_summary.csv`, enriched `data/real_*_routing_dataset_enriched.csv` |
| **Oracle routing eval** reports per-slice accuracy/cost/revise_rate summaries consistent with next-stage tooling. | `oracle_routing_eval_summaries.csv`, `outputs/oracle_routing_eval/*_oracle_summary.json` |
| **Next-stage** budget and cascade behavior can be shown as curves for four dataset keys. | PNGs under `outputs/paper_figures/next_stage/`, merged `next_stage_budget_curves_all_datasets.csv`, `outputs/next_stage_eval/*/` |
| **Strong baselines** JSON was rolled into per-strategy CSVs (see limitations on sample size in section D). | `baselines_*_strategies.csv`, `outputs/baselines/*_baseline_summary.json` |

---

## D. Claims that should wait (stricter export or stronger runs)

| Topic | Why wait | Notes grounded in repo |
|-------|----------|-------------------------|
| **“Full paper export passed `--strict`”** | Simulated sweep + oracle subset inputs missing from manifest blockers. | `export_manifest.json` blockers |
| **Synthetic MCKP / allocation figures as exported paper assets** | No `outputs/simulated_sweep/` in successful export path yet. | Same blocker |
| **Oracle subset strategy table in `paper_tables/`** | Exporter blocked until `outputs/oracle_subset_eval/summary.json` exists. | Manifest; when present, `docs/MANUSCRIPT_RESULTS_READINESS.md` warns small-n issues for the default config |
| **“Strong baselines match main 100-query slices”** | Current `outputs/baselines/*_baseline_summary.json` uses **n_queries 30** (gsm8k, hard_gsm8k) and **15** (math500), not 100. | JSON files under `outputs/baselines/` |
| **AIME as full cross-regime row** | `final_cross_regime_summary.csv` has `aime2024` row but **empty** best-policy columns in this export. | CSV row 5 in current file |
| **“Learned router beats heuristics”** | Not supported as a success story; see `docs/MANUSCRIPT_RESULTS_READINESS.md` re `outputs/real_routing_model/summary.json`. | Separate from paper export; cite only if honestly framed |

---

## E. Exact commands to regenerate current assets

**Paper exports (non-strict — partial OK):**

```bash
python3 scripts/generate_paper_tables.py
python3 scripts/generate_paper_figures.py
```

**Strict export (after filling blockers):**

```bash
python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml
python3 scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml
python3 scripts/generate_paper_tables.py --strict
python3 scripts/generate_paper_figures.py --strict
```

**Upstream binary-routing artifacts (when re-running experiments, not required to re-read existing CSVs/JSON):** see `docs/PAPER_ARTIFACT_GENERATION_STATUS.md` and `docs/NEXT_REPO_RECOVERY_PLAN.md` for `run_build_*_routing_dataset.py`, `run_real_policy_eval.py`, `run_cross_regime_comparison.py`, `run_final_cross_regime_summary.py`, `run_next_stage_postprocess.py`.

---

## F. Manuscript sections ↔ repository artifacts

| Section | Artifacts (primary) | Supporting docs |
|---------|---------------------|-----------------|
| **Introduction / motivation** | `docs/PROJECT_CONTEXT.md` (problem, MCKP framing); optional high-level numbers from `final_cross_regime_summary.csv` (oracle vs fixed gaps where non-empty). | `docs/HARD_GSM8K_ROUTING_RESULTS.md`, `docs/MATH500_ROUTING_RESULTS.md` |
| **Method** | `src/data/build_real_routing_dataset.py`, `src/evaluation/real_policy_eval.py`; policy versions under `src/policies/`; CLI `scripts/run_real_policy_eval.py`. | `docs/REAL_ROUTING_DATASET.md`, `docs/ROUTING_DATASET.md` |
| **Datasets** | `data/real_gsm8k_routing_dataset*.csv`, `data/real_hard_gsm8k_routing_dataset*.csv`, `data/real_hard_gsm8k_b2_routing_dataset*.csv`, `data/real_math500_routing_dataset*.csv`, `data/real_aime2024_routing_dataset.csv`; build scripts `scripts/run_build_*_routing_dataset.py`. | `docs/FULL_PROJECT_STATE_AUDIT.md` (row counts, disagreement rates) |
| **Baselines** | `outputs/paper_tables/baselines/*.csv` ← `outputs/baselines/*_baseline_summary.json`; policy-eval fixed routes in `real_policy_eval_comparison_long.csv`. | `docs/STRONG_BASELINES_REPORT.md`, `docs/NEXT_STAGE_EXPERIMENT_RESULTS.md` |
| **Main results** | `real_policy_eval_comparison_long.csv`; four `real_policy_accuracy_vs_cost_*.png`; `final_cross_regime_summary.csv`. | `docs/MANUSCRIPT_RESULTS_READINESS.md` |
| **Ablations / cross-regime / next-stage** | `cross_regime_summary.csv`; `next_stage_budget_curves_all_datasets.csv`; `next_stage/*.png`; `oracle_routing_eval_summaries.csv`; raw `outputs/next_stage_eval/*`. | `docs/NEXT_STAGE_EXPERIMENT_RESULTS.md` |
| **Limitations** | Thin labels on GSM8K random 100; AIME row gaps; baseline n mismatch; learned-router metrics in `outputs/real_*_routing_model/summary.json`. | `docs/MANUSCRIPT_RESULTS_READINESS.md` §B–C |

---

## G. Recommended paper outline (from current evidence only)

**Title direction (examples, not prescriptions):** adaptive test-time routing between a single reasoning call and a revise path under a cost proxy; empirical study on curated math reasoning slices (GSM8K hard, MATH500) with a random GSM8K control.

**Main empirical narrative:** on **hard GSM8K (100)** and **MATH500 (100)**, pairwise outcomes show non-trivial structure (disagreement and revise-helpful rates documented in `docs/FULL_PROJECT_STATE_AUDIT.md`); **policy evaluation** (`real_policy_eval_comparison_long.csv` and scatter figures) compares fixed routes to adaptive policies on **accuracy vs average cost**. **Cross-regime** table summarizes oracle ceiling, best policy among those evaluated, and revise-helpful prevalence. **Next-stage** curves connect the same slices to budget/cascade analyses already exported.

**Centerpiece dataset / result:** **Hard GSM8K (100)** — strongest binary-routing signal in audited disagreement; full policy-eval row and dedicated figure (`real_hard_gsm8k_policy_eval`).

**Supporting only:** GSM8K random 100 (sparse revise signal); hard GSM8K B2 (parallel hard slice); MATH500; next-stage/oracle routing summaries; baseline rollups (with **n** caveat in section D).

**Omit or defer for now:** multi-action oracle track (`data/multi_action_routing_*.csv` absent); exported simulated-sweep tables/figures until sweep is run; oracle-subset table until eval exists; broad claims that “strong baselines” at current JSON **n** equal main 100-query experiments; AIME as a completed adaptive-policy row without new policy eval.

---

## H. Cross-reference

- Export inventory and strict vs partial: `docs/EXPERIMENT_ARTIFACT_STATUS.md`, `docs/PAPER_ARTIFACT_GENERATION_STATUS.md`
- Honest readiness and gaps: `docs/MANUSCRIPT_RESULTS_READINESS.md`
- Recovery steps: `docs/NEXT_REPO_RECOVERY_PLAN.md`
