# Paper Artifact Status

> OUTDATED / SUPERSEDED (2026-03-31): The canonical final manuscript assets are now
> generated into `outputs/paper_tables_final/` and `outputs/paper_figures_final/`
> via `scripts/generate_final_manuscript_artifacts.py`.
> See `docs/CANONICAL_MANUSCRIPT_DECISIONS.md` and `docs/FINAL_CONSISTENCY_AUDIT.md`.

This file explicitly lists every major artifact in the repository and its
role relative to the manuscript.  Use this to verify which outputs back
specific paper claims.

**Legend:**
- ✅ **Main paper** — cited or cited-eligible in the main body
- 📎 **Appendix** — appendix-only; supporting detail
- 🔬 **Exploratory** — informative but not part of core claims
- ❌ **Incomplete** — blocked, partial, or not run; must not be cited as final

---

## A. Data Files

| File | n | Status | Notes |
|------|---|--------|-------|
| `data/real_gsm8k_routing_dataset_enriched.csv` | 100 | ✅ Main paper | Primary GSM8K regime |
| `data/real_hard_gsm8k_routing_dataset_enriched.csv` | 100 | ✅ Main paper | Hard-GSM8K regime |
| `data/real_hard_gsm8k_b2_routing_dataset_enriched.csv` | 100 | ✅ Main paper | Hard-GSM8K replication |
| `data/real_math500_routing_dataset_enriched.csv` | 100 | ✅ Main paper | MATH500 regime |
| `data/real_aime2024_routing_dataset.csv` | 30 | 🔬 Exploratory | Policy eval not run; omit from main tables |
| `data/gpqa_diamond_normalized.jsonl` | — | 🔬 Exploratory | No policy eval committed |
| `data/consistency_benchmark.json` | — | 🔬 Exploratory | Diagnostic dataset |
| `data/real_gsm8k_routing_dataset.csv` | 100 | ✅ Main paper | Pre-enrichment (base version) |
| `data/real_hard_gsm8k_routing_dataset.csv` | 100 | ✅ Main paper | Pre-enrichment (base version) |
| `data/real_hard_gsm8k_b2_routing_dataset.csv` | 100 | ✅ Main paper | Pre-enrichment (base version) |
| `data/real_math500_routing_dataset.csv` | 100 | ✅ Main paper | Pre-enrichment (base version) |

---

## B. Manuscript Tables (Cleaned — use these for citations)

| File | Table # | Status | Notes |
|------|---------|--------|-------|
| `outputs/paper_tables_cleaned/main_results_summary.csv` | Table 1 | ✅ Main paper | Policy accuracy, cost, revise rate |
| `outputs/paper_tables/real_routing/real_policy_eval_comparison_long.csv` | Table 2 | ✅ Main paper | All routes × all regimes (long format) |
| `outputs/paper_tables_cleaned/final_cross_regime_summary_fixed.csv` | Table 3 | ✅ Main paper | 4-regime summary; use over `cross_regime/` version |
| `outputs/paper_tables_cleaned/oracle_routing_eval.csv` | Table 4 | ✅ Main paper | Oracle upper bounds per regime |
| `outputs/paper_tables_cleaned/budget_curves_all_datasets.csv` | Table 5 | ✅ Main paper | Accuracy vs target cost |
| `outputs/paper_tables_cleaned/baselines_appendix.csv` | Table A1 | 📎 Appendix | n=15–30; cannot compare to n=100 mains |
| `outputs/paper_tables_cleaned/cross_regime_summary.csv` | Table A2 | 📎 Appendix | 3-regime version; superseded by Table 3 |

> **Note:** The `outputs/paper_tables/` directory contains the same data in
> its original exported form.  Prefer `outputs/paper_tables_cleaned/` for
> anything submitted to the journal.

---

## C. Manuscript Figures (Cleaned)

| File | Figure # | Status | Notes |
|------|----------|--------|-------|
| `outputs/paper_figures_cleaned/real_routing/accuracy_vs_cost_2x2_composite.png` | Figure 1 | ✅ Main paper | Primary overview: accuracy vs cost, 4 regimes |
| `outputs/paper_figures_cleaned/real_routing/accuracy_vs_cost_hard_gsm8k_100.png` | Figure 1a | ✅ Main paper | Standalone hard-regime panel |
| `outputs/paper_figures_cleaned/real_routing/accuracy_vs_cost_hard_gsm8k_b2.png` | Figure 1b | ✅ Main paper | Standalone replication panel |
| `outputs/paper_figures_cleaned/next_stage/budget_curve_hard_gsm8k_b2.png` | Figure 2a | ✅ Main paper | Budget curve, best single dataset |
| `outputs/paper_figures_cleaned/next_stage/budget_curve_hard_gsm8k_100.png` | Figure 2b | ✅ Main paper | Budget curve, paired hard-regime |

---

## D. Supporting Outputs (Policy and Oracle Evaluations)

| Path | Status | Notes |
|------|--------|-------|
| `outputs/real_policy_eval/` | ✅ Main paper | GSM8K policy eval summary JSON + CSV |
| `outputs/real_hard_gsm8k_policy_eval/` | ✅ Main paper | Hard GSM8K policy eval |
| `outputs/real_hard_gsm8k_b2_policy_eval/` | ✅ Main paper | Hard GSM8K B2 policy eval |
| `outputs/real_math500_policy_eval/` | ✅ Main paper | MATH500 policy eval |
| `outputs/oracle_routing_eval/` | ✅ Main paper | Oracle routing upper bounds |
| `outputs/budget_sweep/` | ✅ Main paper | Budget-vs-accuracy sweep CSVs |
| `outputs/cross_regime_comparison/` | ✅ Main paper | Cross-regime summary |
| `outputs/baselines/` | ✅ Main paper | Baseline strategy comparison JSON |
| `outputs/next_stage_eval/` | ✅ Main paper | Budget curve per dataset |

---

## E. Exploratory / Diagnostic Outputs

| Path | Status | Notes |
|------|--------|-------|
| `outputs/adaptive_policy_v7/` | 🔬 Exploratory | Diagnostic analysis of v7 signal quality |
| `outputs/real_routing_model/` | 🔬 Exploratory | Learned tree-ensemble routing (not the primary paper method) |
| `outputs/real_math500_routing_model/` | 🔬 Exploratory | MATH500 learned routing model |
| `outputs/real_hard_gsm8k_routing_model/` | 🔬 Exploratory | Hard-regime learned routing model |
| `outputs/real_hard_gsm8k_b2_routing_model/` | 🔬 Exploratory | Hard-regime B2 learned routing model |
| `outputs/hard_regime_selection/` | 🔬 Exploratory | Hard-query selection methodology |
| `outputs/hard_regime_selection_b2/` | 🔬 Exploratory | Hard-query selection B2 |
| `outputs/reasoning_then_revise/` | 🔬 Exploratory | RTR add-on analysis |
| `outputs/real_routing_dataset/raw_responses.jsonl` | 🔬 Exploratory | Raw API responses (traceability only) |
| `outputs/real_aime2024_routing/raw_responses.jsonl` | 🔬 Exploratory | AIME raw API responses |

---

## F. Blocked / Incomplete Artifacts

| Artifact | Status | What is missing |
|----------|--------|-----------------|
| Simulated allocation sweep tables | ❌ Incomplete | Run `python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml` to generate |
| Oracle subset evaluation | ❌ Incomplete | Run `python3 scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml` to generate |
| AIME-2024 policy evaluation | ❌ Incomplete | Only 30 queries; policy eval not executed |
| GPQA-Diamond policy evaluation | ❌ Incomplete | No policy eval run |
| TALE / BEST-Route external baselines | ❌ Incomplete | Wrapper code only; external repos not cloned |

> None of these blocked artifacts are cited as final results in the manuscript.

---

## G. Source Code Status

| Module | Status | Notes |
|--------|--------|-------|
| `src/policies/adaptive_policy_v5.py` | ✅ Main paper | Core routing policy (easy regimes) |
| `src/policies/adaptive_policy_v6.py` | ✅ Main paper | Core routing policy (easy regimes) |
| `src/policies/adaptive_policy_v7.py` | ✅ Main paper | Core routing policy (hard regimes) |
| `src/policies/adaptive_policy_v1.py`–`v4.py` | 🔬 Exploratory | Earlier iterations; superseded by v5–v7 |
| `src/allocators/` | ✅ Main paper | Equal and MCKP budget allocation |
| `src/evaluation/` | ✅ Main paper | Per-query logging, metrics, oracle eval |
| `src/features/` | ✅ Main paper | Query-level routing features |
| `src/datasets/` | ✅ Main paper | Dataset loaders |
| `src/baselines/` | ✅ Main paper | Greedy, best-of-N, self-consistency |
| `src/baselines/external/` | 🔬 Exploratory | TALE/BEST-Route stubs (no official code) |
| `src/methods/` | 🔬 Exploratory | Mode-then-budget, selective escalation |
| `src/paper_artifacts/` | ✅ Main paper | Table/figure export utilities |
| `src/strategies/action_catalog.py` | 🔬 Exploratory | Multi-action space (beyond binary routing) |

---

## H. Documentation Status

| File | Status | Notes |
|------|--------|-------|
| `README.md` | ✅ Public | Main entry point |
| `MANUSCRIPT_REPRODUCTION.md` | ✅ Public | Reviewer reproduction guide |
| `PAPER_ARTIFACT_STATUS.md` | ✅ Public | This file |
| `REPRODUCIBILITY.md` | ✅ Public | Detailed command reference |
| `DATA_AVAILABILITY.md` | ✅ Public | Dataset provenance |
| `MANUSCRIPT_ARTIFACTS.md` | ✅ Public | Artifact inventory |
| `PUBLIC_RELEASE_CHECKLIST.md` | ✅ Public | Release hygiene record |
| `docs/PROJECT_CONTEXT.md` | ✅ Public | Research framing and MCKP connection |
| `docs/BASELINE_TRACKER.md` | ✅ Public | Baseline comparison status |
| `docs/FINAL_MANUSCRIPT_TABLE_FIGURE_INDEX.md` | ✅ Public | Definitive table/figure index |
| `docs/STATE_OF_EVIDENCE.md` | ✅ Public | Honest evidence-level audit |
| `docs/internal/` | 🔬 Internal | AI-agent planning logs; not for external readers |
| All other `docs/*.md` | 🔬 Exploratory | Research working notes; not for citation |
