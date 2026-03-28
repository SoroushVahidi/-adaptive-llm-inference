# Results Inventory

**Date:** 2026-03-28  
**Scope:** All empirical results present in or inferable from the repository.  
**Finding:** The `outputs/` directory does not exist (gitignored). No result files (JSON, CSV, TXT, YAML) are committed. All real-LLM experiments are blocked by a missing `OPENAI_API_KEY`.

---

## Critical Status Note

> **No verified, reproducible results exist in this repository.**  
> All experiments requiring a live OpenAI API key have not been run.  
> The only runnable experiments (dummy model, synthetic TTC) produce synthetic numbers
> with no external validity for the paper.

---

## Documented Numeric Claims (unverified)

The table below lists all numeric claims found in documentation files.
Each row notes the source document and the verification status.

| Experiment | Dataset | Strategies compared | Main metric(s) | Key claim | Source document | Verification status |
|------------|---------|---------------------|----------------|-----------|----------------|---------------------|
| Oracle subset evaluation | GSM8K, 20 queries | `direct_greedy`, `reasoning_greedy`, `reasoning_best_of_3`, `structured_sampling_3`, `direct_plus_verify`, `direct_plus_revise`, `direct_plus_critique_plus_final`, `first_pass_then_hint_guided_reason`, `strong_direct` | Accuracy (exact-match), avg cost per query, wins, fixes | `direct_greedy`=0.50, `reasoning_greedy`=0.65, `reasoning_best_of_3`=0.70, `direct_plus_revise`=0.70, `structured_sampling_3`=0.65, `direct_plus_verify`=0.45, `direct_plus_critique_plus_final`=0.65, `first_pass_then_hint_guided_reason`=0.65, `strong_direct`=0.55; oracle=0.75 | `docs/ORACLE_ANALYSIS_SUMMARY.md` | **UNVERIFIED** — `docs/RESULTS_ORACLE_SUBSET.md` and `docs/EXPERIMENT_LOG_ORACLE_SUBSET.md` both say BLOCKED. No output files exist. Numbers may be from a prior run, a different environment, or planned analysis. |
| Oracle delta analysis | GSM8K, 20 queries | Oracle vs `direct_greedy`, oracle vs `reasoning_greedy` | Oracle gap | Oracle−direct gap=+0.25; oracle−reasoning gap=+0.10; avg oracle cost on successful queries=1.07 | `docs/ORACLE_ANALYSIS_SUMMARY.md` | **UNVERIFIED** (same caveat as above) |
| Strategy redundancy finding | GSM8K, 20 queries | `structured_sampling_3` vs `reasoning_greedy` | Accuracy, cost | `structured_sampling_3` adds zero gain over `reasoning_greedy` (both 0.65) but triples cost (3.0 vs 1.0) | `docs/ORACLE_ANALYSIS_SUMMARY.md` | **UNVERIFIED** |
| Revise vs best-of-3 cost-efficiency | GSM8K, 20 queries | `direct_plus_revise` vs `reasoning_best_of_3` | Accuracy, avg cost | Both reach 0.70; `direct_plus_revise` at cost 2.0, `reasoning_best_of_3` at cost 3.0 | `docs/ORACLE_ANALYSIS_SUMMARY.md` | **UNVERIFIED** |
| Adaptive policy v1 trigger rate | GSM8K, 20 queries | v1 routing: `reasoning_greedy` vs `direct_plus_revise` | % queries escalated to revise | v1 triggered `direct_plus_revise` on 20/20 queries | `docs/ADAPTIVE_POLICY_V3.md` (design note) | **UNVERIFIED** — cited as design context, not as an output file |
| Adaptive policy v2 trigger rate | GSM8K, 20 queries | v2 routing: `reasoning_greedy` vs `direct_plus_revise` | % queries escalated to revise | v2 triggered `direct_plus_revise` on 0/20 queries | `docs/ADAPTIVE_POLICY_V3.md` (design note) | **UNVERIFIED** — same caveat |
| Target-quantity feature fire rates | GSM8K bundled sample, 20 queries | — | Feature activation rates | `asks_remaining_or_left`=0.10, `asks_total`=0.25, `asks_difference`=0.30, `asks_rate_or_unit`=0.25, `asks_money`=0.35, `asks_time`=0.50, `has_subtraction_trap_verb`=0.20, `has_addition_trap_structure`=0.35, `has_multi_operation_hint`=0.40, `likely_intermediate_quantity_ask`=0.25, `potential_answer_echo_risk`=0.25 | `docs/REVISE_HELP_FEATURE_ANALYSIS.md` | **OFFLINE-COMPUTED** — computed from bundled 20-query GSM8K sample using regex features, no API key required. Reproducible by running `python3 scripts/run_revise_help_feature_analysis.py`. |
| MCKP ≥ Equal allocation on synthetic TTC | Synthetic (120 queries, mixed difficulty, budget=300) | `MCKPAllocator` vs `EqualAllocator` | Total utility | MCKP outperforms equal on non-trivial synthetic instances | `tests/test_simulated_allocation.py` (test assertion) | **VERIFIED by test** — asserted by `test_mckp_beats_equal_on_nontrivial_instance`; runnable without API key or network |

---

## Offline-Runnable Experiments (no API key required)

These experiments can produce actual results in the current environment:

| Script | Config | Expected output path | Reproducible? |
|--------|--------|---------------------|---------------|
| `run_simulated_allocation.py` | `configs/simulated_mckp.yaml` | `outputs/simulated_mckp_results.json` | ✅ Yes |
| `run_simulated_allocation.py` | `configs/simulated_equal.yaml` | `outputs/simulated_equal_results.json` | ✅ Yes |
| `run_simulated_allocation.py` | `configs/simulated_sweep.yaml` | `outputs/simulated_sweep/` | ✅ Yes |
| `run_simulated_allocation.py` | `configs/simulated_multi_seed.yaml` | `outputs/simulated_multi_seed/` | ✅ Yes |
| `run_experiment.py` | `configs/greedy.yaml` | `outputs/greedy_*/` | ✅ Yes (dummy model) |
| `run_real_budget_sweep.py` | `configs/real_budget_sweep_gsm8k.yaml` | `outputs/real_budget_sweep/` | ✅ Yes (dummy model, correct_prob=0.3) |
| `build_routing_dataset.py` | `--dry-run` | `outputs/routing_dataset/routing_dataset.csv` | ✅ Yes (schema-only) |
| `run_revise_help_feature_analysis.py` | — | `outputs/revise_help_feature_analysis/` | ✅ Yes (bundled sample fallback) |

---

## Blocked Experiments (require `OPENAI_API_KEY`)

| Script | Config | Expected output path |
|--------|--------|---------------------|
| `run_oracle_subset_eval.py` | `oracle_subset_eval_gsm8k.yaml` | `outputs/oracle_subset_eval/` |
| `run_adaptive_policy_eval.py` | `adaptive_policy_gsm8k.yaml` | `outputs/adaptive_policy_v1/` |
| `run_adaptive_policy_v2_eval.py` | `adaptive_policy_v2_gsm8k.yaml` | `outputs/adaptive_policy_v2/` |
| `run_adaptive_policy_v3_eval.py` | `adaptive_policy_v3_gsm8k.yaml` | `outputs/adaptive_policy_v3/` |
| `run_adaptive_policy_v4_eval.py` | `adaptive_policy_v4_gsm8k.yaml` | `outputs/adaptive_policy_v4/` |
| `run_strategy_diagnostic_math500.py` | `strategy_diagnostic_math500.yaml` | `outputs/strategy_diagnostic_math500/` |
| `run_expanded_strategy_smoke_test.py` | `expanded_strategy_smoke_test_gsm8k.yaml` | `outputs/expanded_strategy_smoke_test/` |
| `run_selective_escalation.py` | `selective_escalation_gsm8k.yaml` | `outputs/selective_escalation/` |
| `run_mode_then_budget.py` | `mode_then_budget_gsm8k.yaml` | `outputs/mode_then_budget/` |
| `run_real_llm_experiment.py` | `real_llm_gsm8k.yaml` | `outputs/real_llm/` |
| `run_model_sampling_diagnostic.py` | `model_sampling_diagnostic_gsm8k.yaml` | `outputs/model_sampling_diagnostic/` |
| `run_real_gain_table.py` | `real_gain_table_gsm8k.yaml` | `outputs/real_gain_table/` |
| `run_real_budget_sweep.py` | `real_budget_sweep_gsm8k.yaml` (real LLM variant) | `outputs/real_budget_sweep/` |
| `run_router_baseline.py` | — | `outputs/router_baseline/` |
| `run_feature_gap_analysis.py` | — (needs oracle CSVs) | `outputs/feature_gap_analysis/` |
