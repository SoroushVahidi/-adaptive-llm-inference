# Baseline Implementation Audit (Canonical Main-Paper Regimes)

## Canonical setup inferred from repository
- Finalized canonical regimes: `gsm8k_random_100`, `hard_gsm8k_100`, `hard_gsm8k_b2`, `math500_100`.
- Canonical cheap baseline: `reasoning_greedy` (always-cheap / reasoning-only).
- Canonical always-revise baseline: `direct_plus_revise`.
- Canonical adaptive policy: `adaptive_policy_v5` (v6/v7 retained as secondary variants).
- Canonical manuscript-facing artifact roots: `outputs/paper_tables_final/` and `outputs/paper_figures_final/`.

## Baseline inventory classification

| Baseline | Classification | Evidence |
|---|---|---|
| reasoning_greedy (always-cheap) | fully implemented and runnable | executed on all 4 regimes in `run_real_policy_eval` outputs |
| direct_plus_revise (always-revise) | fully implemented and runnable | executed on all 4 regimes in `run_real_policy_eval` outputs |
| adaptive_policy_v5 (canonical adaptive) | fully implemented and runnable | executed on all 4 regimes in `run_real_policy_eval` outputs |
| adaptive_policy_v6 / v7 | fully implemented and runnable | executed on all 4 regimes in `run_real_policy_eval` outputs |
| confidence_threshold router | fully implemented and runnable | executed on all 4 regimes via enriched datasets |
| learned_router (logistic/tree) | partially implemented | code and CLI exist; run blocked in this environment by missing `scikit-learn` |
| best_route_official wrapper | stub/placeholder for this repo state | runtime requires `external/best_route/.repo` and bridge remains `NotImplementedError` |
| best_route_adapted | fully implemented and runnable adaptation | tested in repo tests; documented as non-official compatibility adaptation |
| TALE external baseline | partially implemented | external wrapper/docs present but not integrated in canonical finalized comparison pipeline |
| Snell / PRM / SelfBudgeter / DEER tracker entries | absent | tracker lists as planned without integrated runnable implementations in canonical pipeline |
