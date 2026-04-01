# Stronger Baseline Comparison Results (Canonical Finalized Regimes)

## Regimes used
- `gsm8k_random_100`
- `hard_gsm8k_100`
- `hard_gsm8k_b2`
- `math500_100`

## Canonical policy choice used
- Canonical adaptive policy treated as primary comparator: **`adaptive_policy_v5`**.

## Successfully run baselines in this experiment
- `reasoning_greedy` (always-cheap / reasoning-only)
- `direct_plus_revise` (always-revise)
- `adaptive_policy_v5`
- `confidence_threshold` router
- `adaptive_policy_v6`, `adaptive_policy_v7` (context rows)

## Blocked / unavailable baselines
- Learned router (`learned_router_logistic_regression`, `learned_router_decision_tree`): blocked in this environment because `scikit-learn` is unavailable.
- Official BEST-Route: blocked (missing `external/best_route/.repo`; official wrapper bridge not implemented).

## Quantitative summary (accuracy @ avg_cost)

| Regime | always-cheap | always-revise | adaptive v5 | confidence-threshold |
|---|---:|---:|---:|---:|
| gsm8k_random_100 | 0.90 @ 1.00 | 0.92 @ 2.00 | 0.92 @ 1.29 | 0.92 @ 1.11 |
| hard_gsm8k_100 | 0.79 @ 1.00 | 0.86 @ 2.00 | 0.86 @ 1.53 | 0.89 @ 1.13 |
| hard_gsm8k_b2 | 0.83 @ 1.00 | 0.91 @ 2.00 | 0.91 @ 1.41 | 0.89 @ 1.09 |
| math500_100 | 0.64 @ 1.00 | 0.64 @ 2.00 | 0.66 @ 1.71 | 0.66 @ 1.06 |

## Does this materially strengthen the paper?
- **Yes, with caveat.** Relative to only cheap-vs-adaptive comparisons, adding the confidence-threshold router provides a stronger adaptive baseline family and shows where canonical `adaptive_policy_v5` still has value (notably hard_gsm8k_b2 tie at higher accuracy vs confidence router), and where a simpler router can be stronger under tight cost targets (hard_gsm8k_100).
- **Caveat:** learned-router and official BEST-Route could not be included as fully-run baselines in this environment/repo state; this should be stated explicitly in manuscript limitations.

## Output artifacts
- `outputs/stronger_baseline_comparison/baseline_summary.csv`
- `outputs/stronger_baseline_comparison/baseline_summary.json`
- Per-regime raw policy eval outputs in `outputs/stronger_baseline_comparison/raw/*/`
- Learned-router attempt outputs in `outputs/stronger_baseline_comparison/learned_router/`
