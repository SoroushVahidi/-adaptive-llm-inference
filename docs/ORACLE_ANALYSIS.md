# Oracle Strategy Analysis

## What this analysis is

Oracle strategy analysis evaluates a fixed set of strategies on each query and then asks:

- **Accuracy oracle:** which strategy would have been best for this query if we could choose after seeing all outcomes?
- **Cost-aware oracle:** which strategy maximizes `u(x_i, a) - λ c(a)` where correctness utility `u` is binary and cost `c` is sample count?

This yields a per-query matrix `(x_i, a, prediction, correct, cost)` and oracle assignments.

## Why it matters

Aggregate accuracy hides substantial heterogeneity. In practice, different queries are solved by different strategies:

- some are solved by direct greedy,
- some only by critique/revision loops,
- some by multi-sample voting,
- some by stronger models.

Oracle analysis quantifies the **headroom** between current static policies and the best achievable query-wise selection.

## Connection to adaptive compute

Adaptive inference aims to allocate compute where needed. Oracle analysis provides the empirical target:

- `oracle_accuracy - direct_accuracy` estimates potential gains from routing/escalation,
- per-query oracle assignments reveal where extra samples or extra stages are justified,
- cost-aware oracle (λ > 0) traces the quality-cost frontier.

## Connection to routing policies

A learned or heuristic router can be trained/evaluated against oracle assignments:

- predict when direct is sufficient,
- escalate only when likely to improve correctness,
- choose among families (verify/revise/critique/multi-sample/strong model).

This turns oracle labels into supervision for strategy selection.

## Connection to MCKP formulation

In Multi-Choice Knapsack Problem (MCKP) terms:

- each query is a group,
- each strategy is an item choice with value (expected correctness gain) and weight (cost),
- global budget constraints decide one choice per query.

Oracle analysis provides the per-query payoff table needed to estimate item values and understand where budget should be spent.

## Outputs

`scripts/run_oracle_strategy_eval.py` writes:

- `outputs/oracle_strategy_eval/summary.json`
- `outputs/oracle_strategy_eval/summary.csv`
- `outputs/oracle_strategy_eval/per_query_matrix.csv`
- `outputs/oracle_strategy_eval/oracle_assignments.csv`

These files are designed to plug into downstream routing and budget-allocation studies.
