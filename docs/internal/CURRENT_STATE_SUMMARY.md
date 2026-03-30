# Current State Summary

**Date:** 2026-03-28  
**Based on:** Full repository audit (`docs/FULL_REPO_AUDIT.md`) and results inventory (`docs/RESULTS_INVENTORY.md`).

> **All answers below are grounded only in files present in the repository.**  
> No real-LLM results have been verified — `outputs/` does not exist and `OPENAI_API_KEY` is not available.  
> The numeric claims drawn from `docs/ORACLE_ANALYSIS_SUMMARY.md` are noted as UNVERIFIED.

---

## Q1. What is the strongest cheap baseline?

**Grounded answer:** `reasoning_greedy`  

According to `docs/ORACLE_ANALYSIS_SUMMARY.md` (UNVERIFIED), `reasoning_greedy` improves exact-match accuracy from 0.50 → 0.65 on 20 GSM8K queries at the **same average cost (1.0)** as `direct_greedy`. It fixes 3 direct-greedy failures while remaining the cheapest-correct strategy on 3 queries. `direct_greedy` (0.50) and `strong_direct` (0.55) are weaker alternatives. The design docs for adaptive policies v1–v4 consistently treat `reasoning_greedy` as the default low-cost fallback, lending further support to this assessment.

---

## Q2. What is the strongest stronger/corrective baseline?

**Grounded answer:** `direct_plus_revise`  

Per `docs/ORACLE_ANALYSIS_SUMMARY.md` (UNVERIFIED), `direct_plus_revise` matches the top accuracy of 0.70 at average cost 2.0, while `reasoning_best_of_3` also reaches 0.70 but costs 3.0. `direct_plus_revise` is therefore the **dominant corrective strategy** on this subset: same accuracy, lower cost. The `direct_plus_critique_plus_final` and `first_pass_then_hint_guided_reason` strategies reach only 0.65, no better than `reasoning_greedy` but at higher cost. `direct_plus_verify` is weak overall (0.45) but is noted for one unique success not achieved by any other strategy.

---

## Q3. Does extra sampling help?

**Grounded answer:** Very limited marginal benefit, high cost  

Per `docs/ORACLE_ANALYSIS_SUMMARY.md` (UNVERIFIED): `reasoning_best_of_3` improves from 0.65 → 0.70 over `reasoning_greedy` (a gain of +0.05) but triples average cost from 1.0 → 3.0. `structured_sampling_3` adds **zero gain** over `reasoning_greedy` (both 0.65) while also costing 3.0. The oracle analysis concludes: extra sampling is "mostly cost, with limited marginal benefit." This finding is design-documented in `docs/ORACLE_ANALYSIS_SUMMARY.md` and encoded into the adaptive policy v1–v4 designs (which deprioritise best-of-N).

---

## Q4. Does structured sampling help?

**Grounded answer:** No  

`structured_sampling_3` (same accuracy as `reasoning_greedy`, three times the cost) is explicitly called out as "weak on MATH500 and no better than reasoning_greedy here" and placed on the **drop/deprioritize** list in `docs/ORACLE_ANALYSIS_SUMMARY.md`. It is the clearest negative result in the documented evidence.

---

## Q5. Do adaptive policies beat `reasoning_greedy`?

**Grounded answer:** Unknown — no verified outputs exist  

The four adaptive policies (v1–v4) are all fully implemented (`src/policies/`), tested (`tests/test_adaptive_policy*.py`), and have evaluation scripts and configs. However, **all runs are blocked by the missing API key**. No `outputs/adaptive_policy_*/` files exist.

What is known from design documents:
- **v1** was too aggressive: escalated to `direct_plus_revise` on all 20/20 queries (near-always-revise policy), negating cost savings.
- **v2** was too conservative: escalated on 0/20 queries (effectively identical to `reasoning_greedy`).
- **v3** tried weighted threshold calibration to achieve a selectivity between v1 and v2; its design says the "best setting still behaved almost exactly like `reasoning_greedy`".
- **v4** switches to constraint-aware consistency signals (answer-type mismatch, unit mismatch, etc.) hoping to achieve genuine selective escalation.

Whether v4 finally achieves a net improvement over `reasoning_greedy` on a cost-accuracy Pareto frontier is the **open empirical question** this project must answer.

---

## Q6. What is the main bottleneck now?

**The single main bottleneck is the absence of a live `OPENAI_API_KEY`.**

Every experiment from oracle subset evaluation through adaptive policy v4 to the router baseline requires real model calls. Until real results exist:
- The numeric claims in `docs/ORACLE_ANALYSIS_SUMMARY.md` remain unverified.
- Whether any adaptive policy outperforms `reasoning_greedy` is unknown.
- Whether the target-quantity and constraint-violation features actually separate routing groups is unknown.
- Whether the router baseline (decision tree / logistic regression) can learn useful routing from query features is unknown.

A secondary bottleneck is **dataset coverage**: only 20 bundled GSM8K queries are available offline. Even with an API key, results on 20 queries have high variance and cannot support robust paper claims. A 100–500 query oracle evaluation is needed.

---

## Top 3 Highest-Value Next Steps

1. **Run `run_oracle_subset_eval.py` with a live API key on the 20-query bundled sample.**  
   This is the cheapest real experiment (~140 API calls at most) and produces the oracle labels that unlock all downstream analysis: routing dataset, feature gap analysis, router baseline, and adaptive policy comparison. Every other analysis depends on this output.

2. **Extend oracle evaluation to 100–200 queries.**  
   The 20-query subset has variance too high for paper claims (e.g., 1 extra correct answer changes accuracy by 0.05). 100–200 queries would provide enough signal to compare adaptive policies and train the router baseline with some statistical confidence.

3. **Run adaptive policy v4 evaluation and measure revise selectivity.**  
   V4 is the culmination of four policy iterations and the clearest adaptive contribution of the project so far. A single `run_adaptive_policy_v4_eval.py` run produces: (a) accuracy vs `reasoning_greedy`, (b) fraction of queries escalated to revise, (c) signal-firing breakdown. These three numbers directly answer whether the project has a working adaptive policy.

---

## Top 3 Things NOT Worth Doing Right Now

1. **Integrating TALE or BEST-Route from official code.**  
   These are important baselines for the paper, but they require significantly more engineering effort (cloning, adapting, and testing external repos) than running the native experiments. They are not blocking the core empirical contribution. The priority is to first confirm the adaptive policy works at all.

2. **Implementing token-budget, early-exit, tree-of-thoughts, or ReAct strategies.**  
   These are placeholders in the action catalog with no implementation. None of them are required for the current research narrative (adaptive compute allocation over the already-implemented strategy family). Adding them increases scope without directly advancing the paper argument.

3. **Training the router baseline on the current 20-query oracle dataset.**  
   The router baseline (`src/policies/router_baseline.py`) is implemented and tested, but the docs explicitly warn that with ≤ 20 labelled queries any decision tree or logistic regression will overfit or produce near-random results. The correct action is to first expand the oracle dataset (step 2 above), then train the router.
