# Tonight’s Thinking Question

## Main question

**After one `reasoning_greedy` pass, what observable state \(s\) makes the *next* unit of compute (e.g. `direct_plus_revise` or an extra sample) worth more than spending that unit on a *different* query or not spending it at all—and does my current score (v4/v5 / unified error) rank those cases correctly?**

---

## Three subquestions

1. **Marginal vs absolute error:** If the model’s answer is wrong, is it *always* worth paying for revise, or only when revise’s *counterfactual* success probability minus cost beats staying wrong and saving budget for elsewhere?

2. **Baseline counterfactual:** Should escalation be judged against “do nothing” or against “run `reasoning_greedy` again” / “best-of-3 from scratch”? The right baseline determines what \(\Delta(s)\) means for your paper.

3. **Parser and cue failure:** Which failures make \(s\) **misleading** (looks consistent, still wrong; or looks inconsistent, revise doesn’t help)? Those cases dominate regret under a budget.

---

## Five concrete examples / failure modes from the repo story

These are **story anchors** from documented *method structure* and offline tooling—not live GSM8K traces (those CSVs are absent; see `docs/CONSISTENCY_FAILURE_EXAMPLES.md`).

1. **Intermediate-as-final** — Model returns a quantity that appears in the narrative but is not the asked target; `src/analysis/consistency_benchmark.py` labels `intermediate_as_final`. Your triggers must fire when the *asked* target type (remainder vs total) is wrong, not only when arithmetic looks inconsistent.

2. **Wrong target quantity** — Question asks “how many left” but the answer reflects a sum; `wrong_target_quantity` in the same benchmark. This is the core GSM8K routing narrative in `docs/FEATURE_GAP_ANALYSIS.md` (revise-helps patterns).

3. **Rate vs total** — “Per day” / “each” cues vs computed rate; failure type `rate_vs_total`. `src/features/target_quantity_features.py` encodes rate/unit cues—**tonight’s check** is whether those cues predict *marginal* gain from revise, not just presence.

4. **Unified score false comfort** — `src/features/unified_error_signal.py` mixes seven families with fixed weights; a **low** unified error can still be wrong on gold, and a **high** error might be wasted revise if the mistake is unfixable by the revise prompt.

5. **Knapsack without a profit column** — `src/allocators/mckp_allocator.py` is correct *given profits*, but `src/policies/adaptive_policy_v5.py` never outputs a profit row per query for the allocator. The **failure mode** is optimizing allocation in theory while routing in practice uses a different, non-comparable objective.

---

## Why this matters more than adding another feature family

Another regex block or another policy version **does not reduce uncertainty** about the paper’s core claim: **budgeted utility vs a strong one-call baseline.** Without a clear **marginal-value** target, new features only add degrees of freedom to hand-tune. Nailing the estimand (what to predict, how to score rankings, how cost enters) decides whether the next engineering effort is **identification** or **noise**.

---

## Related repo files

- `docs/OFFLINE_CONCEPTUAL_BOTTLENECK_REVIEW.md` — full audit and bottleneck argument  
- `src/analysis/consistency_benchmark.py` — failure-type vocabulary for offline thought experiments  
- `docs/STATE_OF_EVIDENCE.md` — separates implemented vs measured vs blocked  
- `src/features/unified_error_signal.py` — current aggregation you should stress-test conceptually  
