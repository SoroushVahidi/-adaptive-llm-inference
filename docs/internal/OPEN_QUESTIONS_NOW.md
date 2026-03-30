# Open Questions Now (Evidence-first)

## Primary question (highest priority)

### In simple words
Can we route compute **selectively and reliably** so we beat `reasoning_greedy` on quality-per-cost, instead of just adding complexity that does not pay off?

### In technical form
Does any interpretable or lightweight learned policy over query/first-pass features produce a robust positive budgeted-utility delta versus fixed baselines (especially `reasoning_greedy`) across GSM8K and MATH500, with reproducible outputs?

---

## What is currently unknown vs known

### Known (code-level)
- Rich feature families and policy variants v1–v4 exist and are test-covered.
- Strategy space includes cheap, sampled, and corrective pipelines.
- Routing dataset and router baselines are implemented end-to-end.

### Unknown (evidence-level)
- Whether adaptive policies actually outperform `reasoning_greedy` in committed empirical results.
- Whether target-quantity and constraint-aware signals have enough predictive power to justify another hand-crafted router iteration.
- Whether extra sampling (`best_of_3`, structured variants) offers enough marginal gain at realistic cost.

---

## Immediate scientific uncertainties to resolve (ranked)

1. **Policy value question:** Is there measurable, consistent gain from adaptive routing over cheap one-pass reasoning?
2. **Signal adequacy question:** Are current hand-crafted features sufficient, or is learned routing now necessary?
3. **Cost frontier question:** Which corrective strategy, if any, offers the best incremental accuracy per added cost?
4. **Generalization question:** Do findings transfer from GSM8K subset conditions to harder math regimes (MATH500)?

---

## Decision criterion for next phase

Proceed to additional experimental cycles only if evidence collection resolves at least one of these with committed artifacts:
- a clear positive routing delta vs `reasoning_greedy`, or
- a clear negative result showing hand-crafted signals are insufficient (thus justifying learned routing focus).
