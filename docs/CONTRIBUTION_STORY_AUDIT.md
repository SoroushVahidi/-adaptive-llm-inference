# Contribution Story Audit (Repository-Grounded, Manuscript-Oriented)

## Scope and constraints

This audit uses only committed repository artifacts (docs + outputs) and does **not** run new inference.

Requested file `docs/V5_V6_V7_STORY_CHECK.md` is **not present** in the current repository; this audit uses the closest available evidence from `ADAPTIVE_POLICY_V5/V6/V7`, policy outputs, and paper tables.

---

## 1) Strongest true contribution currently supported

### Short answer

The strongest currently-supported contribution is a **policy-design + evaluation** contribution:

> A lightweight, interpretable escalation policy family (V5→V7) can achieve near-Always-DPR accuracy on hard arithmetic regimes at materially lower average cost, while exposing explicit routing headroom and known failure modes.

### Why this is strongest (from artifacts)

- Cross-regime policy comparisons exist for four main 100-query regimes and include RG, DPR, V5/V6/V7, and oracle-derived headroom tables.
- Hard-regime evidence is clear: V5 matches DPR accuracy on hard GSM8K splits with lower cost in paper tables.
- Oracle headroom tables and routing outcome breakdowns provide an interpretable decomposition (`both_correct`, `dpr_only_correct`, `both_wrong`) that supports a *regime-aware* story.
- V6/V7 documentation introduces an interpretable semantic split (explanation-warning vs answer-error) with explicit claims and limitations.

---

## 2) What category is the contribution?

## Best categorization

**Primary:** Policy design contribution  
**Secondary:** Systems/evaluation contribution  
**Not primary:** Framework-theory contribution

Rationale:

- The repository’s strongest novel mechanism is in policy logic changes (especially V6/V7 signal semantics).
- The paper-strengthening artifacts are mostly evaluation/statistical packaging (bootstrap CIs, paired tests, headroom decomposition), which supports systems/evaluation quality.
- There is no large-scale novel formal framework proof; the contribution is better framed as *interpretable policy design under cost constraints* plus *careful empirical decomposition*.

---

## 3) Claim grading: strongly supported vs weakly supported vs unsupported

## Strongly supported (safe to put in main claims)

1. **Regime dependence of escalation value is real.**
   - `revise_helpful_rate` is low on easy GSM8K random (2%), higher on hard regimes (9–12%), and moderate on math500 (6%).
2. **V5 can match DPR on hard regimes with lower average cost.**
   - Hard GSM8K tables show V5 ≈ DPR accuracy with substantial cost reduction vs always revising.
3. **Oracle headroom remains non-zero in some regimes.**
   - Especially hard_gsm8k_100 and math500; routing remains improvable.
4. **V6 semantic decoupling addresses a specific false-positive pattern from V5 in targeted fixtures.**
   - Strong as a *local mechanism claim* (not a full-population claim).

## Weakly supported (keep as qualified analysis, not headline)

1. **V7 broadly improves over V6 at scale.**
   - V7 docs show targeted fix wins on a small probe and preserved concise-correct behavior on fixture cases; broader superiority is explicitly marked exploratory.
2. **Learned routing is currently superior.**
   - Existing learned-router outputs are regime-dependent and often label-scarce; not a stable headline.
3. **Method-fit by query type is fully solved.**
   - Signals point in the right direction, but category-level causal conclusions are not robustly established across large samples.

## Unsupported (avoid as claims)

1. **General SOTA-like broad claims across datasets/tasks (including AIME/GPQA).**
   - AIME/GPQA policy-eval coverage is incomplete.
2. **Causal claims about specific features “causing” performance gains.**
   - Current evidence is correlational/heuristic and sample-limited.
3. **Universal dominance of a single adaptive policy version.**
   - Best policy identity changes by regime/file family; some summary files are inconsistent across artifact sets.

---

## 4) Single-sentence best claim for the paper

> Across multiple arithmetic reasoning regimes, interpretable selective escalation can recover most (and sometimes all) of the accuracy gain from always-revise at much lower cost, with performance tightly coupled to regime-specific revise-helpful prevalence and measurable oracle headroom.

---

## 5) What would make this look stronger to a KBS reviewer

1. **Center the contribution on “interpretable selective escalation” rather than “yet another router.”**
   - Emphasize the semantic routing split: answer-error drives revise; explanation-warning primarily informs caution.
2. **Use headroom decomposition as a first-class analytic lens.**
   - Keep `dpr_only_correct` and `both_wrong` decomposition in main text to show where routing helps vs where model capability is the bottleneck.
3. **Explicitly separate claims by evidence tier.**
   - “Measured-now / strong,” “measured-but-small-sample,” and “exploratory” labeling should appear in main results.
4. **Tighten statistical language.**
   - Use effect sizes + CIs as primary, p-values as secondary; avoid over-reading non-significant small differences at n=100.
5. **Make revise-worthiness taxonomy explicit in manuscript.**
   - This improves conceptual clarity, interpretability, and reusability for KBS readership.
6. **Acknowledge unresolved regions as actionable headroom, not failure.**
   - Especially math500 oracle gap and `both_wrong` mass.

---

## 6) Practical manuscript positioning recommendation

If forced to choose one identity statement:

- **“Interpretable cost-aware policy design with regime-aware evaluation decomposition.”**

Avoid framing as a pure ML classifier paper; the current strongest evidence is in policy semantics + decomposition-based evaluation.
