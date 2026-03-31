# KBS Narrative Recommendation

## 1) Best paper identity

**Recommended identity:** **Hybrid (policy-method + evaluation systematization)**

- Primary identity: interpretable **method/policy design** for selective escalation.
- Secondary identity: **evaluation framework** using regime-aware headroom decomposition.

Avoid framing as pure classifier engineering.

---

## 2) Best title direction

Use a title pattern emphasizing interpretability + cost-aware routing + regime-awareness:

- “Interpretable Selective Escalation for Cost-Aware LLM Reasoning: A Regime-Aware Evaluation”
- “When to Escalate, When to Trust: Interpretable Adaptive Routing for LLM Reasoning”

---

## 3) Best abstract angle (one-paragraph direction)

Proposed angle:

1. Problem: always-revise improves accuracy but doubles cost and over-escalates.
2. Method: interpretable selective escalation with explicit separation of answer-error vs explanation-warning signals (V5→V6→V7 progression).
3. Results: on four 100-query regimes, adaptive policies recover much of DPR gain at lower cost, with strongest gains in hard regimes where revise-helpful prevalence is higher.
4. Analysis: oracle and outcome decomposition expose residual headroom and capability bottlenecks.
5. Limitation: small samples and incomplete AIME/GPQA policy-eval coverage.

---

## 4) Best introduction angle

Frame as a **decision-quality problem**:

- Not “can we make reasoning better overall?”
- But “can we escalate only when expected benefit exceeds cost, with interpretable criteria?”

Then motivate with two tensions:

1. false-positive escalation cost,
2. false-negative missed rescue.

Position V6 separation principle and V7 targeted repair as responses to these two errors.

---

## 5) Placement of feature experiment

- Place feature experiment in **late Results / Analysis** (after main policy results).
- Role: supportive audit of signal validity and headroom, not core novelty.
- If unstable or weak, move to appendix with brief mention in main text.

---

## 6) Placement of V5→V6→V7 story

- Put V5→V6→V7 in **Methods** (design rationale) and revisit in **Error Analysis**.
- Keep concise in main text:
  - V5: strong hard-regime performance but over-escalation risk.
  - V6: semantic decoupling.
  - V7: targeted false-negative fixes with explicit caveats.

---

## 7) Existing repo tables/figures to promote to main manuscript

Promote (main text):

1. `outputs/paper_tables/oracle_headroom_table.csv` (core headroom + gap framing).
2. `outputs/paper_tables/routing_outcome_breakdown.csv` (both_correct / dpr_only / both_wrong decomposition).
3. `outputs/paper_tables/policy_efficiency_table.csv` (accuracy recovered vs cost avoided).
4. `outputs/paper_figures/routing_headroom_barplot.png`.
5. `outputs/paper_figures/routing_outcome_stacked_bar.png`.
6. `outputs/paper_figures/adaptive_efficiency_scatter.png`.

---

## 8) Materials better kept appendix/supplement

1. Threshold sweeps and full sensitivity grids.
2. Full bootstrap/pairwise test tables (reference in main, details in appendix).
3. Full feature-importance listings and fold-level diagnostics.
4. Incomplete-regime artifacts (AIME/GPQA) unless clearly labeled preliminary.

---

## 9) Reviewer-facing narrative discipline

To match KBS expectations:

1. Separate **mechanism claim** (policy semantics) from **effect claim** (accuracy-cost tradeoff).
2. Report unresolved headroom explicitly as future work.
3. Use conservative wording around V7 and feature experiments.
