# Methodology Transfer Notes

Source inspected for transferable ideas:
- External repository: `https://github.com/SoroushVahidi/combinatorial-opt-agent`
- Key artifacts inspected: repository README, `current_repo_vs_manuscript_rerun.md`,
  `publish_now_decision_report.md`, `EXPERIMENTS.md`, and
  `literature_informed_rerun_report.md`.

> Scope guard: this project is **adaptive LLM inference under budget**, not NLP4LP.
> We only transfer methodological patterns, not domain-specific retrieval/schema logic.

---

## 1) Candidate ideas and transfer decisions

| Idea from external repo | Why transferable | How it maps here | Decision |
|---|---|---|---|
| Explicit "run locally" vs "inferred from prior artifacts" split | Prevents overclaiming when parts are blocked | Add an evidence ledger section in eval docs/reports that labels every metric as `measured_now`, `from_prior_artifact`, or `blocked` | **Adopt now** |
| Conservative decision report (publish-now gate) | Forces scientific honesty at decision points | Add a repeatable paper-readiness checklist for adaptive-inference claims (baseline coverage, artifact presence, blocker status) | **Adopt now** |
| Bottleneck-first narrative (strong upstream, weak downstream) | Cleanly separates what works from what blocks impact | Keep and formalize current project bottleneck framing: strategy infrastructure strong, empirical routing gains under budget still bottlenecked by recorded outputs | **Adopt now** |
| Robustness variants (`orig/noisy/short`) as controlled stress tests | Generalizable way to probe method brittleness | Later: define GSM8K perturbation slices (e.g., shortened prompt, wording-noise, number-format perturbation) for routing stability checks | **Adopt later** |
| "No single method dominates all metrics" frontier reporting | Important for multi-objective systems | Report Pareto-style summary (accuracy, average cost, failure-recovery rate) rather than single winner | **Adopt now** |
| Domain-specific schema-slot typing and mention assignment machinery | Tied to NLP4LP grounding task | Not aligned with current adaptive inference objective | **Do not adopt** |
| Domain-specific constrained assignment optimizer internals | Problem-specific to parameter-slot grounding | Not needed for current strategy-routing question | **Do not adopt** |

---

## 2) Adopted now (in this transfer pass)

1. **Evidence ledger principle**
   - Every major result should be labeled by evidence status.
   - Prevents mixing measured metrics with historical/manuscript references.

2. **Paper-readiness gate**
   - A compact checklist for deciding when claims are publish-ready.
   - Includes blocker transparency and reproducibility criteria.

3. **Multi-metric frontier reporting**
   - Prefer tradeoff tables over single-metric winner claims.
   - Particularly important for budget-aware routing (quality vs cost).

These are codified in `docs/UPDATED_EVALUATION_PRINCIPLES.md`.

---

## 3) Adopt later (after current bottleneck is reduced)

1. **Controlled perturbation benchmark variants**
   - Add stress-test slices for adaptive routing robustness.

2. **Expanded decision reports with automatic artifact checks**
   - Move from doc checklist to script-generated readiness reports once more outputs are available.

---

## 4) Not adopted

- External repo's schema/slot-specific modeling and optimization internals are out-of-scope for this project's scientific question and should not be force-fitted.

---

## 5) Expected impact

- **Framing:** clearer intermediate-capability narrative and fewer overclaims.
- **Evaluation:** improved reproducibility and explicit uncertainty accounting.
- **Interpretability:** clearer statement of what each method improves (and what it does not).
- **Method design:** better guidance for prioritizing bottleneck-reducing work over strategy sprawl.
