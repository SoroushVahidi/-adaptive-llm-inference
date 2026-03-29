# Updated Evaluation Principles (Methodology Transfer)

This document introduces transferable evaluation-design principles adapted from
methodological patterns in `combinatorial-opt-agent`, rewritten for the current
adaptive-inference project.

---

## A) Evidence ledger for every reported result

For each table, figure, and headline metric, include exactly one label:

- `measured_now` — computed in the current repo run and backed by local output files.
- `from_prior_artifact` — copied from pre-existing artifacts not rerun in this pass.
- `blocked` — intended metric, but run was blocked (e.g., API access).

### Minimum required metadata per run

1. exact command
2. config path
3. output paths produced
4. blocker status (if any)
5. timestamp

---

## B) Multi-objective reporting (no single-metric winner by default)

Adaptive inference must report at least:

1. quality (accuracy)
2. cost (avg samples/calls or cost proxy)
3. reliability/failure-recovery metric (e.g., fixed-direct-failures)

### Rule

Do not claim a global "best method" unless it is non-dominated across all
required metrics. Otherwise present a tradeoff/frontier statement.

---

## C) Bottleneck-first narrative structure

Every experiment summary should include:

1. **What is strong now** (measured capability)
2. **What bottleneck remains** (measured failure mode)
3. **What next experiment directly targets that bottleneck**

This prevents adding strategies that do not address the current limiting factor.

---

## D) Paper-readiness gate (claim discipline)

A result is claim-ready only if all are true:

- [ ] required baseline comparisons are present
- [ ] outputs are committed or archived with stable paths
- [ ] evidence label is `measured_now` for headline claims
- [ ] blocker notes are explicit for missing sections
- [ ] conclusions distinguish "supports" vs "suggests"

If any item is false, treat the section as exploratory.

---

## E) Immediate integration points in this repo

1. Apply evidence labels to oracle/adaptive policy result summaries.
2. Add frontier-style summary rows to strategy and policy comparison docs.
3. Use the paper-readiness gate before drafting claim-heavy result sections.

---

## F) What is intentionally not transferred

- Domain-specific schema/slot grounding machinery.
- NLP4LP-specific constrained assignment internals.

These are not aligned with adaptive test-time compute routing goals.
