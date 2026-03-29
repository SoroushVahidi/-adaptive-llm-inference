# State of Evidence Audit (Repository-grounded)

**Date:** 2026-03-28  
**Scope:** Full repository audit focused on scientific evidence status, not method redesign.

---

## Evidence legend used in this audit

- **A — Code only:** implemented and testable in code, but no recorded experiment outputs committed.
- **B — Recorded outputs/results:** numeric results backed by committed output files (CSV/JSON) in repo.
- **C — Claimed in docs/comments:** stated in docs but not backed by committed outputs in repo.
- **D — Unknown:** cannot be verified from current files.

---

## 1) Full repository audit

### `src/datasets`

- **Present:** GSM8K loader (`gsm8k.py`), MATH500 loader (`math500.py`), synthetic allocation instance generator (`synthetic_ttc.py`), routing dataset assembler (`routing_dataset.py`), bundled GSM8K sample JSON. **Evidence: A**.
- **Why it matters:** This defines what data regimes are currently reachable offline (bundled sample + synthetic) vs requiring HuggingFace/network/OpenAI runs. **Evidence: A**.

### `src/evaluation`

- **Present:** Strategy diagnostics, oracle subset evaluation, adaptive policy v1–v4 evaluators, real-budget and real-gain analyses, simulated analysis/aggregate, logger/metrics. **Evidence: A**.
- **Why it matters:** The experiment logic is broad and modular, but the existence of evaluators is not equivalent to completed experiments. **Evidence: A**.

### `src/features`

- **Present:** Query-only features (`precompute_features.py`), target-quantity/wording-trap features (`target_quantity_features.py`), constraint-aware violation features (`constraint_violation_features.py`). **Evidence: A**.
- **Why it matters:** Feature engineering coverage is substantial and directly supports routing hypotheses, but impact claims require run outputs that are currently absent. **Evidence: A**.

### `src/policies`

- **Present:** Rule-based adaptive policies v1–v4 and a router baseline module (`router_baseline.py`). **Evidence: A**.
- **Why it matters:** Multiple policy generations exist and are unit-tested; however, empirical policy quality depends on oracle-labeled data that is not committed. **Evidence: A**.

### `src/strategies`

- **Present:** `action_catalog.py` loader/validator for strategy catalog; execution logic actually lives under `src/evaluation/strategy_expansion_eval.py` and `expanded_strategy_eval.py`. **Evidence: A**.
- **Why it matters:** Strategy taxonomy is well-documented; implemented-vs-placeholder distinction is explicit in config/catalog. **Evidence: A/C**.

### `src/analysis`

- **Present:** Feature-gap analysis and revise-help analysis modules with graceful handling of missing oracle files. **Evidence: A**.
- **Why it matters:** Analysis pathways are prepared for decision-making, but with missing oracle outputs they fall back to schema/placeholder analysis. **Evidence: A**.

### `scripts`

- **Present:** Scripts for experiment running across oracle subset, strategy diagnostics, model sampling diagnostic, adaptive policy v1–v4, routing dataset build, router baseline, feature-gap/revise-help analyses, simulated sweeps, etc. **Evidence: A**.
- **Why it matters:** Operational surface is large; most real-LLM scripts explicitly require `OPENAI_API_KEY` and can emit blocked sentinel outputs. **Evidence: A**.

### `configs`

- **Present:** Configs for dummy baselines, simulated allocation sweeps, GSM8K oracle subset, MATH500 diagnostic, model sampling diagnostic, routing/policy evaluations. **Evidence: A**.
- **Why it matters:** Intended experiments are clearly specified and reproducible in principle; completion status depends on outputs. **Evidence: A**.

### `docs`

- **Present:** Rich research/process docs (project context, baseline tracker, action space, policy docs, routing docs, oracle summaries, feature analyses). **Evidence: C**.
- **Why it matters:** Documentation includes both design intent and some numeric claims; some docs explicitly mark runs as BLOCKED while another doc reports numeric oracle results, creating an internal consistency risk without committed output artifacts. **Evidence: C/D**.

### `outputs`

- **Current state in repository:** `outputs/` directory is absent in repo snapshot (gitignored; no committed result artifacts found). **Evidence: D for empirical result reproducibility; no B evidence available**.
- **Why it matters:** No committed CSV/JSON experiment outputs means most performance conclusions are not currently verifiable from repository files alone. **Evidence: B unavailable**.

### `tests`

- **Present:** Extensive unit tests covering baselines, strategy evaluators, oracle summary logic, adaptive policy v1–v4 logic, feature modules, routing dataset and router baseline, simulated allocators. **Evidence: A**.
- **Why it matters:** Strong implementation-level confidence; weak direct scientific-evidence confidence without committed run outputs. **Evidence: A**.

---

## 2) Dataset audit

| Dataset | Loader/module | Source path/source ID | Answer orientation | Recorded run evidence in repo |
|---|---|---|---|---|
| GSM8K | `src/datasets/gsm8k.py` | HF `openai/gsm8k` or local JSON via `data_file` | Numeric final answer extraction (`####`) | **No committed outputs found** (only docs/configs/scripts) |
| MATH500 | `src/datasets/math500.py` | HF `HuggingFaceH4/MATH-500` or local JSON | Symbolic/math answer normalization (`normalize_math_answer`) | **No committed outputs found** |
| Bundled/local sample | `src/datasets/bundled/gsm8k_test_sample.json` + `load_gsm8k(..., data_file=...)` | In-repo JSON sample | Numeric | Used by configs/docs/tests, but no committed `outputs/*` artifacts |
| Synthetic TTC | `src/datasets/synthetic_ttc.py` | Generated in code | Utility-curve allocation (not QA answer matching) | No committed output artifacts |

**Bottom line:** dataset support is real in code (**A**), but empirical results are not committed (**B absent**).

---

## 3) Strategy/policy audit (requested list)

| Strategy/policy | Implemented? | Tested offline? | Supported by committed empirical outputs? | Status class |
|---|---:|---:|---:|---|
| `direct_greedy` | Yes | Yes | No | **A** |
| `reasoning_greedy` | Yes | Yes | No | **A** |
| `reasoning_best_of_3` | Yes | Yes | No | **A** |
| `structured_sampling_3` | Yes | Yes | No | **A** |
| `direct_plus_verify` | Yes | Yes | No | **A** |
| `direct_plus_revise` | Yes | Yes | No | **A** |
| `direct_plus_critique_plus_final` | Yes | Yes | No | **A** |
| `first_pass_then_hint_guided_reason` | Yes | Yes | No | **A** |
| `strong_direct` | Yes (optional strong model path) | Yes (oracle eval logic) | No | **A** |
| `adaptive_policy_v1` | Yes | Yes | No committed outputs | **A** |
| `adaptive_policy_v2` | Yes | Yes | No committed outputs | **A** |
| `adaptive_policy_v3` | Yes | Yes | No committed outputs | **A** |
| `adaptive_policy_v4` | Yes | Yes | No committed outputs | **A** |

**Note on docs:** docs contain numerical narratives about some of these strategies (notably oracle subset summary), but without committed outputs those are **C**, not **B**.

---

## 4) Feature audit

### Query-only features

- **Where:** `src/features/precompute_features.py` (`extract_query_features`).
- **Used in evaluation:** routing dataset assembly + router baseline pipelines; reusable in analyses.
- **Evidence they helped:** no committed empirical output proving effect size.  
**Classification:** implemented and integrated (**A**), helpfulness unproven (**D/B absent**).

### Target-quantity / wording-trap features

- **Where:** `src/features/target_quantity_features.py`.
- **Used in evaluation:** feature-gap and revise-help analyses merge these with base features.
- **Evidence they helped:** docs describe directional promise and non-degenerate firing rates, but committed result artifacts are missing.  
**Classification:** implementation exists (**A**), performance claims mostly doc-level (**C**).

### Constraint-aware violation features

- **Where:** `src/features/constraint_violation_features.py` and adaptive policy v4 usage.
- **Used in evaluation:** adaptive policy v4 scoring + signal summaries in v4 evaluator outputs (path defined).
- **Evidence they helped:** no committed `outputs/adaptive_policy_v4/*` to verify gains.  
**Classification:** code-integrated (**A**), empirical benefit unresolved (**D/B absent**).

### First-pass-output features

- **Where:** `extract_first_pass_features` in `precompute_features.py`; policy-specific first-pass heuristics in adaptive policies.
- **Used in evaluation:** routing schema includes first-pass columns; policies use first-pass outputs for escalation decisions.
- **Evidence they helped:** no committed run outputs.  
**Classification:** implemented (**A**), efficacy unresolved (**D**).

---

## 5) Experiment/results audit (requested checks)

| Experiment family | Script/config | Dataset | Outputs present in repo? | Supported result now | Strength |
|---|---|---|---|---|---|
| GSM8K oracle subset evaluation | `scripts/run_oracle_subset_eval.py` + `configs/oracle_subset_eval_gsm8k.yaml` | GSM8K bundled subset | **No committed outputs** | Only that pipeline exists; docs include both BLOCKED status and separate numeric summary text | **Weak / conflicting (C/D)** |
| MATH500 strategy diagnostic | `scripts/run_strategy_diagnostic_math500.py` + `configs/strategy_diagnostic_math500.yaml` | MATH500 | No committed outputs | Implementation and config exist | **Code-only (A)** |
| Model sampling diagnostics | `scripts/run_model_sampling_diagnostic.py` + `configs/model_sampling_diagnostic_gsm8k.yaml` | GSM8K subset | No committed outputs | Implementation exists | **Code-only (A)** |
| Adaptive policy v1 | `scripts/run_adaptive_policy_eval.py` + `configs/adaptive_policy_gsm8k.yaml` | GSM8K subset | No committed outputs | Policy + evaluator implemented | **Code-only (A)** |
| Adaptive policy v2 | `scripts/run_adaptive_policy_v2_eval.py` + `configs/adaptive_policy_v2_gsm8k.yaml` | GSM8K subset | No committed outputs | Policy + evaluator implemented | **Code-only (A)** |
| Adaptive policy v3 | `scripts/run_adaptive_policy_v3_eval.py` + `configs/adaptive_policy_v3_gsm8k.yaml` | GSM8K subset | No committed outputs | Policy + sweep logic implemented | **Code-only (A)** |
| Adaptive policy v4 | `scripts/run_adaptive_policy_v4_eval.py` + `configs/adaptive_policy_v4_gsm8k.yaml` | GSM8K subset | No committed outputs | Policy + constraint-aware scoring implemented | **Code-only (A)** |
| Revise-case analysis | `scripts/run_revise_help_feature_analysis.py`, `scripts/run_feature_gap_analysis.py` | GSM8K sample + optional oracle files | No committed outputs | Analysis pipelines implemented with fallback mode | **Code-only (A)** |
| Feature-gap analysis | same as above | GSM8K sample + optional oracle files | No committed outputs | Fallback analysis possible without oracle labels | **Code-only (A)** |
| Routing-dataset preparation | `scripts/build_routing_dataset.py` + `src/datasets/routing_dataset.py` | GSM8K sample + optional oracle labels | No committed outputs | Schema/full mode implemented; robust missing-file handling | **Code-only (A)** |
| Router baselines | `scripts/run_router_baseline.py` + `src/policies/router_baseline.py` | Routing dataset | No committed outputs | Binary/multiclass baseline logic implemented; blocks without oracle labels | **Code-only (A)** |

---

## 6) Scientific status summary (direct answers)

### What is the strongest cheap baseline right now?

- **Most defensible from code readiness alone:** `reasoning_greedy` as the primary cheap comparator candidate, because it is first-class across diagnostics/policies and explicitly central in policy logic. **Evidence class: A/C**.
- **Empirical strongest (strict B):** **Unknown**, because no committed result artifacts. **Evidence class: D**.

### What is the strongest stronger/corrective baseline right now?

- **Code-level candidate:** `direct_plus_revise` (implemented end-to-end, repeatedly treated as corrective path in policy/eval design). **A/C**.
- **Empirically proven from committed outputs:** **Unknown** (**D**).

### Does extra sampling help enough to justify cost?

- **From committed outputs:** cannot conclude. **D**.
- **From documentation claims only:** suggested marginal benefit and possible poor cost-efficiency in some subsets. **C (not yet B-validated)**.

### Is structured sampling still worth keeping?

- **Strict evidence:** unresolved without committed outputs. **D**.
- **Doc-level narrative:** weak value relative to cheaper alternatives in cited subset. **C**.

### Have hand-crafted routing signals beaten `reasoning_greedy`?

- No committed outputs demonstrate this. **D**.

### Are target-quantity / constraint-aware signals directionally useful?

- **Code-level:** yes, nontrivial feature families implemented and wired into analyses/policy v4. **A**.
- **Outcome-level:** directional usefulness not yet established in committed experiment outputs. **D**.

### What is the main bottleneck now?

1. **Evidence bottleneck:** lack of committed experiment output artifacts (especially oracle-labeled runs).
2. **Access bottleneck:** many core experiments depend on OpenAI API access; docs already show blocked runs when key absent.

### Do we already have enough evidence to pause and think before doing more experiments?

- **Yes, to pause and reframe the question**, because there is enough design/code scaffolding to identify the core uncertainty.
- **No, for strong empirical claims**, because B-level evidence is sparse/absent in committed files.

---

## 7) Main open problem now

### Simple wording

Can we reliably detect, **before spending extra compute**, which queries actually need correction/escalation so that adaptive routing beats a strong one-pass reasoning baseline on accuracy-per-cost?

### Technical sentence

Given query and optional first-pass features \(z\), does there exist a calibrated routing policy \(\pi(z)\) that yields statistically reliable improvement in expected utility (accuracy under cost budget) over `reasoning_greedy` and other fixed-cost baselines on GSM8K/MATH500?

---

## 8) What is strong vs weak right now

### Strong (high confidence)

1. The repository has broad implementation coverage across strategies, policies, feature engineering, and evaluation plumbing.
2. Unit-test coverage is substantial and includes core routing/allocation logic.
3. The project is ready for controlled evaluation runs once model access and output archival discipline are in place.

### Weak (low confidence)

1. Claims about relative strategy quality/cost tradeoffs are mostly not backed by committed run artifacts.
2. Router/policy superiority claims over `reasoning_greedy` are currently unproven in committed evidence.
3. Some documentation-level numeric summaries cannot be independently verified against committed output files.

---

## 9) Main recommendation at this decision point

Treat the current state as **implementation-ready but evidence-light**. The highest-value next step is not adding more strategy variants; it is establishing a reproducible evidence base (committed summaries/artifacts and consistency checks) sufficient to answer whether routing signals produce real budgeted gains over strong cheap baselines.
