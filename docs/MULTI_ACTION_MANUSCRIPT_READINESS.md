# Multi-action routing — manuscript readiness

## Workspace audit (repository-grounded)

The following paths from the task description **are not present** in this repository snapshot:

| Expected path | Status |
|---------------|--------|
| `docs/MULTI_ACTION_DATA_EXPANSION_REPORT.md` | **Missing** |
| `docs/MULTI_ACTION_FEATURE_ANALYSIS.md` | **Missing** |
| `docs/MULTI_ACTION_ROUTING_RESULTS.md` | **Missing** |
| `scripts/run_build_multi_action_dataset.py` | **Missing** |
| `scripts/run_multi_action_model_eval.py` | **Missing** |
| `src/evaluation/multi_action_routing.py` | **Missing** |

Therefore **none** of the user-reported figures below can be verified from files in this workspace:

- Dataset sizes `hard_gsm8k_large (100)`, `math500_large (100)`, `aime2024 (30)`, GPQA via `aradhye/gpqa_diamond`
- Disagreement rates 48% / 7% / 2% / 0%
- Which runs trained classifiers or skipped AIME for single-class labels

**What exists instead** (grounding for “multi-action” style analysis):

- **Compute-ladder disagreement** — `src/evaluation/strong_baselines_eval.py` → `compute_disagreement_analysis` → JSON under `outputs/baselines/*_disagreement_analysis.json` when `run_strong_baselines.py` has been run.
- **Oracle over multiple actions** — `src/evaluation/recent_baselines_eval.py` → `compute_oracle_summaries` → `multi_action_oracle` in experiment outputs (when `run_recent_baselines_experiment.py` is used).
- **GPQA loading** — `src/datasets/gpqa.py` (official `Idavidrein/gpqa` / fallback `hendrydong/gpqa_diamond_mc`; **no** reference to `aradhye/gpqa_diamond` in code).

**Current workspace artifacts** (`outputs/baselines/*_disagreement_analysis.json`) reflect a **dummy-model** strong-baselines smoke run (`run_log.json`: `model_type: dummy`, `n_queries` 20/20/2), not the user’s described large-scale study:

| Dataset (artifact) | `n_queries` | `pct_at_least_two_actions_differ` | Notes |
|--------------------|-------------|-------------------------------------|--------|
| gsm8k | 20 | 0.05 | Mostly `none_correct` under dummy |
| hard_gsm8k | 20 | 0.05 | Same pattern |
| math500 | 2 | 0.50 | **Too small** to interpret |

**Conclusion for this repo state:** multi-action routing is **not** manuscript-ready here; only plumbing-level disagreement JSON exists from a toy run.

---

## A. Strongest empirical story (if your external results match the task description)

*Conditional on* artifacts that match the user narrative (high GPQA disagreement, trained routers on GPQA / MATH500 / hard GSM8K, degenerate AIME):

- **GPQA** is the natural **centerpiece** for “adaptive routing has headroom”: high action disagreement implies non-trivial oracle routing value; MCQ format aligns with baseline families in `PROJECT_CONTEXT.md`.
- **Hard GSM8K / MATH500 large** can **support** a breadth claim only if learned routing **beats** fixed baselines by a **clear margin** on accuracy–cost curves (not yet verifiable here).
- **AIME** at **0% disagreement** is a **negative/control**: either drop from routing claims or reframe as “single optimal action on this ladder slice” (not evidence for routing).

---

## B. Weakest points / risks

1. **No reproducible bundle in repo** — reviewers cannot align claims with committed scripts/outputs until the multi-action pipeline is merged with docs + `outputs/` or a results archive.
2. **GPQA provenance** — Loader documents **official** vs **`hendrydong/gpqa_diamond_mc`** fallback; **`aradhye/gpqa_diamond`** is not referenced in `src/datasets/gpqa.py`. Any mirror must be documented (schema parity, license, overlap with official diamond).
3. **“Training ran” ≠ paper evidence** — Need **held-out** or **cross-validated** routing metrics vs **always-action-X** baselines; code path for that evaluation is **missing** in this snapshot.
4. **Cost/latency** — Strong baselines use **cost proxy** (`LADDER_METHOD_COST`); real latency is **not** in the disagreement JSON. Manuscript needs explicit cost model or abstain from latency claims.
5. **Single-class / degenerate labels** — If oracle best-action collapses (AIME), routing is **ill-posed**; keeping it in a table without reframing invites criticism.

---

## C. Minimum next experiments (before freezing multi-action claims)

Prioritized for **evidence strength**, assuming the new pipeline exists locally outside this snapshot:

| Priority | Experiment | Why | Needs API / HF |
|----------|------------|-----|----------------|
| 1 | **GPQA full ladder + disagreement + learned router eval** on ≥100 items (or full diamond N≈198) with **train/val/test** split and **vs** always-greedy / always-SC / oracle upper bound | Centers the positive case with statistical heft | Yes (OpenAI); HF if loading from Hub |
| 2 | **Cost–accuracy curves** for GPQA (λ sweeps or fixed budget), same as `recent_baselines_eval` utility oracles conceptually | Separates “pipeline works” from “routing improves utility” | Yes |
| 3 | **Stronger GPQA baselines** (e.g. higher-N self-consistency only if justified; document cost) | EAAI framing expects competitive baselines (`PROJECT_CONTEXT.md`) | Yes |
| 4 | **MATH500 or hard GSM8K**: one targeted run with **richer action set** or **harder slice** if disagreement stays ~7% / 2% — else **do not** claim routing gains | Avoids empty positive case | Yes |
| 5 | **AIME**: **drop** from routing headline **or** change action menu so disagreement is non-zero **or** report as **control** only | Prevents false-positive routing narrative | Yes |

**Repo hygiene:** Add the missing `MULTI_ACTION_*.md` reports, scripts, and `src/evaluation/multi_action_routing.py` (or equivalent) so results are **traceable**.

---

## D. Should GPQA be the centerpiece?

**Yes, conditionally:** if (and only if) held-out evaluation shows **learned routing** beating **credible fixed baselines** with **documented cost**, and GPQA mirror/official provenance is spelled out for reviewers.

**No:** if the only signal is high disagreement on **train** without generalization, or gains vs baselines are **within noise**.

---

## Decision table (labels apply to **user-described** setup; workspace = NOT READY)

| Dataset | Label (if user narrative accurate) | Label (this workspace) |
|---------|-------------------------------------|-------------------------|
| GPQA | **CORE POSITIVE EVIDENCE** (pending rigorous eval) | **NOT READY** (no artifacts) |
| MATH500 large | **SUPPORTING BUT WEAK** at ~7% disagreement unless routing beats baselines clearly | **NOT READY** |
| Hard GSM8K large | **SUPPORTING BUT WEAK** / trending **NEGATIVE/CONTROL** at ~2% disagreement | **NOT READY** |
| AIME 2024 | **NEGATIVE/CONTROL** at 0% disagreement; skip or reframe | **NOT READY** |

---

## Learned-routing checklist (when code exists)

For each trained dataset, a manuscript needs:

1. **Metrics:** accuracy, avg cost proxy (or tokens), vs **fixed** baselines on the **same** queries.
2. **Significance / variance:** multiple seeds or CI (even simple bootstrap) if N < 200.
3. **Oracle gap:** how close routing gets to `multi_action_oracle` upper bound (pattern from `recent_baselines_eval.py`).
4. **Failure cases:** where router underperforms (e.g. degenerate slices).

*None of these can be filled from the current workspace for the new pipeline.*
