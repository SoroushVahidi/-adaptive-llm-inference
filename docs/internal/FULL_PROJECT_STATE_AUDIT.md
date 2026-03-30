# Full project state audit (repository-grounded)

**Audit date:** 2026-03-30  
**Method:** Inspection of files under `/workspace` only (code, docs, tracked `data/` and `outputs/` exceptions in `.gitignore`). No external API calls or re-runs for this document.

**Not found in repo (requested but absent):**

- `docs/PAPER_ARTIFACT_GENERATION_STATUS.md`
- `docs/EXPERIMENT_ARTIFACT_STATUS.md`
- `outputs/paper_tables/` (directory does not exist)
- `outputs/paper_figures/` (directory does not exist)
- `outputs/multi_action_models/` (directory does not exist at audit time)
- `data/multi_action_routing_*.csv` (no multi-action routing CSVs present)

**Tests:** `pytest` → **617 passed** (run 2026-03-30).

---

## 1. High-level understanding

**Problem.** The project studies **adaptive test-time compute allocation** for LLM reasoning under a fixed inference budget: how to spend samples, tokens, or escalation steps across queries (or actions) to maximize quality. Routing between a cheap reasoning path and a more expensive corrective path (e.g. direct + revise) is one concrete instantiation documented in real-data scripts and evaluation modules.

**System structure (as implemented in code).** The docs in `docs/PROJECT_CONTEXT.md` describe a pipeline **datasets → models → baselines/allocators → evaluation**. The repository extends this with **offline “oracle” preparation** for routing: scripts load or select queries (`src/datasets/*`, `scripts/run_build_*`), call an API model via `src/data/build_real_routing_dataset.py` / `src/models/openai_llm.py`, write **per-query CSVs** under `data/real_*_routing_dataset*.csv` and **run logs** under `outputs/real_*_routing/`, then **train shallow classifiers** and simulate policies (`scripts/run_real_routing_model_eval.py`, `src/evaluation/real_routing_model_eval.py`) and compare adaptive policies (`scripts/run_real_policy_eval.py`, `src/policies/adaptive_policy_v*.py`). **Paper-ready table/figure export directories are not present** (see §6). Multi-action oracle expansion is implemented in code and documented in `docs/MULTI_ACTION_ROUTING_RESULTS.md`, but **no corresponding artifacts** exist in this workspace snapshot (see §5).

---

## 2. Repository structure and main components

### Key directories

| Path | Role |
|------|------|
| `src/datasets/` | GSM8K, MATH500, AIME 2024, GPQA loaders and routing dataset readers |
| `src/data/` | `build_real_routing_dataset.py` — core real API dataset build |
| `src/models/` | Dummy + OpenAI LLM, revise-helpful classifier |
| `src/baselines/` | Greedy, best-of-N, self-consistency; external stubs under `external/` |
| `src/allocators/` | Equal budget, MCKP-style pieces |
| `src/evaluation/` | Simulated eval, oracle subset, real routing model eval, adaptive policy eval, multi-action routing helpers |
| `src/features/`, `src/policies/`, `src/analysis/` | Precomputed features, adaptive policies v1–v7, analysis utilities |
| `scripts/` | CLI entry points for builds, sweeps, baselines, policy eval (52 scripts) |
| `configs/` | YAML for `run_experiment.py` and oracle configs |
| `data/` | Gitignored except whitelisted real routing CSVs + `gpqa_diamond_normalized.jsonl` (see `.gitignore`) |
| `outputs/` | Gitignored except whitelisted experiment outputs (see `.gitignore`) |
| `docs/` | Research notes, methodology, per-dataset result reports |
| `tests/` | 617 tests including routing, GPQA loader, baselines |

### Main pipelines (evidence in repo)

1. **Simulated allocation / knapsack-style experiments:** `scripts/run_simulated_allocation.py`, `run_simulated_sweep.py`, `run_simulated_multi_seed.py`, `src/evaluation/simulated_*`.
2. **Binary real routing (reasoning vs revise):** `scripts/run_build_real_routing_dataset.py` (GSM8K 100), `run_build_hard_gsm8k_routing_dataset.py`, `run_build_math500_routing_dataset.py`, `run_build_aime_routing_dataset.py` → CSVs in `data/` + summaries in `outputs/real_*_routing/`.
3. **Oracle metrics (paired upper bound):** JSON summaries under `outputs/oracle_routing_eval/*.json` (used by cross-regime tooling).
4. **Routing model + simulation:** `scripts/run_real_routing_model_eval.py` → `outputs/real_*_routing_model/summary.json`, `routing_simulation.csv`, etc.
5. **Adaptive policy eval:** `scripts/run_real_policy_eval.py`, `run_adaptive_policy_v6_eval.py`, `run_adaptive_policy_v7_eval.py` → `outputs/real_*_policy_eval/`, `outputs/adaptive_policy_v7/`.
6. **Multi-action routing:** `scripts/run_build_multi_action_dataset.py`, `run_multi_action_model_eval.py` (code present); **artifacts missing** in this tree (§5).
7. **Paper artifacts:** **No** `outputs/paper_tables/` or `outputs/paper_figures/`; aggregation via CSV/JSON elsewhere (`scripts/summarize_simulated_results.py`, `run_final_cross_regime_summary.py`).

### Representative scripts (non-exhaustive)

| Script | Purpose |
|--------|---------|
| `scripts/run_experiment.py` | Config-driven baseline + allocator experiments (dummy or real backends) |
| `scripts/run_build_real_routing_dataset.py` | Build GSM8K real routing dataset (100-query flow documented in header) |
| `scripts/run_build_hard_gsm8k_routing_dataset.py` | Hard GSM8K slice from `outputs/hard_regime_selection/` CSV |
| `scripts/run_build_math500_routing_dataset.py` | MATH500 routing build (`dataset="math500"`, optional `data/math500_uploaded_normalized.jsonl`) |
| `scripts/run_build_aime_routing_dataset.py` | AIME 2024 via `src/datasets/aime2024.py` + `build_real_routing_dataset` |
| `scripts/run_real_routing_model_eval.py` | Train/eval revise-helpful-style routing models on enriched CSVs |
| `scripts/run_multi_action_model_eval.py` | Multi-action classifiers + policy simulation (expects `qf__`/`fp__` columns in CSV) |
| `scripts/run_cross_regime_comparison.py` / `run_final_cross_regime_summary.py` | Cross-dataset summaries → `outputs/cross_regime_comparison/` |
| `scripts/run_next_stage_baselines.py` | Stronger baseline / next-stage eval → `outputs/next_stage_eval/` |
| `scripts/run_oracle_subset_eval.py` | Documented in `docs/ORACLE_ANALYSIS_SUMMARY.md`; **no** `outputs/oracle_subset_eval/` in workspace |

---

## 3. Data inventory

Counts below use **Python `csv.DictReader`** row counts (handles multiline fields). **Line counts** from `wc -l` on these CSVs are inflated by embedded newlines in `reasoning_raw`.

| Dataset | Storage (paths) | Generation (scripts) | Rows (verified) | Routing usability |
|---------|------------------|----------------------|-----------------|-------------------|
| **Hard GSM8K (large / 100-query run)** | `data/real_hard_gsm8k_routing_dataset.csv`, `data/real_hard_gsm8k_routing_dataset_enriched.csv` | `scripts/run_build_hard_gsm8k_routing_dataset.py` (queries from `outputs/hard_regime_selection/hard_gsm8k_selection.csv`) | **100** | **Usable:** `reasoning_correct` vs `revise_correct` disagree on **17/100** (17%); `revise_helpful` true on **12/100** in enriched CSV — non-degenerate binary label |
| **MATH500 (100)** | `data/real_math500_routing_dataset.csv`, `..._enriched.csv` | `scripts/run_build_math500_routing_dataset.py` | **100** | **Usable:** disagree **12/100** (12%); `revise_helpful` **6/100** |
| **GSM8K random 100 (paired)** | `data/real_gsm8k_routing_dataset.csv`, `..._enriched.csv` | `scripts/run_build_real_routing_dataset.py` with `--paired-outcomes` (per `run_build_real_routing_dataset.py` docs) | **100** | **Weak for routing signal:** disagree **2/100** (2%); `revise_helpful` **2/100** — mostly single-action optimal |
| **AIME 2024** | `data/real_aime2024_routing_dataset.csv` | `scripts/run_build_aime_routing_dataset.py` (`HuggingFaceH4/aime_2024` via `src/datasets/aime2024.py`) | **30** | **Very weak:** disagree **2/30** (6.7%); `revise_helpful` **0/30** per `outputs/real_aime2024_routing/aime_run_summary.json` — oracle has no revise benefit in logged summary |
| **GPQA Diamond (normalized)** | `data/gpqa_diamond_normalized.jsonl` | Documented regeneration via `write_normalized_gpqa_jsonl()` in `src/datasets/gpqa.py` (`docs/GPQA_ACCESS_RECHECK.md`) | **198** lines | **MCQ labels present** (`choices`, `answer` A–D index in sample rows) — suitable for routing **if** LLM eval is run; **no** `outputs/real_*gpqa*` or routing CSV found in repo |
| **Hard GSM8K B2** | `data/real_hard_gsm8k_b2_routing_dataset.csv`, `..._enriched.csv` | Selection pipeline `outputs/hard_regime_selection_b2/` + build script pattern (same family as B1) | **100** rows (CSV) | Parallel track to hard GSM8K; model eval exists (`outputs/real_hard_gsm8k_b2_routing_model/summary.json`, 100 rows) |

**Additional:** Cached GSM8K HF metadata path exists: `data/openai___gsm8k/.../dataset_info.json` (HF cache layout).

---

## 4. Oracle and disagreement analysis

### Binary routing (reasoning vs revise) — computed from CSVs

| Dataset | `reasoning_correct` vs `revise_correct` disagreement | Notes |
|---------|--------------------------------------------------------|--------|
| Hard GSM8K (100) | 17 / 100 = **0.17** | Substantial pairwise diversity |
| MATH500 (100) | 12 / 100 = **0.12** | Moderate |
| AIME 2024 (30) | 2 / 30 ≈ **0.067** | Low |
| GSM8K random 100 | 2 / 100 = **0.02** | Very low |

### Oracle summaries (JSON files)

| File | Content (verbatim from JSON) |
|------|------------------------------|
| `outputs/oracle_routing_eval/hard_gsm8k_100_oracle_summary.json` | `accuracy` 0.91, `avg_cost` 1.12, `revise_rate` 0.12, `n` 100 |
| `outputs/oracle_routing_eval/math500_100_oracle_summary.json` | `accuracy` 0.7, `avg_cost` 1.06, `revise_rate` 0.06, `n` 100 |
| `outputs/oracle_routing_eval/gsm8k_random100_oracle_summary.json` | `accuracy` 0.92, `avg_cost` 1.02, `revise_rate` 0.02, `n` 100 |
| `outputs/oracle_routing_eval/hard_gsm8k_b2_oracle_summary.json` | `accuracy` 0.92, `avg_cost` 1.09, `revise_rate` 0.09, `n` 100 |

**AIME:** `outputs/real_aime2024_routing/aime_run_summary.json` gives `reasoning_accuracy` 0.133…, `revise_accuracy` 0.066…, `revise_helpful_rate` **0.0**, `reasoning_then_revise_accuracy` 0.166… — **no meaningful “revise helps” diversity** for the logged binary oracle story.

### Multi-action oracle (documentary only)

`docs/MULTI_ACTION_ROUTING_RESULTS.md` states that for **n=15** GSM8K hard tail and **n=12** MATH500, **`best_accuracy_action` was always `reasoning_greedy`** (full ties on correctness across four actions). **This audit did not re-verify** those JSON/CSV paths because **`outputs/multi_action_oracle/` and `data/multi_action_routing_*.csv` are absent** from the workspace.

### GSM8K oracle subset (20 queries)

`docs/ORACLE_ANALYSIS_SUMMARY.md` reports rich multi-strategy oracle results (oracle accuracy 0.75, etc.) and cites `outputs/oracle_subset_eval/` — **that directory is not present** in this repository snapshot.

---

## 5. Model training and evaluation status

### Binary routing models (`run_real_routing_model_eval.py` → `outputs/real_*_routing_model/summary.json`)

| Dataset | `summary.json` | `num_rows` | `num_positive` | Training | Notes |
|---------|----------------|------------|----------------|----------|--------|
| Hard GSM8K | `outputs/real_hard_gsm8k_routing_model/summary.json` | 100 | 12 | **OK** (`run_status`) | Best bagging F1 0.69; precision/recall moderate |
| MATH500 | `outputs/real_math500_routing_model/summary.json` | 100 | 6 | **OK** | Rare positive class; **bagging_trees** F1 **0.0** in summary |
| GSM8K 100 | `outputs/real_routing_model/summary.json` | 100 | 2 | **OK** | **All listed models F1 0.0** — extreme imbalance |
| Hard GSM8K B2 | `outputs/real_hard_gsm8k_b2_routing_model/summary.json` | 100 | (in file) | **OK** | Parallel track |

**Scientific strength:** Hard GSM8K is the only slice where both **disagreement** and **positive prevalence** support a non-trivial learned router; MATH500 is **borderline** (6 positives); random GSM8K is **very weak** for learning.

### Multi-action (`scripts/run_multi_action_model_eval.py`)

- **Script behavior:** Trains classifiers only when **≥2 classes** per label target; otherwise `run_status: SKIPPED` with reason in JSON (see lines 318–334).
- **Workspace state:** **No** `outputs/multi_action_models/*.json` and **no** input CSVs under `data/multi_action_routing_*.csv` → **no multi-action training/eval artifacts to report** for this audit.
- **Doc vs code:** `docs/MULTI_ACTION_ROUTING_RESULTS.md` mentions stderr text `Label ... has fewer than 2 classes`; current `run_multi_action_model_eval.py` prints **`[WARN] ... only one class`** — minor **documentation inconsistency**.

---

## 6. Paper artifact status

| Item | Status |
|------|--------|
| `outputs/paper_tables/` | **Absent** |
| `outputs/paper_figures/` | **Absent** |
| `docs/PAPER_ARTIFACT_GENERATION_STATUS.md` | **Not in repository** |
| `docs/EXPERIMENT_ARTIFACT_STATUS.md` | **Not in repository** |
| `docs/MAIN_BRANCH_STATUS_AUDIT.md` | Confirms missing paper dirs (grep-backed) |

**Upstream artifacts that exist and could feed tables:** `outputs/cross_regime_comparison/final_cross_regime_summary.csv`, `outputs/next_stage_eval/**`, `outputs/budget_sweep/**`, `outputs/baselines/*.json`. **Strict LaTeX/table export** and **strict mode** for paper builds are **not evidenced** by dedicated scripts or directories.

**`outputs/cross_regime_comparison/final_cross_regime_summary.csv` anomaly:** The `aime2024` row has **empty** `best_policy_accuracy`, `best_policy_cost`, `best_policy_name`, `revise_helpful_rate` (while other columns are filled). That suggests **incomplete aggregation** for that regime in the checked artifact.

---

## 7. Current strengths (top 5)

1. **End-to-end real routing pipeline** for GSM8K / hard GSM8K / MATH500 / AIME with **checked-in** CSVs and JSON summaries (`data/real_*`, `outputs/real_*_routing/`).
2. **Hard GSM8K (100)** shows **meaningful pairwise disagreement (17%)** and **non-trivial oracle** (0.91 accuracy, 0.12 revise rate in `oracle_routing_eval`) plus **routing model `run_status: OK`**.
3. **Rich evaluation surface:** adaptive policies v5–v7, cross-regime comparison, next-stage eval, budget sweeps, baseline summaries — **many JSON/CSV artifacts** under whitelisted `outputs/`.
4. **GPQA loader + 198-question normalized JSONL** (`data/gpqa_diamond_normalized.jsonl`, `src/datasets/gpqa.py`, tests in `tests/test_gpqa_loader.py`) — **strong potential** for a second domain.
5. **Reproducibility tooling:** config-driven experiments, documented build scripts, **617 passing tests**, clear `.gitignore` allowlist for key datasets/outputs.

---

## 8. Current weaknesses / problems (top 10)

1. **No paper_tables / paper_figures** — no evidenced automated export path for camera-ready artifacts.
2. **Random GSM8K 100** — **2%** reasoning/revise disagreement and **F1 0** routing models → **weak scientific case** for routing on easy random slice.
3. **AIME slice (30)** — very low accuracies; **revise_helpful_rate 0**; **no policy columns** in cross-regime CSV → **control / stress case**, not main evidence.
4. **MATH500 routing model** — severe class imbalance (6 positives); one family **F1 0** in `summary.json`.
5. **Multi-action track** — **no artifacts** in tree; docs describe **degenerate oracle labels** on small slices.
6. **Missing `outputs/oracle_subset_eval/`** while `docs/ORACLE_ANALYSIS_SUMMARY.md` references it — **doc/artifact mismatch**.
7. **`docs/GPQA_ACCESS_CHECK.md` vs `docs/GPQA_ACCESS_RECHECK.md`** — contradictory **gating** narrative over time; **both files exist** (resolve in prose for readers).
8. **`real_routing_dataset/`** partial snapshot (e.g. `gsm8k_per_query_outputs.csv`) — may not match **full** script doc layout (`routing_dataset.csv` not listed in glob).
9. **External baselines** (TALE, BEST-Route) still **stubs** per `docs/BASELINE_TRACKER.md`.
10. **Cross-regime CSV** incomplete cells for **aime2024**.

---

## 9. Blockers

| Blocker | Evidence |
|---------|----------|
| **OpenAI API** | All `outputs/real_*_routing/*_run_summary.json` list `"provider": "openai"`, `"model_name": "gpt-4o-mini"` — rebuilding real datasets requires **`OPENAI_API_KEY`** and network |
| **HuggingFace / network** | GSM8K auto-download (`docs/PROJECT_CONTEXT.md`); MATH500 build optionally uses HF / uploaded JSONL; AIME uses HF dataset |
| **GPQA Hub access** | `docs/GPQA_ACCESS_CHECK.md` records past gating failure; `docs/GPQA_ACCESS_RECHECK.md` records **success with config name** — **environment-dependent**; normalized file **is** in repo for offline MCQ text |
| **Missing multi-action CSVs** | Cannot run `run_multi_action_model_eval.py` without generating `data/multi_action_routing_*.csv` first |
| **Paper export** | No strict pipeline — **not a runtime blocker** but blocks **paper-ready** automation |

---

## 10. Done vs remains

### A. Fully implemented and working (in repo + tests pass)

- Core package install layout, dummy model experiments, native baselines, allocators.
- Real routing dataset builders for GSM8K / hard GSM8K / MATH500 / AIME.
- Binary routing model eval and adaptive policy eval (v5–v7) with stored outputs.
- Simulated evaluation scripts and cross-regime / next-stage summarization (artifacts present).
- GPQA normalized JSONL + loader tests.

### B. Implemented but weak / not paper-ready

- Learned routing on **random GSM8K 100** (labels nearly constant).
- **AIME** slice as **routing** evidence (revise never helpful in summary; n=30).
- **MATH500** classifier metrics (imbalance; weak bagging).
- **Multi-action** code path without non-degenerate labels in **this** workspace.
- **“Paper artifacts”** as publishable tables/figures from a dedicated export.

### C. Partially implemented

- External baseline wrappers (stubs).
- MCKP / richer allocators vs full baseline comparisons from `BASELINE_TRACKER.md`.
- `outputs/real_routing_dataset/` vs documented full file set.
- Cross-regime summary for **all** fields per regime.

### D. Missing or not yet attempted (in this tree)

- `outputs/paper_tables/`, `outputs/paper_figures/`.
- `docs/PAPER_ARTIFACT_GENERATION_STATUS.md`, `docs/EXPERIMENT_ARTIFACT_STATUS.md` (requested names absent).
- GPQA **end-to-end routing runs** (no `outputs/*gpqa*` found).
- Multi-action **data/results files** on disk.
- `outputs/oracle_subset_eval/` bundle referenced by oracle analysis doc.

---

## 11. Overall project assessment

| Question | Answer |
|----------|--------|
| **Pipeline end-to-end?** | **Yes** for the **binary real routing** path: build scripts → `data/real_*` → policy/model eval → `outputs/real_*` and comparison CSVs. **No** for **multi-action** and **paper export** (no artifacts / dirs). |
| **Results strong enough for a paper?** | **Partially.** Strongest quantitative story is **hard GSM8K** (disagreement + oracle + OK model). Random GSM8K and AIME are **weak** for routing claims. MATH500 is **supporting** but imbalanced. Missing strong baselines (Snell, TALE integrated runs) and formal paper tables. |
| **Strongest dataset (evidence)?** | **Hard GSM8K (100)** — `data/real_hard_gsm8k_routing_dataset_enriched.csv` + `outputs/oracle_routing_eval/hard_gsm8k_100_oracle_summary.json` + `outputs/real_hard_gsm8k_routing_model/summary.json`. |
| **Weakest / control cases?** | **GSM8K random 100** (2% disagree) and **AIME 2024** (30 queries, revise_helpful 0) for **routing**; **multi-action small slices** per doc (not on disk here). |

---

## 12. Next-step recommendations (prioritized)

### Experiments

1. **Regenerate multi-action datasets at n≥100** with `scripts/run_build_multi_action_dataset.py` — **why:** enables non-degenerate labels for `scripts/run_multi_action_model_eval.py`. **API/internet:** **yes** (OpenAI + data fetch).
2. **Run GPQA routing build** (new script or adapt `build_real_routing_dataset` for MCQ) using `data/gpqa_diamond_normalized.jsonl` — **why:** second domain with rich labels. **API:** **yes**; **HF:** optional if using normalized JSONL only.
3. **Expand AIME or drop from main claims** — **why:** n=30 and zero revise helpful limits inference. **API + HF:** **yes** if expanding.

### Evaluation improvements

4. **Fix `final_cross_regime_summary.csv` for aime2024** — **why:** complete aggregation; **files:** `scripts/run_final_cross_regime_summary.py`, `scripts/run_cross_regime_comparison.py`. **API:** **no** if only postprocessing existing CSVs.
5. **Re-run oracle subset eval and commit outputs** or **update doc paths** — **why:** remove mismatch between `docs/ORACLE_ANALYSIS_SUMMARY.md` and tree. **API:** depends on rerun.

### Data improvements

6. **Stratify MATH500 / hard GSM8K** for more `revise_helpful` positives — **why:** stabilize classifiers. **Scripts:** same build scripts with selection logic in `src/data/hard_gsm8k_selection.py`. **API:** **yes**.

### Writing / paper preparation

7. **Add `outputs/paper_tables/` + export script** consuming `outputs/cross_regime_comparison/final_cross_regime_summary.csv` and `outputs/next_stage_eval/**` — **why:** reproducible tables. **API:** **no**.
8. **Author `docs/PAPER_ARTIFACT_GENERATION_STATUS.md`** tracking figure/table provenance — **why:** single source of truth (file currently missing).
9. **Integrate one external baseline** (TALE or BEST-Route) per `docs/BASELINE_TRACKER.md` — **why:** competitive positioning. **Deps:** clone + possibly GPU; **mixed**.
10. **Reconcile GPQA access docs** (`GPQA_ACCESS_CHECK.md` vs `GPQA_ACCESS_RECHECK.md`) — **why:** avoid contradictory onboarding. **API:** **no**.

---

## 13. Inconsistencies checklist

- Oracle subset doc vs missing `outputs/oracle_subset_eval/`.
- Multi-action doc stderr message vs current script message.
- GPQA gating described differently in two access docs (historical vs recheck).
- `final_cross_regime_summary.csv` missing policy fields for `aime2024`.
- `PROJECT_CONTEXT.md` “Stage 2 planned” for MATH500 vs **implemented** MATH500 routing artifacts in repo.

---

*End of audit.*
