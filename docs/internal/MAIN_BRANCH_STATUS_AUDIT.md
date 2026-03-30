# Main branch status audit (repository-grounded)

**Audit date:** 2026-03-30 (workspace clock)  
**Evidence scope:** Files and git refs **present in this clone** only. No network was used to compare against GitHub beyond what is already in `remotes/origin/*`.

---

## Critical git fact (read first)

| Ref | Commit (short) | Meaning |
|-----|----------------|---------|
| `origin/main` | `f0b9d8c` | Full codebase (420 tracked files in this tree when on that commit). |
| Local branch `main` | `8ce0383` | **Only tracks `LICENSE`** — 1 file. This local ref is **not** aligned with `origin/main`. |
| `HEAD` (audit run) | `f0b9d8c` | Branch `cursor/main-branch-status-audit-e5ff`; **140 commits ahead** of local `main`, **0 behind** `origin/main`. |

**Implication:** Any statement “on main” in this report means **`origin/main` @ `f0b9d8c`** (the real mainline), not the stale local `main` ref. **Fix:** `git branch -f main origin/main` (or `git fetch` + reset) so local `main` matches remote.

**Merge conflict markers:** Ripgrep for `<<<<<<<`, `=======`, `>>>>>>>` across the repo returned **no matches**.

**Tests (evidence):** After `pip install -e ".[dev]"`, `pytest` reported **609 passed** (2026-03-30 in this environment).

---

## 1. High-level repository audit

### 1.1 Research tracks present in code

Grounded in `docs/PROJECT_CONTEXT.md`, `README.md`, and `scripts/`:

1. **Simulated allocation / MCKP** — synthetic utility tables, budget sweeps, noise analysis (`src/datasets/synthetic_ttc.py`, `src/evaluation/simulated_*.py`, `scripts/run_simulated_*.py`, `configs/simulated_*.yaml`).
2. **Oracle / multi-strategy evaluation** — GSM8K subset, strategy matrices (`src/evaluation/oracle_subset_eval.py`, `scripts/run_oracle_subset_eval.py`, `scripts/run_oracle_strategy_eval.py`).
3. **Real LLM routing datasets** — build per-query outputs + flat CSVs for GSM8K, MATH500, hard GSM8K, AIME (`scripts/run_build_*_routing_dataset.py`, `src/data/build_real_routing_dataset.py`, `src/datasets/routing_dataset.py`).
4. **Learned routing / policies** — (a) sklearn routers in `src/policies/router_baseline.py` + `scripts/run_real_routing_model_eval.py`; (b) tree ensembles on `revise_helpful` in `src/evaluation/real_routing_model_eval.py` (used by tests; **no dedicated `scripts/` entrypoint** that calls it — see §4).
5. **Adaptive policies v2–v7** — heuristic routing with offline configs (`src/policies/adaptive_policy_*.py`, `scripts/run_adaptive_policy_*_eval.py`, committed `outputs/adaptive_policy_v7/`).
6. **Strong / recent baselines** — compute ladder, routers, multi-action oracle summaries (`scripts/run_strong_baselines.py`, `scripts/run_recent_baselines_experiment.py`, `src/evaluation/strong_baselines_eval.py`, `src/evaluation/recent_baselines_eval.py`).
7. **Next-stage EAAI-style bundle** — oracle routing summaries, budget sweeps, cascade curves, cross-regime tables (`scripts/run_next_stage_*.py`, `run_reasoning_then_revise_addon.py`, `run_final_cross_regime_summary.py`, `outputs/next_stage_eval/`, `outputs/budget_sweep/`, `outputs/oracle_routing_eval/`).

### 1.2 Intended pipeline (as implemented in repo)

| Stage | Role | Primary paths |
|-------|------|----------------|
| Data | Loaders + optional normalized JSONL | `src/datasets/{gsm8k,math500,hard_gsm8k,aime2024,gpqa}.py`, `data/*.csv`, `data/gpqa_diamond_normalized.jsonl` |
| Oracle / adaptive eval | API or bundled samples | `scripts/run_oracle_subset_eval.py`, `run_oracle_strategy_eval.py`, `src/evaluation/oracle_subset_eval.py` |
| Routing dataset build | Real per-query CSV + summaries | `scripts/run_build_real_routing_dataset.py` (+ math500/hard/aime variants), `outputs/real_*_routing/` |
| Learned routing | Fit models + simulation CSVs | `src/evaluation/real_routing_model_eval.py`, `scripts/run_real_routing_model_eval.py` (sklearn path), committed `outputs/real_*_routing_model/` |
| Policies | Offline / CSV-based | `scripts/run_real_policy_eval.py`, `run_adaptive_policy_v*_eval.py` |
| Baselines | JSON summaries | `scripts/run_strong_baselines.py`, `run_recent_baselines_experiment.py`, `run_next_stage_baselines.py`, `outputs/baselines/` |
| Aggregation / “paper-adjacent” tables | CSV/JSON from existing outputs | `scripts/run_cross_regime_comparison.py`, `run_final_cross_regime_summary.py`, `summarize_simulated_results.py` |
| Docs | Status + results narratives | `docs/*.md` (many regime-specific reports) |

**Gap vs “paper tables/figures”:** There are **no** `outputs/paper_tables/` or `outputs/paper_figures/` directories and **no** matplotlib-based paper plot scripts under `scripts/` (search for `matplotlib`, `paper_tables`, `paper_figures` in `scripts/` returned nothing). Summarization is CSV/terminal-oriented (`summarize_simulated_results.py`).

---

## 2. Implementation status by component

Legend: **COMPLETE** = code + (where applicable) committed artifacts match intent; **PARTIAL** = code exists, artifacts or integration incomplete; **MISSING** = not in tree; **BLOCKED** = depends on API/secrets/artifacts.

| Component | Status on `origin/main` @ `f0b9d8c` | Evidence |
|-----------|--------------------------------------|----------|
| Simulated allocation / MCKP pipeline | **COMPLETE (code)** / **PARTIAL (committed artifacts)** | Code: `scripts/run_simulated_allocation.py`, `src/evaluation/simulated_evaluator.py`, `configs/simulated_mckp.yaml`. No `outputs/simulated_*` directories present in this clone. |
| Oracle subset evaluation | **COMPLETE (code)** / **BLOCKED (default run)** | `scripts/run_oracle_subset_eval.py` documents `OPENAI_API_KEY`; `docs/EXPERIMENT_LOG_ORACLE_SUBSET.md` status BLOCKED. **`outputs/oracle_subset_eval/` directory absent** (doc references sentinel path that is not in the committed tree). |
| Real routing dataset build | **COMPLETE (code)** / **PARTIAL (build dir vs data CSVs)** | Scripts exist. **`outputs/real_routing_dataset/`** has `gsm8k_per_query_outputs.csv`, `raw_responses.jsonl`, summaries — **no** `routing_dataset.csv` there. **Committed** paired dataset: `data/real_gsm8k_routing_dataset.csv` (+ enriched variant). |
| Learned routing training/evaluation | **PARTIAL** | `src/evaluation/real_routing_model_eval.py` + tests; **committed** `outputs/real_routing_model/` (and regime variants). `scripts/run_real_routing_model_eval.py` is a **different** pipeline (sklearn → `outputs/real_router_eval/`); that output dir **not** committed. **README/docs** instruct `python3 scripts/run_real_routing_model_eval.py` for the revise_helpful study — **inconsistent** with script implementation (see §4). |
| Baseline evaluations | **PARTIAL** | Code: `run_strong_baselines.py`, `run_recent_baselines_experiment.py`, `run_next_stage_baselines.py`. **Committed:** `outputs/baselines/*.json`. **`outputs/recent_baselines/` absent** (recent baselines script targets that dir per docstring). |
| Oracle routing evaluation | **PARTIAL (artifacts present)** | **Committed:** `outputs/oracle_routing_eval/*_oracle_summary.json` (e.g. `gsm8k_random100_oracle_summary.json`, `math500_100_oracle_summary.json`, `hard_gsm8k_*`). |
| Paper table generation | **MISSING** | No `outputs/paper_tables/`; no dedicated LaTeX/table export scripts located. |
| Paper figure generation | **MISSING** | No `outputs/paper_figures/`; no plot scripts in `scripts/` (matplotlib not used in scripts grep). |
| GPQA loader / normalization / fallback | **COMPLETE (code)** / **PARTIAL (tests)** | `src/datasets/gpqa.py` documents official vs mirror fallback. **Committed** `data/gpqa_diamond_normalized.jsonl`. Tests in `tests/test_gpqa_loader.py` skip Hub tests when unreachable; `test_gpqa_jsonl_roundtrip` conditional on file. |
| Tests for loaders / routing logic | **COMPLETE (in this env)** | `609 passed` includes `test_real_routing_model_eval.py`, `test_gpqa_loader.py`, `test_recent_baselines_oracle.py`, policy tests, etc. |
| Manuscript / project docs | **PARTIAL** | `docs/PROJECT_CONTEXT.md`, `docs/BASELINE_TRACKER.md`, many result docs. **Internal tension:** `docs/FULL_REPO_AUDIT.md` and `docs/OFFLINE_CONCEPTUAL_BOTTLENECK_REVIEW.md` note **oracle docs vs BLOCKED logs** — `docs/EXPERIMENT_LOG_ORACLE_SUBSET.md` is BLOCKED; `docs/RESULTS_ORACLE_SUBSET.md` exists; numeric claims elsewhere may lack matching `outputs/oracle_subset_eval/`. |

---

## 3. Artifact existence check

Paths relative to repo root. “Present” = exists in this clone at audit time.

| Expected / asked path | Status | Notes |
|----------------------|--------|--------|
| `outputs/real_policy_eval/` | **Present** | e.g. `summary.json`, `policy_comparison.csv`, `per_query_policy_decisions.csv` |
| `outputs/real_routing_model/` | **Present** | `routing_simulation.csv`, `summary.json`, `model_metrics.csv`, … |
| `outputs/multi_action_models/` | **Absent** | No directory; multi-action logic lives in **evaluation code** (`multi_action_oracle` in `src/evaluation/recent_baselines_eval.py`), not this path. |
| `outputs/baselines/` | **Present** | `gsm8k_baseline_summary.json`, `hard_gsm8k_baseline_summary.json`, `math500_baseline_summary.json` |
| `outputs/oracle_routing_eval/` | **Present** | Multiple `*_oracle_summary.json` files |
| `outputs/paper_tables/` | **Absent** | |
| `outputs/paper_figures/` | **Absent** | |
| Real oracle/adaptive per-query CSVs | **Present (multiple)** | Examples: `outputs/real_routing_dataset/gsm8k_per_query_outputs.csv`, `outputs/real_math500_routing/per_query_outputs.csv`, `outputs/real_hard_gsm8k_routing/per_query_outputs.csv`, `outputs/real_aime2024_routing/per_query_outputs.csv`, `outputs/adaptive_policy_v7/per_case_results.csv` |
| Routing dataset CSVs under `data/` | **Present** | `data/real_gsm8k_routing_dataset.csv`, `data/real_gsm8k_routing_dataset_enriched.csv`, `data/real_math500_routing_dataset.csv`, `data/real_math500_routing_dataset_enriched.csv`, `data/real_hard_gsm8k_routing_dataset.csv`, `data/real_hard_gsm8k_routing_dataset_enriched.csv`, `data/real_hard_gsm8k_b2_routing_dataset.csv`, `data/real_hard_gsm8k_b2_routing_dataset_enriched.csv`, `data/real_aime2024_routing_dataset.csv` |
| Blocker / status docs | **Present** | e.g. `docs/EXPERIMENT_LOG_ORACLE_SUBSET.md` (BLOCKED), `docs/GPQA_ACCESS_CHECK.md`, `docs/UPDATED_RECENT_BASELINES_AND_DATASETS_REPORT.md` (blockers section) |
| `outputs/oracle_subset_eval/` | **Absent** | Referenced in experiment log as sentinel location; not in repo |
| `outputs/recent_baselines/` | **Absent** | Expected by `scripts/run_recent_baselines_experiment.py` docstring when run |
| `outputs/simulated_sweep/` / `outputs/simulated_multi_seed/` | **Absent** | Not committed (see `.gitignore` pattern `outputs/*` with exceptions) |

---

## 4. Script readiness check

### 4.1 Requested scripts (CLI, inputs, outputs, runnable class)

| Script | CLI | Expected inputs | Outputs | Runnable as-is? |
|--------|-----|-----------------|---------|-----------------|
| `scripts/run_simulated_allocation.py` | `--config` (required) | YAML/JSON: `budget`, `synthetic`, `allocator`, optional `output` | Default `outputs/simulated_allocation_results.json` | **Yes** (no API; uses `load_config`) |
| `scripts/run_simulated_sweep.py` | `--config` | `configs/simulated_sweep.yaml` style | Writes under config `output_dir` (CSV/JSON) | **Yes** (no API) |
| `scripts/run_simulated_multi_seed.py` | `--config` | Seeds / `n_seeds` in config | `outputs/simulated_multi_seed` default | **Yes** (no API) |
| `scripts/run_oracle_subset_eval.py` | `--config` | `configs/oracle_subset_eval_gsm8k.yaml`; GSM8K via loader | `outputs/oracle_subset_eval/` + optional `docs/EXPERIMENT_LOG_ORACLE_SUBSET.md` | **Blocked** without `OPENAI_API_KEY` (script documents this) |
| `scripts/run_build_real_routing_dataset.py` | `--subset-size`, `--paired-outcomes`, `--output-dataset-csv`, `--gsm8k-data-file`, `--output-dir`, … | HF or bundled GSM8K; API | Default `outputs/real_routing_dataset/`; optional `data/real_gsm8k_routing_dataset.csv` | **Blocked** without `OPENAI_API_KEY`; needs network or local data file |
| Training routing models | **`scripts/run_real_routing_model_eval.py`** | `outputs/real_routing_dataset/routing_dataset.csv` **or** path via `--routing-csv` | Default **`outputs/real_router_eval/`** (`router_eval_results.json`, etc.) | **Blocked** if CSV missing or no oracle labels; **sklearn** recommended |
| **Revise-helpful routing eval (committed artifacts)** | **No `scripts/` wrapper found** | `data/real_gsm8k_routing_dataset.csv` | **`outputs/real_routing_model/`** (from `run_real_routing_model_eval` in `src/evaluation/real_routing_model_eval.py`) | **Invokable via Python/tests**; README line 324 documents CLI that runs the **other** script |
| Baselines | `scripts/run_strong_baselines.py --config ...` | Config selects dummy vs OpenAI, datasets | Under config (e.g. `outputs/strong_baselines/`) | **Dummy: yes**; **OpenAI/HF: API + network + gated HF for GPQA** |
| Paper-ready summarization | `scripts/summarize_simulated_results.py` | `--input-dir` default `outputs/simulated_sweep` | **Stdout** summary | **Runnable** if input CSVs exist; **blocked** if dir empty |
| `scripts/run_cross_regime_comparison.py` | (none; fixed paths) | JSON summaries under `outputs/` | `outputs/cross_regime_comparison/cross_regime_summary.csv` | **Yes** if referenced files exist (they do for main regimes) |
| `scripts/run_final_cross_regime_summary.py` | (none) | `data/*routing_dataset*.csv`, policy `summary.json` files | `outputs/cross_regime_comparison/final_cross_regime_summary.csv` | **Yes** with committed CSVs |

### 4.2 Stale / inconsistent

- **`scripts/run_real_routing_model_eval.py` docstring** claims outputs `outputs/real_router_eval/` but also duplicates README wording about “follow-up to run_build…”; **README.md** (~324) says the same command produces revise_helpful / `outputs/real_routing_model/` style results — that matches **`src/evaluation/real_routing_model_eval.py`**, not the sklearn script body.
- **`run_build_real_routing_dataset.py` epilog** suggests `python3 scripts/run_real_routing_model_eval.py` after build; sklearn script expects **`routing_dataset.csv`** in default path — verify file name alignment when using default dirs.

---

## 5. Branch / merge awareness (local only)

| Item | Finding |
|------|---------|
| Current branch | `cursor/main-branch-status-audit-e5ff` |
| Local `main` vs `origin/main` | **Local `main` is 140 commits behind** `origin/main` (stuck at initial commit). |
| Other local branches | `cursor/development-environment-setup-d0f8` — **43 commits ahead** of local `main` (still behind `origin/main` if that branch is old). |
| Work “not on main” | All substantive code is on **`origin/main` @ `f0b9d8c`**, not on local `main` ref. |
| Conflict markers | **None** found. |
| Coherence | **`origin/main` is the coherent mainline** for development. Local `main` ref should be updated to avoid mistaken checkouts. |

---

## 6. Done vs remaining

### A. Definitely done on `origin/main` (evidence-based)

- Full Python package layout, configs, and **609** passing tests (this environment).
- Simulated allocation/sweep **codepaths** and configs under `configs/simulated_*.yaml`.
- Real routing **data** CSVs and **enriched** variants under `data/` (explicitly un-ignored in `.gitignore`).
- Committed **policy** outputs: `outputs/real_policy_eval/`, `outputs/real_math500_policy_eval/`, `outputs/real_hard_gsm8k_policy_eval/`, `outputs/real_hard_gsm8k_b2_policy_eval/`.
- Committed **learned routing** outputs: `outputs/real_routing_model/`, `outputs/real_math500_routing_model/`, `outputs/real_hard_gsm8k_routing_model/`, `outputs/real_hard_gsm8k_b2_routing_model/`.
- **Oracle routing** summaries under `outputs/oracle_routing_eval/`.
- **Next-stage** curves and merged JSON under `outputs/next_stage_eval/`, **budget** CSVs under `outputs/budget_sweep/`.
- **Cross-regime** summaries under `outputs/cross_regime_comparison/`.
- **Baselines** JSON under `outputs/baselines/`.
- **Adaptive policy v7** committed outputs under `outputs/adaptive_policy_v7/`.
- GPQA normalization module + committed `data/gpqa_diamond_normalized.jsonl`.

### B. Done on another branch / hinted in docs but not verified on `origin/main`

- Anything claimed only in **`docs/ORACLE_ANALYSIS_SUMMARY.md`**-style narratives without matching **`outputs/oracle_subset_eval/`** files — **not verified** from artifacts (see `docs/FULL_REPO_AUDIT.md` discussion).
- **`outputs/recent_baselines/`** — not present; docstring of `run_recent_baselines_experiment.py` describes intended outputs; **not evidence of a completed committed run**.

### C. Remaining engineering work

| Item | Classification |
|------|----------------|
| Align local `git` branch `main` with `origin/main` | **documentation / git hygiene** (not code) |
| Resolve **README / docs / `run_real_routing_model_eval.py` script** mismatch (sklearn vs `src/evaluation/real_routing_model_eval.py`) | **code implementation** + **documentation cleanup** |
| Add CLI wrapper or rename script for revise-helpful pipeline | **code implementation** |
| Optional: LaTeX/table + figure export pipeline | **code implementation** |
| Paper tables directory convention | **code implementation** + **result aggregation** |

### D. Remaining experiments / artifact generation

| Item | Classification |
|------|----------------|
| Run `run_oracle_subset_eval.py` with API to populate `outputs/oracle_subset_eval/` | **experiment run** + **API/internet** |
| Run simulated sweeps and commit or document output paths | **experiment run** |
| Run `run_recent_baselines_experiment.py` end-to-end | **experiment run** + **API/internet** + possibly **HF gated** |
| Strong baselines on real models / GPQA official Hub | **experiment run** + **API** + **HF auth** |
| Regenerate any stale CSV if configs changed | **experiment run** / **result aggregation** |

---

## 7. Prioritized next 10 steps (from `origin/main`)

1. **Repair local `main` pointer** — **Why:** prevents wrong baseline for future work. **How:** `git fetch origin main && git branch -f main origin/main`. **Deps:** none. **Branch:** any; fix is local ref only.

2. **Reconcile routing eval entrypoints** — **Why:** README currently misdirects. **How:** Inspect `scripts/run_real_routing_model_eval.py`, `src/evaluation/real_routing_model_eval.py`, `README.md`, `docs/REAL_ROUTING_MODEL_RESULTS.md`; add thin CLI or rename. **Deps:** none for code edit. **Branch:** new feature branch recommended.

3. **Run `pytest` + `ruff`** on PRs — **Why:** already green here; keep CI parity. **Files:** `pyproject.toml`, `tests/`. **Deps:** dev install. **Branch:** any.

4. **Re-run or archive `oracle_subset_eval`** — **Why:** close gap between BLOCKED docs and paper claims. **Script:** `scripts/run_oracle_subset_eval.py --config configs/oracle_subset_eval_gsm8k.yaml`. **Deps:** **OPENAI_API_KEY**, internet. **Branch:** experiment branch optional.

5. **Decide commit policy for `outputs/recent_baselines/`** — **Why:** script expects it; dir missing. **Script:** `run_recent_baselines_experiment.py`. **Deps:** API + HF. **Branch:** new branch if adding large JSON.

6. **Simulated sweep artifact bundle** — **Why:** `summarize_simulated_results.py` needs `outputs/simulated_sweep` data. **Scripts:** `run_simulated_sweep.py`, `summarize_simulated_results.py`. **Deps:** none. **Branch:** new branch if committing outputs.

7. **Paper table export** — **Why:** no `outputs/paper_tables/`. **How:** script consuming `outputs/cross_regime_comparison/*.csv` and `outputs/next_stage_eval/**`. **Deps:** none. **Branch:** new branch.

8. **Paper figures** — **Why:** no figure pipeline. **How:** matplotlib or export from CSVs. **Deps:** optional sklearn/plotting deps. **Branch:** new branch.

9. **Document oracle doc conflicts** — **Why:** explicit “same page” for readers. **Files:** `docs/ORACLE_ANALYSIS_SUMMARY.md` vs `docs/EXPERIMENT_LOG_ORACLE_SUBSET.md`. **Deps:** none. **Branch:** docs-only branch on `origin/main`.

10. **GPQA end-to-end strong baselines** — **Why:** loader ready; gated Hub may block. **Script:** `run_strong_baselines.py` with real config. **Deps:** **API**, **HF_TOKEN** / gating. **Branch:** experiment branch.

---

## 8. Summary table: `origin/main` readiness

| Goal | Ready? |
|------|--------|
| Reproduce offline tests | **Yes** (609 passed here) |
| Reproduce real LLM runs from clone alone | **No** (needs keys + network) |
| Paper-ready tables/figures from repo | **No** (no `paper_tables` / `paper_figures`; aggregation scripts are partial) |
| Single clean story for oracle subset | **No** (BLOCKED log; missing `outputs/oracle_subset_eval/`) |

---

*End of audit. All file paths verified from workspace listing and/or direct reads at generation time.*
