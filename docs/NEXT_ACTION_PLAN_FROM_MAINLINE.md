# Next action plan from mainline

**Scope:** Follow-up engineering after the routing-script coherence pass (branch based on `origin/main`). **Git:** Treat `origin/main` as authoritative; realign local `main` if it is stale (see `README.md` / `AGENTS.md`).

---

## A. Coherence fixes completed in this pass

1. **Split two routing evaluation entry points**
   - **`scripts/run_real_routing_model_eval.py`** — thin CLI for `src.evaluation.real_routing_model_eval.run_real_routing_model_eval` (paired-outcomes CSV, `revise_helpful`, default **`outputs/real_routing_model/`**).
   - **`scripts/run_router_baseline_eval.py`** — sklearn oracle-label routers (`router_baseline` module), default **`outputs/real_router_eval/`**, input `routing_dataset.csv` with oracle columns.

2. **Documentation aligned** with the split: `README.md`, `AGENTS.md`, `docs/REAL_ROUTING_MODEL_RESULTS.md`, `docs/FINAL_PRE_RUN_AUDIT.md`, `docs/CODEX_NEXT_EXPERIMENT_RECOMMENDATION.md`, `scripts/run_build_real_routing_dataset.py` “next step” hint.

3. **Mainline vs local `main` confusion** — short notes in `README.md` and `AGENTS.md` that `origin/main` is the real baseline and local `main` may need `git branch -f main origin/main`.

---

## B. Remaining mismatches not fixed (intentionally or pre-existing)

| Item | Notes |
|------|--------|
| `docs/ROUTER_BASELINE.md`, `docs/STATE_OF_EVIDENCE.md`, `docs/RESULTS_INVENTORY.md` | Still reference **`scripts/run_router_baseline.py`** (separate legacy script). Not renamed; may confuse readers next to **`run_router_baseline_eval.py`**. Consider one doc paragraph cross-linking the three: `run_router_baseline.py` vs `run_router_baseline_eval.py` vs `run_real_routing_model_eval.py`. |
| `docs/MAIN_BRANCH_STATUS_AUDIT.md` | **Not on `origin/main`** in this workspace until merged from the audit PR; if present elsewhere, update its §4 table to reflect the new script split. |
| `run_real_routing_model_eval.py` exit code | Returns **1** when `run_status != OK` (e.g. blocked / single-class); callers should not assume 0 for all non-crash exits. |

---

## C. Missing scripts / artifacts (paper, baselines, oracle)

### Paper tables

- **No** `outputs/paper_tables/` or LaTeX export pipeline.
- **Reusable logic:** `src/evaluation/analysis_summary.py` — `write_csv_table`, `summarize_simulated_results`, `format_terminal_summary`; `scripts/run_cross_regime_comparison.py`, `scripts/run_final_cross_regime_summary.py` produce CSVs under `outputs/cross_regime_comparison/`.
- **Suggested next script:** `scripts/export_paper_tables.py`
  - **Inputs:** `outputs/cross_regime_comparison/*.csv`, `outputs/next_stage_eval/**/*.json`, optional `outputs/budget_sweep/*.csv`.
  - **Outputs:** `outputs/paper_tables/*.csv` and optional `.tex` fragments (no fabricated numbers).

### Paper figures

- **No** `outputs/paper_figures/` directory convention in repo.
- **Reusable logic:** `src/evaluation/analysis_summary.py` — `generate_summary_plots()` (matplotlib; expects in-memory summary rows). `README.md` mentions `outputs/simulated_sweep/plots/` when matplotlib is available (downstream of sweep runs).
- **Suggested next script:** `scripts/generate_paper_figures.py`
  - **Inputs:** same as table script or explicit paths to existing CSVs.
  - **Outputs:** `outputs/paper_figures/*.png` (or PDF); skip gracefully if matplotlib missing.

### Recent baselines

- **Artifact gap:** `scripts/run_recent_baselines_experiment.py` targets **`outputs/recent_baselines/`** when run; directory not committed.
- **Next step:** run with API + document whether outputs should be gitignored vs selectively un-ignored (policy decision).

### Oracle subset / real routing experiments

- **Artifact gap:** `outputs/oracle_subset_eval/` absent; `docs/EXPERIMENT_LOG_ORACLE_SUBSET.md` records BLOCKED runs.
- **Scripts:** `scripts/run_oracle_subset_eval.py`, `scripts/run_build_real_routing_dataset.py` (unchanged contract aside from next-step hint).

### Multi-action

- **No** `outputs/multi_action_models/`; multi-action summaries live in evaluation JSON (`multi_action_oracle` in recent baselines code). No separate artifact directory required unless you add saved checkpoints.

---

## D. Prioritized next steps (files to touch)

1. **Optional doc consolidation** — `docs/ROUTER_BASELINE.md`, `docs/RESULTS_INVENTORY.md`, `docs/STATE_OF_EVIDENCE.md`: add a small “which script?” table (`run_router_baseline.py` / `run_router_baseline_eval.py` / `run_real_routing_model_eval.py`).

2. **`scripts/export_paper_tables.py`** — wrap CSV reads + column selection; write under `outputs/paper_tables/` (gitignore unless you add exceptions).

3. **`scripts/generate_paper_figures.py`** — call or refactor `generate_summary_plots` / new plotting helpers; standardize `outputs/paper_figures/`.

4. **Merge audit doc update** — if `docs/MAIN_BRANCH_STATUS_AUDIT.md` is merged, patch §2–§4 for the two-script layout.

5. **Run and optionally commit** `run_recent_baselines_experiment.py` outputs — or document “never commit” in `.gitignore` comments.

6. **Oracle subset re-run** — with `OPENAI_API_KEY`; populate `outputs/oracle_subset_eval/` to match `docs/EXPERIMENT_LOG_*` expectations.

---

*Last updated as part of the mainline coherence pass.*
