# Repository Polish Changelog

**Date:** 2026-04-01  
**Branch:** `copilot/full-repo-polish-recheck`  
**Goal:** Turn the repo from an evolving research workspace into a cleaner,
better organized, manuscript-supporting package — without changing scientific
claims or inventing new results.

---

## Files Moved

| From | To | Reason |
|------|----|--------|
| `MANUSCRIPT_ARTIFACTS.md` | `docs/archive/MANUSCRIPT_ARTIFACTS.md` | Marked "OUTDATED / SUPERSEDED (2026-03-31)" in its own banner; superseded by `outputs/paper_tables_final/` and `FINAL_MANUSCRIPT_ASSET_INDEX.md` |
| `MANUSCRIPT_SUPPORT_ADDITIONS.md` | `docs/archive/MANUSCRIPT_SUPPORT_ADDITIONS.md` | Marked "OUTDATED / SUPERSEDED (2026-03-31)" in its own banner; superseded by `FINAL_MANUSCRIPT_ASSET_INDEX.md` |
| `PAPER_ARTIFACT_STATUS.md` | `docs/archive/PAPER_ARTIFACT_STATUS.md` | Marked "OUTDATED / SUPERSEDED (2026-03-31)" in its own banner; superseded by `docs/CANONICAL_MANUSCRIPT_DECISIONS.md` |
| `PUBLIC_RELEASE_CHECKLIST.md` | `docs/PUBLIC_RELEASE_CHECKLIST.md` | Process/housekeeping document; not manuscript-facing; more appropriate under `docs/` |

---

## Files Created

| File | Reason |
|------|--------|
| `docs/archive/README.md` | Explains the purpose of `docs/archive/` and what to use instead |
| `docs/README.md` | Four-tier navigation index for `docs/` folder (Canonical, Supporting, Internal, Archive + Working Notes) |
| `docs/REPO_ORGANIZATION_AUDIT.md` | Full repo audit: strengths, weaknesses, duplicates, classification, decisions |
| `docs/CANONICAL_REPO_GUIDE.md` | New-reader guide: where to start, what is canonical, what is supplementary, what is legacy |
| `docs/REPO_POLISH_CHANGELOG.md` | This file |

---

## Files Edited

### `README.md`
- **Committed Results table:** Replaced `paper_tables_cleaned/` and
  `paper_figures_cleaned/` (incorrectly marked `✅ Main paper`) with
  `paper_tables_final/` and `paper_figures_final/` as canonical. Demoted the
  `_cleaned/` directories to `📦 Historical`. Added `oracle_routing_eval/`,
  `cross_regime_comparison/`, and `baselines/` entries for completeness.
- **Key documentation table:** Replaced stale references to moved files; added
  entries for `docs/CANONICAL_REPO_GUIDE.md` and `docs/PUBLIC_RELEASE_CHECKLIST.md`.
- **Artifact inventory pointer:** Changed from `MANUSCRIPT_ARTIFACTS.md` and
  `PAPER_ARTIFACT_STATUS.md` (now archived) to `FINAL_MANUSCRIPT_ASSET_INDEX.md`.

### `MANUSCRIPT_REPRODUCTION.md`
- **Section 3 (Manuscript Tables and Figures):** Rewrote the entire section.
  - Added a prominent callout box pointing to `paper_tables_final/` and
    `paper_figures_final/` as the authoritative canonical targets.
  - Replaced the old main-paper table list (pointing to `paper_tables_cleaned/`)
    with a new table reflecting `paper_tables_final/` files.
  - Replaced the old main-paper figure list (pointing to `paper_figures_cleaned/`)
    with a new table reflecting `paper_figures_final/` files.
  - Demoted the old `paper_tables_cleaned/` references to a clearly-labelled
    "Historical intermediate tables and figures" sub-table.
  - Removed appendix table rows pointing to now-superseded paths.

### `REPRODUCIBILITY.md`
- **Step A6:** Fixed path `outputs/real_budget_sweep/` → `outputs/budget_sweep/`
  (the committed directory name is `budget_sweep/` not `real_budget_sweep/`).
- **Step B7 / Committed Artifacts table:** Replaced non-existent
  `scripts/run_paper_table_export.py` with the correct scripts:
  `generate_paper_tables.py`, `generate_paper_figures.py`, and
  `generate_final_manuscript_artifacts.py`. Added `outputs/paper_tables_final/`
  and `outputs/paper_figures_final/` to the committed artifacts table. Corrected
  BLOCKED step reference from "A8" to "A9" (simulated sweep step).

---

## Files Reclassified

| File | Old classification | New classification |
|------|--------------------|--------------------|
| `MANUSCRIPT_ARTIFACTS.md` | Root-level, "public" (per `PAPER_ARTIFACT_STATUS.md` H section) | `docs/archive/` — historical/superseded |
| `MANUSCRIPT_SUPPORT_ADDITIONS.md` | Root-level | `docs/archive/` — historical/superseded |
| `PAPER_ARTIFACT_STATUS.md` | Root-level, "public" | `docs/archive/` — historical/superseded |
| `PUBLIC_RELEASE_CHECKLIST.md` | Root-level | `docs/` — process/housekeeping |

---

## What Was Not Changed and Why

| Item | Decision | Rationale |
|------|----------|-----------|
| ~70 working-note docs in `docs/` | Not moved to `docs/archive/` | Heavy internal cross-references; moving would break ~200+ inter-doc links with no benefit to external readers. `docs/README.md` tiered index achieves the same discoverability improvement safely. |
| `config/` vs `configs/` split | Not merged | `routing/token_budget_router/` uses `config/`; `configs/` is used by all other scripts. Merging risks breaking scripts. |
| `outputs/expanded_main_regimes/` | Not added to artifact index | Presence confirmed; content verified as per-query CSV results for all 4 regimes. Safe to retain. Future pass should add it to `FINAL_MANUSCRIPT_ASSET_INDEX.md` if cited. |
| Source code, configs, tests | Not touched | No code changes needed; all paths validated. |
| `outputs/paper_tables_cleaned/` and `outputs/paper_figures_cleaned/` | Not deleted | Retained as historical intermediates; deletion would lose traceability. |
| `scripts/run_paper_table_export.py` | Not created | The file doesn't exist; the correct scripts are already present. REPRODUCIBILITY.md now points to them. |

---

## Validation Summary (Post-Cleanup)

### Path checks

| Check | Result |
|-------|--------|
| `outputs/paper_tables_final/` exists | ✅ Yes (8 CSV files + README) |
| `outputs/paper_figures_final/` exists | ✅ Yes (9 PNG/PDF + caption text) |
| `outputs/budget_sweep/` exists | ✅ Yes (4 CSVs) |
| `data/real_*_routing_dataset_enriched.csv` (4 files) | ✅ All present |
| `scripts/generate_final_manuscript_artifacts.py` | ✅ Exists |
| `scripts/generate_paper_tables.py` | ✅ Exists |
| `scripts/generate_paper_figures.py` | ✅ Exists |
| `scripts/run_real_policy_eval.py` | ✅ Exists |
| `docs/CANONICAL_MANUSCRIPT_DECISIONS.md` | ✅ Exists |
| `docs/FINAL_CONSISTENCY_AUDIT.md` | ✅ Exists |
| `docs/STATE_OF_EVIDENCE.md` | ✅ Exists |
| `docs/BASELINE_TRACKER.md` | ✅ Exists |
| `docs/PUBLIC_RELEASE_CHECKLIST.md` | ✅ Exists (moved from root) |
| `docs/archive/README.md` | ✅ Exists (new) |
| `docs/README.md` | ✅ Exists (new) |

### Path fixes applied

| Old path (broken) | New path (correct) |
|-------------------|--------------------|
| `outputs/real_budget_sweep/` (in REPRODUCIBILITY.md A6) | `outputs/budget_sweep/` |
| `scripts/run_paper_table_export.py` (in REPRODUCIBILITY.md B7) | `scripts/generate_final_manuscript_artifacts.py` etc. |

### Inconsistency fixes applied

| Inconsistency | Fix |
|---------------|-----|
| `MANUSCRIPT_REPRODUCTION.md` Section 3 listed `paper_tables_cleaned/` as canonical despite its own header banner saying `paper_tables_final/` was canonical | Section 3 rewritten to use `paper_tables_final/` throughout |
| `README.md` Committed Results table listed `paper_tables_cleaned/` as `✅ Main paper` | Updated to `paper_tables_final/` as canonical; `paper_tables_cleaned/` marked `📦 Historical` |

---

## Remaining Issues / Open Items

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| ~70 working-note docs in `docs/` are not clearly separated from canonical docs | Low | `docs/README.md` index now provides separation without file moves. Future pass: batch-move to `docs/archive/`. |
| `outputs/expanded_main_regimes/` not in any artifact index | Low | Verify whether its data is used by any canonical script; if yes, add to `FINAL_MANUSCRIPT_ASSET_INDEX.md`. |
| AIME-2024 and GPQA-Diamond data licensing | Medium (legal, not structural) | Human must verify redistribution rights before public release; see `docs/PUBLIC_RELEASE_CHECKLIST.md`. |
| `routing/` package at root (not under `src/`) | Low (cosmetic) | Inconsistent project layout. Merging under `src/routing/` would be clean but risks breaking imports; out of scope for this pass. |
