# Final Consistency Audit

Date: 2026-03-31  
Scope: Manuscript-facing docs/tables/figures and committed output artifacts only.

## Inconsistencies Found and Resolution

| Inconsistency | Affected files | Proposed resolution | Canonical final choice | Status |
|---|---|---|---|---|
| Docs reference `outputs/paper_figures_cleaned/*`, but directory is absent | `README.md`, `MANUSCRIPT_REPRODUCTION.md`, `PAPER_ARTIFACT_STATUS.md`, `docs/FINAL_MANUSCRIPT_TABLE_FIGURE_INDEX.md`, `docs/FINAL_MANUSCRIPT_ASSET_MAP.md` | Introduce one authoritative final bundle generated from committed sources | Use `outputs/paper_tables_final/*` and `outputs/paper_figures_final/*` | Fixed automatically |
| Mixed regime naming (`gsm8k_random100` vs `gsm8k_random_100`) | `outputs/paper_tables/cross_regime/final_cross_regime_summary.csv` and multiple docs | Normalize in final export layer and canonical docs | Canonical ids: `gsm8k_random_100`, `hard_gsm8k_100`, `hard_gsm8k_b2`, `math500_100` | Fixed automatically |
| Main-story policy ambiguity (v5 vs v6/v7) | `docs/PAPER_STRENGTHENING_FROM_REPO.md`, `docs/FINAL_MANUSCRIPT_ASSET_MAP.md`, various summary docs | Decide explicitly and apply consistently in final outputs | `adaptive_policy_v5` is canonical primary adaptive comparator; v6/v7 retained in main comparison tables for transparency | Fixed automatically (author rationale documented) |
| Cross-regime table in old exports includes AIME row with incomplete best-policy fields | `outputs/paper_tables/cross_regime/final_cross_regime_summary.csv` and docs citing it | Exclude AIME from main-paper canonical tables; keep as exploratory/supplementary | Main regimes are only the four n=100 regimes | Fixed automatically |
| Stale claims that GPQA policy eval is missing while GPQA build artifacts now exist | `PAPER_ARTIFACT_STATUS.md`, older GPQA notes | Distinguish GPQA build completion from manuscript main-regime scope | GPQA remains supplementary in this pass; main paper still four-regime set | Needs author judgment for publication placement |
| Contradiction between "main-paper baselines" and n-mismatched baseline files | `MANUSCRIPT_ARTIFACTS.md`, `PAPER_ARTIFACT_STATUS.md`, baseline CSV docs | Keep baseline strategy rollups appendix-only with explicit sample-size caveat | `outputs/paper_tables_final/baseline_comparison_appendix.csv` marked appendix-only | Fixed automatically |
| Overlapping "final/enhanced/cleaned" table families create ambiguity | `outputs/paper_tables_cleaned/*`, `outputs/paper_tables/*`, `outputs/paper_tables_enhanced/*`, multiple indices | Declare one authoritative final family and keep others as historical | Authoritative: `outputs/paper_tables_final/*` | Fixed automatically |
| Overlapping figure families and non-authoritative graphic abstract location | `outputs/paper_figures/*`, `outputs/paper_figures_enhanced/*`, docs | Regenerate one canonical figure set and graphic abstract in one place | Authoritative: `outputs/paper_figures_final/*` | Fixed automatically |

## Notes Requiring Human Judgment

- Whether v6 should be foregrounded as a lower-cost operating point in prose even though v5 is the primary accuracy comparator.
- Whether GPQA-Diamond should remain supplementary or be moved into appendix tables in the next manuscript revision.
