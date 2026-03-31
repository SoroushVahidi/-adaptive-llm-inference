# Current Repository Consistency Audit

**Date:** 2026-03-30  
**Scope:** Align documentation with the test suite, scripts, and committed `outputs/` after the small-experiment pass (AIME + confidence-threshold baseline).

---

## 1. What Was Inaccurate

### 1.1 Unit test counts and timing

- **README**, **REPRODUCIBILITY.md**, **MANUSCRIPT_REPRODUCTION.md**, and **PUBLIC_RELEASE_CHECKLIST.md** claimed **612 passed / 5 skipped** and **~10 s** for `pytest`.
- The repository now has **677 collected tests** (`pytest --collect-only`). A full run takes **tens of seconds** (not ~10 s). The **pass vs skip** split depends on environment: e.g. GPQA loader tests skip when Hugging Face Hub access fails; with Hub reachable, **all 677 can pass** with **0 skipped**.
- Root cause of older mismatch: test files and parametrized cases grew; the documented numbers were stale.
- **Dependency note:** `datasets` 2.x with **pyarrow ≥ 15** fails at import time (`PyExtensionType` removed). The project dependency was raised to **`datasets>=3.0`** so a normal `pip install -e .` matches current pyarrow stacks.

### 1.2 Small-pass output paths

- **docs/SMALL_EXPERIMENT_PASS_AIME_GPQA.md**, **docs/BLOCKERS_AIME_GPQA_SMALL_PASS.md**, **docs/CONFIDENCE_ROUTER_BASELINE.md**, and **MANUSCRIPT_SUPPORT_ADDITIONS.md** referenced **`outputs/small_pass/...`** and **`outputs/paper_tables_small_pass/...`**.
- Those directories were **not present** in the checkout until **`python scripts/run_small_pass.py`** was run; the docs described intended artifacts, not committed files.

### 1.3 Confidence-threshold baseline

- **`outputs/baselines/confidence_threshold/`** was already committed (sweep, summary CSV, summary JSON).
- **`outputs/paper_tables_small_pass/confidence_baseline_main_regimes.csv`** was missing until the small-pass run regenerated manuscript-oriented tables.
- **MANUSCRIPT_SUPPORT_ADDITIONS.md** named a **non-existent script** `scripts/run_confidence_threshold_baseline.py`; the real entry point is **`scripts/run_confidence_baseline.py`**.

### 1.4 Manuscript reproduction vs AIME

- **MANUSCRIPT_REPRODUCTION.md** stated that AIME policy evaluation was **not** run, while the small-pass docs described **offline** AIME evaluation. After committing small-pass outputs, the manuscript guide needed to distinguish **main-paper tables** (no AIME) from **supplementary committed eval** under `outputs/small_pass/`.

### 1.5 README data table

- The AIME row in **README** said exploratory data had **no policy eval**; supplementary policy eval is now committed under `outputs/small_pass/`.

---

## 2. What Was Fixed

| Area | Change |
|------|--------|
| Dependencies | **`pyproject.toml`:** `datasets>=3.0` (with comment on pyarrow compatibility). |
| Test expectations | **README**, **REPRODUCIBILITY.md**, **MANUSCRIPT_REPRODUCTION.md**, **PUBLIC_RELEASE_CHECKLIST.md**, **MANUSCRIPT_SUPPORT_ADDITIONS.md:** **677 tests collected**, **0 failures**; timing **tens of seconds**; optional **skips** documented where relevant. |
| Small-pass outputs | Ran **`python scripts/run_small_pass.py`** and **committed** `outputs/small_pass/` and `outputs/paper_tables_small_pass/` (AIME summaries, policy comparison, confidence sweeps, combined tables, run summaries). **`.gitignore`** previously ignored these paths under the blanket `outputs/*` rule; added **exceptions** so they can be tracked. |
| Confidence baseline | Re-ran **`python scripts/run_confidence_baseline.py`** so **`outputs/baselines/confidence_threshold/`** matches the same sweep logic; verified **sweep CSV matches** `outputs/small_pass/confidence_threshold/confidence_threshold_sweep.csv`. |
| Script name | **MANUSCRIPT_SUPPORT_ADDITIONS.md:** `run_confidence_baseline.py` everywhere. |
| Manuscript guide | **MANUSCRIPT_REPRODUCTION.md:** Section 2 blockquote and section 4 tree for **`small_pass/`** and **`paper_tables_small_pass/`**; Step 2 command for **`run_small_pass.py`**; section 7 table row for AIME **exploratory** outputs. |
| README | Committed results table + AIME data row; test count line. |
| Small-pass doc | **docs/SMALL_EXPERIMENT_PASS_AIME_GPQA.md:** note on **two** valid output locations for confidence CSVs (`baselines/` vs `small_pass/confidence_threshold/`). |

---

## 3. What Is Still Blocked or Environment-Sensitive

| Item | Status |
|------|--------|
| **GPQA / GPQA-Diamond policy eval** | Still **blocked** (no committed routing features / model responses). Unchanged; see **docs/BLOCKERS_AIME_GPQA_SMALL_PASS.md**. |
| **Exact pytest duration** | Varies by CPU and I/O; docs use **qualitative** wording instead of a single second count. |
| **Skipped tests** | **Environment-dependent** (`skipif` on GPQA Hub tests, sklearn, etc.). Use `pytest -r skip` to list skips; **failure count should remain 0**. |

---

## 4. Verification Commands (this checkout)

```bash
pip install -e ".[dev]"
pytest -q    # expect 677 collected, 0 failures (optional skips)
python scripts/run_small_pass.py
python scripts/run_confidence_baseline.py
```

No API key is required for the commands above.
