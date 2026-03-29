# Uploaded Dataset Validation Report

Date run: **2026-03-29 (UTC)**

## 1) ZIP discovery

The repository scan found exactly two ZIP archives at the repo root:

- `archive.zip`
- `archive (1).zip`

Discovery command:

```bash
rg --files -g '*.zip'
```

## 2) Archive contents and inspection

### A. `archive.zip`

`unzip -l archive.zip` shows a `gsm8k/` tree with HuggingFace-style parquet shards:

- `gsm8k/main/test-00000-of-00001.parquet`
- `gsm8k/main/train-00000-of-00001.parquet`
- `gsm8k/socratic/test-00000-of-00001.parquet`
- `gsm8k/socratic/train-00000-of-00001.parquet`
- `gsm8k/README.md`

Sample parsed fields from parquet are:

- `question`
- `answer`

Validation outcome:

- `valid_gsm8k = yes`
- `valid_math500 = no`
- `uncertain = no`
- reason: GSM8K-like question/answer structure and language patterns.

Normalization outcome:

- normalized to `data/gsm8k_uploaded_normalized.jsonl`
- rows written: `17584`

---

### B. `archive (1).zip`

`unzip -l 'archive (1).zip'` shows one CSV file:

- `math_500_test.csv`

Sample parsed fields are capitalized:

- `Question`
- `Answer`

Sample rows contain olympiad/contest-style math prompts with symbolic LaTeX solutions and boxed final answers.

Validation outcome:

- `valid_gsm8k = no`
- `valid_math500 = yes`
- `uncertain = no`
- reason: MATH500-like symbolic/problem-solution patterns.

Normalization outcome:

- normalized to `data/math500_uploaded_normalized.jsonl`
- rows written: `500`

## 3) Schema differences vs repo loaders

Observed uploaded schema differs from canonical loader input in two ways:

1. **GSM8K ZIP is parquet-based**, not JSON/JSONL.
2. **MATH500 CSV uses `Question`/`Answer` capitalization**, not lowercase `question`/`answer`.

The validation/normalization pipeline now handles parquet + case-insensitive field extraction and writes canonical JSONL rows.

## 4) Canonical normalized format used

### GSM8K normalized schema

- `question_id`
- `question`
- `gold_answer`
- `answer_mode` = `numeric`

### MATH500 normalized schema

- `question_id`
- `question`
- `gold_answer`
- `answer_mode` = `math`

## 5) Repo usability status

Both uploaded ZIPs are now validated and ingested into local canonical files usable by this repository's dataset loading path.

Artifacts generated:

- `outputs/dataset_validation/validation_summary.json`
- `outputs/dataset_validation/gsm8k_sample_preview.json`
- `outputs/dataset_validation/math500_sample_preview.json`
- `outputs/dataset_validation/offline_checks.json`
- `data/gsm8k_uploaded_normalized.jsonl`
- `data/math500_uploaded_normalized.jsonl`
