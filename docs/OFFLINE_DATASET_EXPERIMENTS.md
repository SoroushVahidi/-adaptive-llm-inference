# Offline Dataset Experiments (Uploaded ZIP Path)

## What was run
1. Uploaded ZIP discovery + validation pipeline.
2. Normalization gate (only runs when validation succeeds).
3. Dataset-validation artifact generation.

Command used:

```bash
python3 scripts/validate_uploaded_datasets.py
```

## Datasets used
- Uploaded GSM8K ZIP: not found.
- Uploaded MATH500 ZIP: not found.

## What succeeded
- The new validation pipeline executed successfully.
- Validation summary and preview output files were generated.
- Graceful blocked behavior for missing ZIPs worked.

## What was blocked
- GSM8K uploaded-data ingestion and offline feature/loader checks (blocked: archive missing).
- MATH500 uploaded-data ingestion and offline feature/loader checks (blocked: archive missing).
- Any downstream offline experiments that require uploaded normalized files.

## Meaning for next stage
The repository now has a reproducible validator/normalizer path, but uploaded-data experiments cannot proceed until the ZIP files are physically present in this checkout.

Once the ZIPs are present and validated, the next offline stage should run:
- normalization to `data/gsm8k_uploaded_normalized.jsonl` and/or `data/math500_uploaded_normalized.jsonl`,
- loader sanity checks on normalized files,
- lightweight feature extraction compatibility checks,
- routing dataset generation compatibility checks where applicable.
