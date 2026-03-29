# Uploaded Dataset Validation Report

## Scope
This report validates uploaded ZIP archives claimed to contain GSM8K and MATH500.

## Discovery result
Validation run searched the repository recursively for `*.zip` and found **no ZIP files**.

- `num_zip_files_found = 0`
- status: `blocked`
- block reason: `No ZIP files were found in repository.`

Because no ZIPs were found, dataset-specific validation could not be completed for either GSM8K or MATH500.

## Per-dataset conclusion

### GSM8K uploaded ZIP
- Found: **No**
- `valid_gsm8k`: **No** (not validated; no archive found)
- `uncertain`: **Yes**
- Reason: no uploaded ZIP artifact was discoverable in this checkout.
- Normalization: not run.
- Repo usability: not updated from uploaded ZIP path.

### MATH500 uploaded ZIP
- Found: **No**
- `valid_math500`: **No** (not validated; no archive found)
- `uncertain`: **Yes**
- Reason: no uploaded ZIP artifact was discoverable in this checkout.
- Normalization: not run.
- Repo usability: not updated from uploaded ZIP path.

## Schema differences observed
No uploaded archive schema was available to inspect.

## Outputs written
- `outputs/dataset_validation/validation_summary.json`
- `outputs/dataset_validation/gsm8k_sample_preview.json`
- `outputs/dataset_validation/math500_sample_preview.json`

## Next action required
Place the two uploaded ZIP archives in the repository (or provide exact paths), then rerun:

```bash
python3 scripts/validate_uploaded_datasets.py
```
