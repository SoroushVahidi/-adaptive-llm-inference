# Offline Dataset Experiments (Uploaded ZIP Validation Path)

Date run: **2026-03-29 (UTC)**

## Experiments/checks run (offline only)

## 1) Uploaded archive validation + normalization

Command:

```bash
python3 scripts/validate_uploaded_datasets.py
```

Result:

- both uploaded ZIPs discovered
- GSM8K archive validated and normalized
- MATH500 archive validated and normalized
- summary + previews written under `outputs/dataset_validation/`

## 2) Loader compatibility checks on normalized files

Command:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from src.datasets.gsm8k import load_gsm8k
from src.datasets.math500 import load_math500
from src.features.precompute_features import extract_query_features

out_dir=Path('outputs/dataset_validation')
out_dir.mkdir(parents=True, exist_ok=True)
checks={}

gsm=load_gsm8k(data_file='data/gsm8k_uploaded_normalized.jsonl', max_samples=10)
math=load_math500(data_file='data/math500_uploaded_normalized.jsonl', max_samples=10)
checks['gsm8k_loader_sample_count']=len(gsm)
checks['math500_loader_sample_count']=len(math)
checks['gsm8k_first_id']=gsm[0].id if gsm else None
checks['math500_first_id']=math[0].id if math else None
checks['gsm8k_feature_example']=extract_query_features(gsm[0].question)
checks['math500_feature_example']=extract_query_features(math[0].question)
(out_dir/'offline_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
PY
```

Result:

- normalized JSONL files load successfully via repository loaders
- query feature extraction runs on both datasets for sampled rows
- check artifact written: `outputs/dataset_validation/offline_checks.json`

## 3) Automated validation tests

Command:

```bash
pytest -q tests/test_uploaded_dataset_validation.py tests/test_math500_loader.py
```

Result:

- all tests passed

## What was blocked

No API-based experiments were run by design.

The following remain intentionally blocked in this task because they require model inference/API or oracle files beyond raw datasets:

- full strategy-comparison evaluations using external/hosted LLM calls
- oracle-dependent routing evaluations that require precomputed model outputs

## What this means for next stage

The uploaded dataset assets are now ready for local/offline integration work:

- local canonical data files are available
- loader path supports normalized JSONL inputs
- validation and ingestion are reproducible via script

You can proceed to model/evaluation phases once the intended offline/online inference backend is selected.
