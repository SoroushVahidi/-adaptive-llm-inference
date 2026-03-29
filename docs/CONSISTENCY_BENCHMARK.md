# Consistency Benchmark (Old vs Role vs Unified)

## Setup

- Dataset: `data/consistency_benchmark.json`
- Size: 20 questions, 60 candidates (40 wrong, 20 correct)
- Variants compared:
  - `old_checker`
  - `raw_role_checker`
  - `calibrated_role_checker`
  - `unified_error_checker`

## Measured metrics (offline)

| checker | wrong recall | FPR on correct | recall-FPR |
|---|---:|---:|---:|
| old_checker | 0.725 | 0.05 | 0.675 |
| raw_role_checker | 0.975 | 0.85 | 0.125 |
| calibrated_role_checker | 0.85 | 0.40 | 0.45 |
| unified_error_checker | 0.95 | 0.60 | 0.35 |

## Key failure-type recalls

| failure type | old | raw-role | calibrated-role | unified |
|---|---:|---:|---:|---:|
| intermediate_as_final | 0.00 | 1.00 | 0.50 | 1.00 |
| wrong_target_quantity | 0.545 | 1.00 | 0.818 | 1.00 |
| rate_vs_total | 1.00 | 1.00 | 1.00 | 1.00 |
| total_vs_remaining | 1.00 | 1.00 | 1.00 | 1.00 |

## Interpretation

- Unified integration improves targeted recall over calibrated-role, but currently regresses FPR materially.
- This means unified checker is currently recall-oriented and still exploratory as a routing gate.

## Evidence labels

- Synthetic benchmark numbers: **measured_now**
- Real API-based routing impact: **blocked**
- Practical deployment claim: **exploratory_only**
