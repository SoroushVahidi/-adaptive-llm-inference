# CODEX Critical State Review

## 1) Current project goal

The project goal is adaptive test-time compute/routing for reasoning under cost constraints (not just raw accuracy), with practical evaluation on reasoning datasets such as GSM8K/MATH500 and comparison against fixed strategies and routing policies.

## 2) Currently implemented methods (code exists)

Implemented and wired methods are broad:

- Strategy runners include direct, reasoning, verify/revise, sampled variants, and oracle-style wrappers used across evaluators.
- Multiple adaptive policy generations (v1–v5) and router-related evaluation infrastructure exist in-repo (code + tests).
- Feature families are heavily implemented: query features, target-quantity, constraint-violation, number-role/calibrated role, self-verification, selective prediction/calibration, step verification, and unified error signals.
- Uploaded dataset validation/normalization supports JSON/JSONL/CSV/parquet and canonical JSONL outputs for GSM8K/MATH500.
- Real-routing build/eval scaffolding exists and is CLI-invokable.

## 3) Currently implemented datasets/loaders

- GSM8K and MATH500 loaders support HuggingFace source plus local files, including normalized JSONL path usage (`question_id`/`gold_answer`).
- Uploaded ZIP validation currently detects both uploaded archives and normalizes them to:
  - `data/gsm8k_uploaded_normalized.jsonl` (17584 rows)
  - `data/math500_uploaded_normalized.jsonl` (500 rows)
  when run locally via script.

## 4) Currently measured results (hard evidence now)

### Measured-now (reproducible from current run)

- Uploaded archive validation succeeds and normalization succeeds for both datasets via `scripts/validate_uploaded_datasets.py`.
- Real GSM8K routing dataset build is currently blocked in this environment due missing `OPENAI_API_KEY`; script emits explicit blocked summary artifact instead of fake data.
- Learned routing model eval is currently blocked because dataset CSV was not produced, and also has explicit sklearn gating.

### What is *not* measured-now

- No committed repository evidence shows real query-level routing performance gains over `reasoning_greedy`.
- No committed repository evidence shows learned routing model metrics on real GSM8K rows.
- Prior narrative result docs exist, but without committed output artifacts they are not strong empirical evidence by themselves for publication-grade claims.

## 5) Currently blocked results

Blocked paths are explicit in code:

- Real inference path requires `OPENAI_API_KEY` and model access; missing key hard-blocks routing dataset creation.
- Learned model phase requires both real dataset CSV and sklearn availability; either missing condition blocks model training/eval.

In the present environment, both blockers are active (no API key, no sklearn), so there are no new real-routing metrics yet.

## 6) Strongest evidence-backed conclusion so far

**The strongest evidence-backed conclusion is operational, not scientific:**

The repository is implementation-rich and can validate/normalize real uploaded GSM8K/MATH500 data, but it still lacks measured real query-level routing evidence because live inference is blocked. So the scientific claim (“adaptive routing improves accuracy-cost tradeoff”) remains unproven in current measurable evidence.

## 7) Current unresolved bottleneck

**Single bottleneck:** lack of real per-query strategy outcomes on GSM8K (especially paired `reasoning_greedy` vs `direct_plus_revise`) due blocked live inference.

Without this table, every downstream component (learned routing, policy comparisons, feature utility claims) is effectively speculative infrastructure.

## 8) What appears unnecessary/redundant *right now*

Critical prioritization judgment:

1. **Adding more feature families right now is low-value.** Feature breadth is already high; label scarcity is the limiting factor.
2. **Further synthetic-benchmark elaboration is low-value for the main claim.** Useful for smoke tests, but no longer the core training evidence for routing.
3. **Expanding adaptive-policy variants before collecting real routing rows is low-value.** v1–v5 policy complexity already exceeds current empirical grounding.
4. **External-baseline integration (TALE/BEST-Route) should be deprioritized until the core real-routing table exists.** Otherwise effort shifts to engineering breadth without answering the central question.

Bottom line: stop adding mechanism; start collecting real paired outcomes.
