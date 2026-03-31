# Signal Ablation Results (Offline, Repository-Grounded)

## Experiment summary

We ran an offline ablation using existing enriched routing datasets and per-query correctness labels only (no new model calls), comparing explanation-warning-focused versus answer-error-focused revise triggers.

Primary artifacts:
- `outputs/signal_ablation/policy_comparison.csv`
- `outputs/signal_ablation/per_query_decisions.csv`
- `outputs/signal_ablation/regime_summary.csv`
- `outputs/signal_ablation/summary.json`
- `outputs/signal_ablation/bootstrap_ci.csv`
- `outputs/signal_ablation/paired_tests.csv`
- `outputs/paper_tables/signal_ablation_main_table.csv`

---

## Variants evaluated

- `explanation_only_router`
- `answer_error_only_router`
- `combined_equal_router`
- `answer_error_dominant_router`
- `explanation_dominant_router`

All variants choose between `reasoning_greedy` (cost 1) and `direct_plus_revise` (cost 2).

---

## Main empirical answers

## 1) Does answer-error-focused routing outperform explanation-warning-focused routing?

### Strong support (in this repository’s four regimes)

Yes on 3/4 regimes by accuracy, and tie on 1/4:

- `gsm8k_random_100`: best answer-focused 0.92 vs best explanation-focused 0.90.
- `hard_gsm8k_100`: 0.81 vs 0.79.
- `hard_gsm8k_b2`: 0.89 vs 0.84.
- `math500_100`: tie 0.66 vs 0.66.

Thus, answer-error-focused variants are never worse in best-vs-best regime comparisons here.

### Paired uncertainty (lightweight)

- Hard GSM8K b2 difference (+0.05) has positive CI and low p-value in paired bootstrap.
- Other regime differences are directionally positive but sample-limited / not decisive.
- Math500 tie is exact under this setup.

---

## 2) In which regimes is the difference strongest?

Strongest in **hard_gsm8k_b2** (+5 points for best answer-focused vs best explanation-focused). Moderate in `gsm8k_random_100` and `hard_gsm8k_100` (+2 points each). Near-zero in `math500_100`.

This indicates the largest gain where revise-helpful prevalence is moderate and explanation-only gating misses helpful escalations.

---

## 3) Does explanation-warning routing mainly over-escalate?

### Partial support only

Not uniformly. In these runs, explanation-focused variants tend to be **conservative / under-escalating** in hard regimes (high false negatives), not massively over-escalating:

- `explanation_only_router` has very low revise rates (0.00, 0.02, 0.00 across GSM8K random / hard / hard_b2), with high missed revise-helpful counts on hard sets.
- `explanation_dominant_router` increases revise modestly, but still underperforms answer-focused variants on hard regimes.

So the stronger claim is: explanation-warning-focused routing is a weaker revise trigger (often too conservative in this proxy setup), not simply an over-escalation trigger.

---

## 4) Do combined variants help, or reintroduce false positives?

### Partial support

- `combined_equal_router` is generally between explanation-only and answer-only performance.
- On hard regimes it improves over explanation-only but remains below answer-focused best variants.
- It can reduce cost versus answer-only but at an accuracy tradeoff.

Hence, equal combination does not dominate answer-focused routing in this experiment.

---

## 5) How this supports or complicates the V5→V6→V7 story

## Supports

- Supports the core V6 design claim: answer-error signals are better primary revise drivers than generic explanation warnings.
- Consistent with the rationale for separating explanation quality from answer wrongness.

## Complicates

- Gains are regime-dependent and modest in some settings.
- Math500 shows tie behavior among best explanation/answer variants, suggesting that answer-error dominance is not universally stronger in every regime.

---

## 6) Safe manuscript claim from this experiment

A conservative claim supported by this ablation:

> In offline evaluations over four 100-query regimes, routers keyed primarily to answer-error signals matched or exceeded the best explanation-warning-focused routers in all regimes (strictly better in three, tied in one), with the largest benefit on hard GSM8K b2.

---

## 7) Caveats and limitations

## Strong caveats

1. This is a proxy ablation on precomputed aggregate scores (`v6_answer_error_score`, `v6_explanation_warning_score`), not full recomputation of all internal policy sub-signals.
2. Each regime has n=100; uncertainty remains for small differences.
3. Thresholds are fixed and grounded in V6 defaults; no independent train/validation split was used for threshold tuning.
4. Results are correlational for routing design choice; no causality claim about individual features.

## Unsupported claims

- Universal superiority of answer-error-only routing across all tasks/domains/models.
- Causal attribution that any single signal family alone “causes” gains.

---

## Claim strength rubric for this experiment

- **Strong support:** answer-error-focused > explanation-focused in hard_gsm8k_b2; non-inferior overall across four regimes.
- **Partial support:** hard-regime advantage as a general trend (present but moderate in one hard split).
- **Unsupported:** broad, domain-general dominance and causal feature claims.

---

## Figure note

A figure was attempted in the script (`outputs/paper_figures/signal_ablation_summary.png`) but plotting dependency support may be unavailable in this environment. The CSV artifacts are complete and sufficient for manuscript tables.
