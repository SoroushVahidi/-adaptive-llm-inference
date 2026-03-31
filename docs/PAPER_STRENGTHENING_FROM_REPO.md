# Paper Strengthening From Repository

This note is written without access to the manuscript. It summarises, based solely on
repository artifacts, what the paper likely claims, what new evidence was added in this
coding pass, the main caveats, and which new artifacts are most suitable for a revision.

---

## Likely Paper Claim (Inferred from Repository)

Based on repository structure, dataset names, policy naming conventions, output files, and
documentation, the paper likely claims:

> **Selective escalation with a lightweight adaptive routing policy can recover most of the
> accuracy gain of Always-DPR at a fraction of its cost, across multiple reasoning regimes.**

More specifically:
- The paper studies *cost-aware adaptive routing* for LLM reasoning, where a cheap
  (reasoning-greedy, cost=1) and an expensive (direct+revise, cost=2) action are available.
- An adaptive routing policy (trained or rule-based, using pre-computed text features)
  decides per-query whether to escalate.
- The main metric space is accuracy vs. average cost, and the goal is to sit on the
  Pareto frontier between Always-RG (low cost, lower accuracy) and Always-DPR (high cost,
  higher accuracy).
- An oracle upper bound (per-query optimal routing) quantifies remaining headroom.
- Multiple regimes are tested: GSM8K random, hard-filtered GSM8K, MATH-500, and AIME 2024.

Key supporting evidence already in repo:
- Adaptive v5 matches DPR accuracy on hard_gsm8k_100/b2 and gsm8k_random_100, at 1.29–1.53 cost vs. 2.0.
- Adaptive v6 achieves similar accuracy with lower cost (1.03–1.27), especially on easy regimes.
- On math500_100, the gain over Always-RG is small (0.64 → 0.65–0.66) but the oracle gap is 0.04–0.06.
- Oracle accuracy approaches or equals DPR on all four regimes, confirming the routing headroom.

---

## Strongest New Lightweight Evidence Added in This Pass

### 1. Bootstrap 95% Confidence Intervals (`bootstrap_accuracy_ci.csv`)
- Per-policy accuracy CIs for all 4 regimes from 10,000 bootstrap resamples (seed=42).
- Allows reviewers to assess whether accuracy differences are statistically plausible given n=100.
- Key finding: adaptive v5 CI [0.79, 0.92] on hard_gsm8k_100 overlaps with DPR CI [0.79, 0.92],
  consistent with the claim that v5 is competitive with DPR at lower cost.

### 2. Paired Difference Tests (`paired_difference_tests.csv`)
- Paired bootstrap tests for best-adaptive vs. Always-RG and oracle vs. best-adaptive.
- Shows that in hard_gsm8k_b2, adaptive v5 is significantly better than RG (p=0.0002), while
  the oracle gap is non-significant (p=0.73), indicating near-oracle performance.
- On math500_100, the oracle gap is significant (p=0.033), showing real room for improvement.

### 3. Routing Outcome Breakdown (`routing_outcome_breakdown.csv` + stacked bar figure)
- Quantifies the fraction of queries where each routing outcome occurs.
- "DPR-only correct" = routing headroom: 7–12% on hard regimes, 2% on easy regimes.
- Supports the narrative that adaptive routing is most valuable on hard regimes.

### 4. Oracle Headroom Table (`oracle_headroom_table.csv`)
- Directly shows adaptive_to_oracle_gap, rg_to_oracle_gap, dpr_to_oracle_gap.
- Best adaptive (v5) closes the gap to oracle to 0 on gsm8k_random_100, 0.01 on hard_gsm8k_b2.

### 5. Cost Ratio Sensitivity (`cost_ratio_sensitivity.csv` + `policy_ranking_stability.csv`)
- Shows that the best adaptive policy (v5 on hard regimes) remains best across 1:1.5, 1:2, 1:3 cost ratios.
- Ranking is stable: v5 consistently dominates v6/v7 on hard regimes in all cost assumptions.

### 6. Policy Efficiency Table (`policy_efficiency_table.csv`)
- Reports `gain_per_extra_cost_unit` and `frac_dpr_accuracy_recovered`.
- On hard_gsm8k_b2: v5 recovers 100% of DPR accuracy while avoiding 59% of DPR cost.
- On math500_100: v6 achieves 50% accuracy recovery while avoiding 97% of DPR cost (very efficient).

---

## Main Caveats That Remain

1. **Small sample sizes (n=100 per regime):** All regimes have exactly 100 queries.
   Bootstrap CIs are wide (±5–9 percentage points). Claims about 1–2 pp improvements are
   not statistically distinguishable from noise. This is now explicit in the CI table.

2. **math500_100 regime is weak:** The adaptive-vs-RG gain is only 0.01–0.02. DPR does
   not help at all (0.64 = RG = DPR). The oracle gap (0.06) suggests some headroom exists
   but neither DPR nor any adaptive policy captures it. This should be acknowledged in the paper.

3. **AIME 2024 and GPQA are incomplete:** These regimes have routing data but no policy
   evaluation. The paper should either present AIME/GPQA as preliminary or acknowledge
   they were not included in the main analysis.

4. **Adaptive policy v5 is a rule-based heuristic:** The "learning" is in the feature
   engineering and threshold calibration, not a trained ML model. The learned-router
   baselines (LR, DT) show that pure ML matching is not straightforwardly better.

5. **Feature leakage risk:** The pre-computed features in enriched CSVs include oracle
   signals (e.g., `revise_helpful`). The adaptive policies themselves use only input features
   (question text + first-pass output), but this distinction should be clearly stated.

---

## Most Suitable New Tables/Figures for a Revision

| Artifact | Usefulness | Recommended Section |
|---|---|---|
| `routing_headroom_barplot.png` | High – clear main results visual | Main results / intro figure |
| `oracle_headroom_table.csv` | High – shows remaining gap concisely | Main results table |
| `bootstrap_accuracy_ci.csv` | High – required for credibility | Appendix / results narrative |
| `paired_difference_tests.csv` | High – answers reviewer "is it significant?" | Results / supplementary |
| `routing_outcome_stacked_bar.png` | Medium – supports headroom interpretation | Analysis / discussion |
| `adaptive_efficiency_scatter.png` | Medium – shows frontier across regimes | Results figure |
| `policy_efficiency_table.csv` | Medium – DPR-recovery framing | Results table or appendix |
| `cost_ratio_sensitivity.csv` | Medium – robustness check | Supplementary |
| `policy_ranking_stability.csv` | Low-Medium – ranking unchanged, confirms robustness | Supplementary |
| `threshold_tradeoff_curve.png` | Low – already present implicitly in sweep data | Appendix |
