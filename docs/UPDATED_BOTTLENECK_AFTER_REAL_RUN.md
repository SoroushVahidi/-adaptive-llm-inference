# Bottleneck After First Real GSM8K Routing Run

**Evidence mix:** **measured_now** (100-query CSVs/summaries) + **exploratory_only** (what to do next).

---

## 1. Strongest cheap baseline (real)

**reasoning_greedy** — **90%** exact-match on this 100-query test slice (`outputs/real_policy_eval/summary.json`).

## 2. Strongest corrective baseline (real)

**direct_plus_revise** (direct + self-revision in repo) — **92%** accuracy, cost proxy **2** per query.

## 3. Prevalence of revise_helpful

**2 / 100 = 2%** (`revise_helpful` = wrong reasoning, correct revise). Too sparse for stable learned routing in this draw.

## 4. Is V7 a meaningful real improvement over V6?

**On accuracy:** **No** — both **0.92** on the same 100 rows.  
**On cost:** V7 **higher** revise rate (0.30 vs 0.18) with **same** accuracy → **worse** cost–quality tradeoff on this slice.  
**On the two revise_helpful cases:** Both V6 and V7 **chose revise** (no false negative on those two).

## 5. Is revise-worthiness learnable here?

**Not from this table alone** — only **2** positives; CV models get **F1 = 0** (always-negative collapse). **Blocked** for claim-ready learning until label balance improves.

## 6. Main bottleneck now

1. **Label scarcity** for `revise_helpful` at realistic scales when the base model is already strong (90% one-shot reasoning).  
2. **Policy cost** — heuristic revise triggers (especially V7) add cost without accuracy gain on this slice.  
3. **Objective alignment** — `direct_plus_revise` ≠ “revise after reasoning_greedy”; paper narrative should state the paired pipeline used.

## 7. Deprioritize

- **Further heuristic threshold tuning** on tiny slices without cost–accuracy frontiers.  
- **Heavy learned-router work** until **O(10²)** revise_helpful positives or curated hard subset.

## 8. Prioritize

- Larger or **hard-filtered** subsets (e.g. problems where reasoning accuracy < 70%).  
- **Budgeted** metrics: accuracy under fixed average cost, not accuracy alone.  
- Optional: align paired pipeline with paper story (reasoning first, then revise).
