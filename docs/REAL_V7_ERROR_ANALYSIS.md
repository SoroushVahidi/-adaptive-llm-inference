# Real V7 Error Analysis (100-query GSM8K run)

**Source:** `outputs/real_policy_eval/per_query_policy_decisions.csv` joined with `outputs/real_routing_dataset/gsm8k_per_query_outputs.csv`.  
**Evidence:** **measured_now**.

**Definitions:**

- **False positive (routing):** `policy_v7 == direct_plus_revise` and `reasoning_correct == 1` (correct without second stage; paid extra cost).
- **False negative (routing):** `policy_v7 != direct_plus_revise` and `revise_helpful == 1` (should have revised).

---

## False negatives (V7 missed revise when helpful)

**Count:** **0** on this run.  
Both queries with `revise_helpful=1` (`gsm8k_test_39`, `gsm8k_test_98`) received `direct_plus_revise` from V7.

---

## False positives (V7 revised when reasoning already correct)

**Count:** **25** on this run. Below are **5** examples (full text from collected CSV).

### 1. `gsm8k_test_0`

- **Gold:** 18  
- **Reasoning answer:** 18  
- **Revise answer:** 18  
- **Policy:** `direct_plus_revise`  
- **Why wrong:** V7 (and V6) escalation heuristics fired on the **reasoning trace** despite a **correct** parsed final — template signals / unified path still recommend revise on some long or “busy” outputs.  
- **Lesson:** **Accuracy-only** comparison hides **wasted second-stage cost**; need **cost-aware** evaluation when reasoning is already right.

### 2. `gsm8k_test_4`

- **Gold:** 20  
- **Reasoning answer:** 20  
- **Revise answer:** 20  
- **Policy:** `direct_plus_revise`  
- **Why wrong:** Same pattern — correct first pass, heuristic revise trigger still on.  
- **Lesson:** Thresholds tuned on **snapshots** can **over-revise** on easy GSM8K items in a **strong** model regime.

### 3. `gsm8k_test_8`

- **Gold:** 45  
- **Reasoning answer:** 45  
- **Revise answer:** 45  
- **Policy:** `direct_plus_revise`  
- **Why wrong:** Long multi-step story → high **unified / role / constraint** proxy scores despite correct final.  
- **Lesson:** **Length and cue density** ≠ **wrong answer**; policies need **tighter coupling** to **parsed final vs gold** before paying revise.

### 4. `gsm8k_test_14`

- **Gold:** 60  
- **Reasoning answer:** 60  
- **Revise answer:** 60  
- **Policy:** `direct_plus_revise`  
- **Why wrong:** Correct reasoning; heuristics still escalate.  
- **Lesson:** Percent / multi-clause word problems trigger **defensive** routing.

### 5. `gsm8k_test_15`

- **Gold:** 125  
- **Reasoning answer:** 125  
- **Revise answer:** 125  
- **Policy:** `direct_plus_revise`  
- **Why wrong:** Correct; still revised.  
- **Lesson:** **Financial / percent** wording + multiple dollar figures may inflate **suspicion** without error.

---

## Summary

| Error type | Count (N=100) |
|------------|----------------|
| V7 false negative (missed revise_helpful) | 0 |
| V7 false positive (revise despite reasoning correct) | 25 |

**Main takeaway:** On this slice, V7 **does not** improve accuracy over V6 but **increases** unnecessary revise vs V6 (0.30 vs 0.18 revise rate), matching the FP pattern above.
