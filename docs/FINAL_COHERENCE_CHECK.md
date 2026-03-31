# Final Coherence Check

Date: 2026-03-31  
Method: lightweight repository-grounded checks over canonical final assets; no new experiments.

## Checks performed

1. Policy names in `outputs/paper_tables_final/policy_comparison_main.csv`
2. Regime ids in final tables
3. Canonical policy choice alignment in `outputs/paper_tables_final/main_results_summary.csv`
4. Presence of all required final figures and graphic abstract files
5. README top quickstart path hygiene (no noncanonical cleaned-directory references)
6. Graphic abstract caption alignment with canonical story keywords

## Results

- Pass: policy set matches canonical six policies  
  (`reasoning_greedy`, `adaptive_policy_v5`, `adaptive_policy_v6`, `adaptive_policy_v7`, `direct_plus_revise`, `oracle`)
- Pass: regime set matches canonical four  
  (`gsm8k_random_100`, `hard_gsm8k_100`, `hard_gsm8k_b2`, `math500_100`)
- Pass: `adaptive_primary_policy` is consistently `adaptive_policy_v5`
- Pass: all required final figure/graphic-abstract files are present in `outputs/paper_figures_final/`
- Pass: README top quickstart does not reference `paper_tables_cleaned` or `paper_figures_cleaned`
- Pass: graphic abstract caption text matches canonical story framing

## Mismatches found

None.

## Conclusion

Canonical final asset family is coherent and internally consistent for manuscript support.
