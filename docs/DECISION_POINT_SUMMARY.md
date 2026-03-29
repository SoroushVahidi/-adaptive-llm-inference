# Decision Point Summary

## One-paragraph status
The repository is scientifically **implementation-rich but evidence-light**: strategies, adaptive policies (v1–v4), feature families, routing dataset tooling, and diagnostics are all present and tested, but committed experiment output artifacts are largely absent, so most performance conclusions remain code- or doc-level rather than reproducible empirical evidence.

## Strongest evidence-backed conclusions
1. Core experimentation infrastructure exists across datasets, strategies, policies, and analysis pipelines.
2. Unit tests provide good confidence that implemented logic executes as intended.
3. The main blocker is not missing architecture; it is missing committed empirical evidence for comparative claims.

## Weakly supported conclusions
1. Any claim that one strategy definitively dominates another on cost-quality tradeoff.
2. Any claim that hand-crafted routing (v1–v4) has already beaten `reasoning_greedy`.
3. Any claim that target-quantity/constraint-aware features have proven downstream gains.

## Main problem to think about now
Can we produce a **reliable, reproducible routing advantage** over a strong cheap baseline (`reasoning_greedy`) under budget constraints, using current signals and policy families?

## Is evidence enough to pause experiments and think?
**Yes.** There is enough structural and code-level evidence to pause broad experimentation and focus on the central scientific decision (signal adequacy and routing value) before expanding method scope.

## Three things likely not worth doing right now
1. Adding more strategy variants before establishing reproducible evidence on current shortlist.
2. Deep hand-tuning of new heuristic rules without oracle-labeled outputs to validate signal value.
3. Writing stronger paper claims from doc narratives without committed result artifacts.
