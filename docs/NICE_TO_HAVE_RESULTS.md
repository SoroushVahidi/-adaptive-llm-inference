# NICE-TO-HAVE Lightweight Results

## What was actually worth doing
- **Headroom decomposition refinement**: worth doing; directly sharpens where routing can and cannot help.
- **Policy efficiency refinement**: worth doing; provides clean systems-facing efficiency ratios.
- **Cost-ratio sensitivity**: worth doing as a robustness check; confirms conclusions are not tied to a single price ratio.
- **Cross-regime stability summary**: worth doing to tighten claim scope by regime.

## What strengthened the paper meaningfully
- Hard-regime DPR-only success mass is materially above easy-regime mass (hard-regime average 0.105 vs easy-regime 0.020), supporting regime-dependent headroom claims.
- Best adaptive policies often recover large DPR gains with less than full DPR cost, improving practical deployment framing.

## Noisy or weaker evidence
- Easy-regime gain-recovery percentages are weakly informative because DPR gain itself is near zero; report cautiously.
- Boolean cross-regime summaries are descriptive and should not be over-interpreted as causal.

## Placement recommendations
- **Main text candidates**: headroom decomposition refinement; policy efficiency refinement.
- **Appendix candidates**: cost-ratio sensitivity; cross-regime stability summary.
- **Omit**: no implemented result should be fully omitted, but avoid emphasizing easy-regime gain-recovery ratios.

## Top-2 main-paper results
1. Hard-regime headroom decomposition shows meaningful DPR-only recoverable mass while also quantifying irreducible both-wrong cases.
2. Efficiency refinement shows adaptive routing can retain substantial revise benefit while avoiding notable revise cost.

## Top-2 appendix-only results
1. Cost-ratio sensitivity at 1:1.5, 1:2, 1:3.
2. Cross-regime stability matrix for claim calibration.

## Results to ignore
- Any single-number easy-regime gain-recovery percentage that might look unstable due to tiny denominator headroom.