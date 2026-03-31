# Cost-ratio sensitivity (lightweight recomputation)

Using existing per-query decisions only, adaptive-vs-baseline ordering by accuracy is unchanged across 1:1.5, 1:2, and 1:3.
Higher expensive-model cost increases adaptive-policy cost advantage relative to always-revise whenever revise_rate < 1.