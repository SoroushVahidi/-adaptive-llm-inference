# BEST-Route Status (Repo-Truth Audit)

## Verdict
- **Official BEST-Route is not runnable end-to-end in this repository state.**
- **Reason:** official repository clone expected at `external/best_route/.repo` is missing, and official bridge is intentionally unimplemented pending the full multi-stage pipeline.

## Exact blocker evidence
- `BESTRouteBaseline.solve()` raises a `RuntimeError` when official code is not installed.
- After installation check, wrapper still raises `NotImplementedError` indicating the official bridge has not been completed.
- Direct runtime check in this run: `installed False` and runtime message requiring clone into `/workspace/-adaptive-llm-inference/external/best_route/.repo`.

## Adapted alternative
- `BESTRouteAdaptedBaseline` is implemented and tested, but it is a documented compatibility adaptation (binary routing + heuristic router), not a faithful official BEST-Route reproduction.
- Therefore it is reported separately and **not** used to claim official BEST-Route performance.
