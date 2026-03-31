#!/usr/bin/env python3
"""Run the feature-method-fit offline analysis and generate all outputs.

This script builds a unified per-query analysis dataset from the four main
manuscript regimes, derives the 15 candidate features defined in
docs/FEATURE_METHOD_FIT_EXPERIMENT.md, runs lightweight univariate and
multivariate analyses, and writes:

  outputs/feature_method_fit/
    feature_analysis_dataset.csv       — one row per query, 15 features + labels
    univariate_feature_summary.csv     — feature × outcome group means
    target_quantity_type_breakdown.csv — target_quantity_type × outcome rates
    method_fit_descriptive_summary.csv — method × feature means
    revise_helpful_model_summary.csv   — logistic regression results
    method_fit_model_summary.csv       — decision tree results
    feature_importance.csv             — LR coef + DT importance
    final_feature_ranking.csv          — manuscript feature ranking

  outputs/paper_tables/
    feature_method_fit_main_table.csv  — compact manuscript-ready table

  outputs/paper_figures/
    feature_method_fit_summary.png     — bar chart of top-feature effect sizes

  docs/
    FEATURE_METHOD_FIT_EXPERIMENT_RESULTS.md

No API calls, no new LLM inference. Uses only existing repo artifacts.

Usage:
    python3 scripts/run_feature_method_fit_analysis.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path when run as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.feature_method_fit import (  # noqa: E402
    OUTCOME_LABELS,
    run_feature_method_fit_analysis,
)

_SEP = "=" * 72


def _print_section(title: str) -> None:
    print()
    print(_SEP)
    print(f"  {title}")
    print(_SEP)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _generate_figure(
    univariate: list[dict],
    output_path: Path,
) -> bool:
    """Generate a grouped bar chart of top features by effect size.

    Returns True if successful, False if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    # Sort by effect_size_proxy descending; exclude target_quantity_type
    sorted_feats = sorted(
        [r for r in univariate],
        key=lambda r: -(float(r.get("effect_size_proxy") or 0)),
    )[:12]  # top 12 for readability

    labels = [r["feature"] for r in sorted_feats]
    rh_means = [float(r.get("mean_revise_helpful") or 0) for r in sorted_feats]
    sc_means = [float(r.get("mean_safe_cheap") or 0) for r in sorted_feats]
    bw_means = [float(r.get("mean_both_wrong") or 0) for r in sorted_feats]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, rh_means, width, label="revise_helpful", color="#d62728", alpha=0.8)
    ax.bar(x, sc_means, width, label="safe_cheap", color="#2ca02c", alpha=0.8)
    ax.bar(x + width, bw_means, width, label="both_wrong", color="#7f7f7f", alpha=0.8)

    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean value (binary) / normalised (numeric)")
    ax.set_title(
        "Feature means by routing outcome class\n"
        "(top 12 features by effect-size proxy across all regimes)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Results report generator
# ---------------------------------------------------------------------------

def _build_results_report(result: dict) -> str:
    """Generate docs/FEATURE_METHOD_FIT_EXPERIMENT_RESULTS.md content."""

    univ = result["univariate"]
    ranking = result["ranking"]
    outcome_counts = result["outcome_counts"]
    by_regime = result["by_regime"]
    n_total = result["n_total"]
    lr = result["logistic_result"]
    dt = result["dt_result"]
    method_fit = result["method_fit"]

    # Sort univariate by effect size
    sorted_univ = sorted(univ, key=lambda r: -(float(r.get("effect_size_proxy") or 0)))

    # Top recommended features
    top_rec = [r for r in ranking if r["manuscript_recommendation"] == "yes"]
    maybe_rec = [r for r in ranking if r["manuscript_recommendation"] == "maybe"]

    # LR coefficients sorted
    coefs = lr.get("coefficients", {})
    sorted_coefs = sorted(coefs.items(), key=lambda x: -abs(x[1]))

    # DT importances sorted
    dt_imp = dt.get("feature_importances", {})
    sorted_imp = sorted(dt_imp.items(), key=lambda x: -x[1])

    def _univ_row(feat: str) -> dict:
        return next((r for r in univ if r["feature"] == feat), {})

    lines: list[str] = []
    a = lines.append

    a("# Feature–Method-Fit Experiment Results")
    a("")
    a("**Date:** auto-generated  ")
    a("**Method:** Offline only — no API calls or new LLM inference.  ")
    a("**Data:** Existing routing datasets + policy-eval outputs.  ")
    a(f"**Total rows:** {n_total} ({', '.join(f'{r}: {n}' for r, n in by_regime.items())})")
    a("")
    a("---")
    a("")
    a("## 0. Dataset and Outcome Summary")
    a("")
    a("### Regime counts")
    a("")
    a("| Regime | N |")
    a("|---|---|")
    for regime, n in by_regime.items():
        a(f"| `{regime}` | {n} |")
    a("")
    a("### Outcome class sizes (across all regimes)")
    a("")
    a("| Outcome label | Count | Rate |")
    a("|---|---|---|")
    for outcome in OUTCOME_LABELS:
        cnt = outcome_counts.get(outcome, 0)
        rate = cnt / n_total if n_total else 0
        a(f"| `{outcome}` | {cnt} | {rate:.3f} |")
    a("")
    a("---")
    a("")
    a("## 1. Which features are most associated with revise-helpful cases?")
    a("")
    a("Features with the highest mean in the `revise_helpful` group "
      "(RG wrong, DPR correct):")
    a("")
    a("| Feature | Side | Mean (revise_helpful) | Mean (safe_cheap) | Effect size proxy |")
    a("|---|---|---|---|---|")
    top_rh = sorted(
        [r for r in univ if r.get("mean_revise_helpful") == r.get("mean_revise_helpful")],
        key=lambda r: -(float(r.get("mean_revise_helpful") or 0)),
    )[:8]
    for r in top_rh:
        a(
            f"| `{r['feature']}` | {r['side']} "
            f"| {r.get('mean_revise_helpful', ''):.4f} "
            f"| {r.get('mean_safe_cheap', ''):.4f} "
            f"| {r.get('effect_size_proxy', ''):.4f} |"
        )
    a("")
    a("**Observation:** Features with high `mean_revise_helpful` and a positive "
      "difference versus `mean_safe_cheap` indicate that when a revise is "
      "genuinely helpful, the routing features are systematically different from "
      "when the cheap route already succeeds.")
    a("")
    a("---")
    a("")
    a("## 2. Which features are most associated with safe cheap routing?")
    a("")
    a("Features with the highest mean in the `safe_cheap` group (RG correct):")
    a("")
    a("| Feature | Side | Mean (safe_cheap) | Mean (revise_helpful) | Effect size proxy |")
    a("|---|---|---|---|---|")
    top_sc = sorted(
        [r for r in univ if r.get("mean_safe_cheap") == r.get("mean_safe_cheap")],
        key=lambda r: -(float(r.get("mean_safe_cheap") or 0)),
    )[:8]
    for r in top_sc:
        a(
            f"| `{r['feature']}` | {r['side']} "
            f"| {r.get('mean_safe_cheap', ''):.4f} "
            f"| {r.get('mean_revise_helpful', ''):.4f} "
            f"| {r.get('effect_size_proxy', ''):.4f} |"
        )
    a("")
    a("---")
    a("")
    a("## 3. Which features are most associated with both-wrong hard cases?")
    a("")
    a("| Feature | Side | Mean (both_wrong) | Mean (safe_cheap) | Effect size proxy |")
    a("|---|---|---|---|---|")
    top_bw = sorted(
        [r for r in univ if r.get("mean_both_wrong") == r.get("mean_both_wrong")],
        key=lambda r: -(float(r.get("mean_both_wrong") or 0)),
    )[:8]
    for r in top_bw:
        a(
            f"| `{r['feature']}` | {r['side']} "
            f"| {r.get('mean_both_wrong', ''):.4f} "
            f"| {r.get('mean_safe_cheap', ''):.4f} "
            f"| {r.get('effect_size_proxy', ''):.4f} |"
        )
    a("")
    a("---")
    a("")
    a("## 4. Which features appear to favor each routing method?")
    a("")
    a("Feature means by best-method group:")
    a("")
    # Table: methods as rows, top 6 features as columns
    top6 = [r["feature"] for r in sorted_univ[:6] if r.get("feature")]
    header = "| method_best_label | n | " + " | ".join(top6) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(top6)) + "|"
    a(header)
    a(sep)
    for mf in method_fit:
        method = mf["method_best_label"]
        n = mf["n"]
        vals = " | ".join(
            str(mf.get(f"mean_{f}", "")) for f in top6
        )
        a(f"| `{method}` | {n} | {vals} |")
    a("")
    a("---")
    a("")
    a("## 5. Do feature results support the answer-error > explanation-warning story?")
    a("")
    r_ae = _univ_row("answer_error_signal")
    r_ew = _univ_row("explanation_warning_signal")
    r_cc = _univ_row("cheap_route_confidence")
    a("Key comparisons:")
    a("")
    a("| Signal | Side | Mean (revise_helpful) | Mean (safe_cheap) | Effect proxy |")
    a("|---|---|---|---|---|")
    for r in [r_ae, r_ew, r_cc]:
        if r:
            a(
                f"| `{r['feature']}` | {r['side']} "
                f"| {r.get('mean_revise_helpful', ''):.4f} "
                f"| {r.get('mean_safe_cheap', ''):.4f} "
                f"| {r.get('effect_size_proxy', ''):.4f} |"
            )
    a("")
    ae_effect = float(r_ae.get("effect_size_proxy") or 0)
    ew_effect = float(r_ew.get("effect_size_proxy") or 0)
    if ae_effect > ew_effect:
        a(
            f"**Result:** `answer_error_signal` has a larger effect-size proxy "
            f"({ae_effect:.4f}) than `explanation_warning_signal` "
            f"({ew_effect:.4f}) for separating `revise_helpful` from `safe_cheap`. "
            f"This supports the story that answer-error-focused signals are more "
            f"discriminative than generic explanation irregularity."
        )
    else:
        a(
            f"**Result:** `explanation_warning_signal` effect ({ew_effect:.4f}) "
            f">= `answer_error_signal` effect ({ae_effect:.4f}). "
            f"The story is **not confirmed** by this analysis on available data."
        )
    a("")
    # Logistic coefficients
    if coefs:
        ae_coef = coefs.get("answer_error_signal", float("nan"))
        ew_coef = coefs.get("explanation_warning_signal", float("nan"))
        a("Logistic regression coefficients (standardised):")
        a("")
        a("| Feature | Coefficient |")
        a("|---|---|")
        for feat, coef in sorted_coefs[:8]:
            a(f"| `{feat}` | {coef:.4f} |")
        a("")
        if ae_coef == ae_coef and ew_coef == ew_coef:
            if abs(ae_coef) > abs(ew_coef):
                a(
                    f"The logistic model also ranks `answer_error_signal` "
                    f"(coef={ae_coef:.4f}) above `explanation_warning_signal` "
                    f"(coef={ew_coef:.4f})."
                )
            else:
                a(
                    f"The logistic model ranks `explanation_warning_signal` "
                    f"(coef={ew_coef:.4f}) above or equal to `answer_error_signal` "
                    f"(coef={ae_coef:.4f})."
                )
    a("")
    a("---")
    a("")
    a("## 6. Which 8–10 features should be kept in the manuscript?")
    a("")
    a("### Recommended features (`yes`)")
    a("")
    a("| Feature | Side | Effect proxy | Interpretability | Notes |")
    a("|---|---|---|---|---|")
    for r in top_rec:
        a(
            f"| `{r['feature_name']}` | {r['question_or_output_side']} "
            f"| {r['effect_size_proxy']} "
            f"| {r['interpretability_score']} "
            f"| {r['redundancy_notes'] or '—'} |"
        )
    a("")
    a("### Conditional features (`maybe`)")
    a("")
    a("| Feature | Side | Effect proxy | Interpretability | Notes |")
    a("|---|---|---|---|---|")
    for r in maybe_rec:
        a(
            f"| `{r['feature_name']}` | {r['question_or_output_side']} "
            f"| {r['effect_size_proxy']} "
            f"| {r['interpretability_score']} "
            f"| {r['redundancy_notes'] or '—'} |"
        )
    a("")
    a("---")
    a("")
    a("## 7. Caveats")
    a("")
    a("1. **Sample size:** Each regime has 100 rows; combined dataset has "
      f"{n_total} rows across 4 regimes. Effect sizes should be treated as "
      "exploratory and order-of-magnitude only.")
    a("")
    a("2. **Class imbalance:** `revise_helpful` is rare (see outcome summary "
      "above). Logistic regression and decision tree results may be dominated "
      "by the majority class (`safe_cheap`). Cross-validated accuracy may be "
      "misleading on imbalanced data.")
    a("")
    a("3. **V6/V7 features only for output-side:** Features F8–F15 are "
      "derived from V6/V7 feature extraction run offline on real first-pass "
      "outputs. Results are specific to the `gpt-4o-mini` model run captured "
      "in the routing datasets.")
    a("")
    a("4. **`copied_question_number_as_final_answer` (F12)** is a question-side "
      "proxy (`tq_potential_answer_echo_risk`) rather than an output-side signal, "
      "as no direct output-side echo detection column exists in the routing CSVs.")
    a("")
    a("5. **`body_final_numeric_mismatch` (F9)** relies on `v7_extra_answer_error`, "
      "which fires only for the two specific V7 patterns (weekday+numeric, "
      "need_more+list_price, tail_equals). Its base rate is very low in most "
      "regimes.")
    a("")
    a("6. **Evidence labels:** All population-level claims carry "
      "`exploratory_only` status per the repo's existing evidence conventions. "
      "The fixture- and probe-level V5/V6/V7 analysis in "
      "`docs/V5_V6_V7_STORY_CHECK.md` remains the more controlled evidence "
      "for the routing-signal story.")
    a("")
    a("---")
    a("")
    a("## Appendix: Decision Tree Feature Importances")
    a("")
    a("| Feature | Importance |")
    a("|---|---|")
    for feat, imp in sorted_imp[:10]:
        a(f"| `{feat}` | {imp:.4f} |")
    a("")
    a("## Appendix: Logistic Regression CV Performance")
    a("")
    if "cv_accuracy_mean" in lr:
        a(
            f"3-fold cross-validated accuracy: "
            f"**{lr['cv_accuracy_mean']:.4f}** "
            f"(±{lr['cv_accuracy_std']:.4f}).  "
        )
        a(
            "Note: this is class-imbalanced data; accuracy may be near the "
            "majority-class baseline."
        )
    else:
        a("(Not available — scikit-learn not installed or insufficient data.)")
    a("")
    a("---")
    a("")
    a("*Report auto-generated by `scripts/run_feature_method_fit_analysis.py`.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run feature-method-fit offline analysis"
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "outputs" / "feature_method_fit"),
        help="Directory for analysis outputs (default: outputs/feature_method_fit/)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    _print_section("Feature–Method-Fit Offline Analysis")
    print(f"  Repo root : {_REPO_ROOT}")
    print(f"  Output dir: {output_dir}")

    # ---- Run analysis -------------------------------------------------------
    result = run_feature_method_fit_analysis(
        repo_root=_REPO_ROOT,
        output_dir=output_dir,
    )

    # ---- Print summary ------------------------------------------------------
    _print_section("Dataset Summary")
    print(f"  Total rows : {result['n_total']}")
    for regime, n in result["by_regime"].items():
        print(f"  {regime:<25}: {n}")
    print()
    print("  Outcome counts:")
    for outcome, cnt in result["outcome_counts"].items():
        rate = cnt / result["n_total"]
        print(f"    {outcome:<35}: {cnt:4d}  ({rate:.3f})")

    _print_section("Top Features by Effect Size (revise_helpful vs safe_cheap)")
    univ_sorted = sorted(
        result["univariate"],
        key=lambda r: -(float(r.get("effect_size_proxy") or 0)),
    )
    print(f"  {'Feature':<45}  {'effect':>8}  {'RH mean':>8}  {'SC mean':>8}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in univ_sorted[:10]:
        print(
            f"  {r['feature']:<45}  "
            f"{r.get('effect_size_proxy', 0):>8.4f}  "
            f"{r.get('mean_revise_helpful', 0):>8.4f}  "
            f"{r.get('mean_safe_cheap', 0):>8.4f}"
        )

    _print_section("Answer-Error vs Explanation-Warning Story Check")
    univ_by_feat = {r["feature"]: r for r in result["univariate"]}
    for feat in ("answer_error_signal", "explanation_warning_signal", "cheap_route_confidence"):
        r = univ_by_feat.get(feat, {})
        print(
            f"  {feat:<45}  "
            f"effect={r.get('effect_size_proxy', 'N/A')}  "
            f"RH={r.get('mean_revise_helpful', 'N/A')}  "
            f"SC={r.get('mean_safe_cheap', 'N/A')}"
        )

    _print_section("Manuscript-Recommended Features")
    for r in result["ranking"]:
        rec = r["manuscript_recommendation"]
        if rec in ("yes", "maybe"):
            sym = "✓" if rec == "yes" else "~"
            print(f"  [{sym}] {r['feature_name']:<45}  effect={r['effect_size_proxy']}")

    # ---- Generate figure ----------------------------------------------------
    _print_section("Figure")
    fig_path = _REPO_ROOT / "outputs" / "paper_figures" / "feature_method_fit_summary.png"
    ok = _generate_figure(result["univariate"], fig_path)
    if ok:
        try:
            rel = fig_path.relative_to(_REPO_ROOT)
        except ValueError:
            rel = fig_path
        print(f"  Figure saved: {rel}")
    else:
        print("  (Skipped — matplotlib not available)")

    # ---- Generate results report --------------------------------------------
    _print_section("Results Report")
    report_content = _build_results_report(result)
    report_path = _REPO_ROOT / "docs" / "FEATURE_METHOD_FIT_EXPERIMENT_RESULTS.md"
    report_path.write_text(report_content, encoding="utf-8")
    try:
        rel = report_path.relative_to(_REPO_ROOT)
    except ValueError:
        rel = report_path
    print(f"  Report written: {rel}")

    # ---- Print output paths -------------------------------------------------
    _print_section("Output Files")
    for key, path in result["output_paths"].items():
        try:
            rel_path = Path(path).relative_to(_REPO_ROOT)
        except ValueError:
            rel_path = Path(path)
        print(f"  {key:<25}: {rel_path}")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
