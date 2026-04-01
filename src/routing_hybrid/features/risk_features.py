from __future__ import annotations

from typing import Any


def add_risk_features(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for r in candidate_rows:
        parse_success = float(r.get("feat_fp_first_pass_parse_success", 1.0))
        hedging = float(r.get("feat_self_contains_hedging_language", 0.0))
        warning = float(r.get("feat_role_warning_score", 0.0))
        err = float(r.get("hfeat_prompt_error_score", r.get("feat_unified_error_score", 0.0)))
        r["hfeat_risk_parse_failure"] = 1.0 - parse_success
        r["hfeat_risk_hedging"] = hedging
        r["hfeat_risk_warning"] = warning
        r["hfeat_risk_composite"] = 0.4 * (1.0 - parse_success) + 0.3 * hedging + 0.3 * min(1.0, err + warning)
    return candidate_rows

