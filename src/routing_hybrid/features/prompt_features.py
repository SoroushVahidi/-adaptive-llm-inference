from __future__ import annotations

import re
from typing import Any


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def add_prompt_features(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for r in candidate_rows:
        q = str(r.get("question", ""))
        toks = q.split()
        r["hfeat_prompt_length_chars"] = float(len(q))
        r["hfeat_prompt_length_tokens"] = float(len(toks))
        r["hfeat_prompt_num_count"] = float(len(_NUM_RE.findall(q)))
        r["hfeat_prompt_has_question_mark"] = 1.0 if "?" in q else 0.0
        r["hfeat_prompt_has_percent"] = 1.0 if "%" in q else 0.0
        # Reuse existing engineered features when present.
        r["hfeat_prompt_unified_confidence"] = float(r.get("feat_unified_confidence_score", 0.0))
        r["hfeat_prompt_error_score"] = float(r.get("feat_unified_error_score", 0.0))
    return candidate_rows

