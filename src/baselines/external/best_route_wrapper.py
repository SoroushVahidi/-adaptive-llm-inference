"""BEST-Route baseline wrappers.

This module provides two classes:

BESTRouteBaseline
    A thin adapter that delegates to the *official* BEST-Route code when the
    official repository has been cloned into ``external/best_route/.repo``.
    At present this wrapper raises ``RuntimeError`` because the official repo
    is not bundled (it requires multi-model response pre-generation and a
    trained DeBERTa reward-model/router — see below for setup instructions).

BESTRouteAdaptedBaseline
    A *documented compatibility adaptation* of the BEST-Route algorithm for
    this repository's binary cheap-vs-revise routing setting.  It is
    immediately runnable with any ``Model`` backend without external
    dependencies, and every design decision is traced to the official
    code/paper below.

---

Official Paper
--------------
BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute
Ding, Mallick, Zhang, Wang, Madrigal, Garcia, Xia, Lakshmanan, Wu, Rühle
Forty-second International Conference on Machine Learning (ICML 2025)
arXiv: https://arxiv.org/abs/2506.22716

Official Code
-------------
https://github.com/microsoft/best-route-llm   (MIT license)

Official Algorithm (summary)
-----------------------------
1. Action space: candidate (LLM-model, best-of-n) pairs, e.g.
   {llama-31-8b_ourRM_bo1, …, llama-31-8b_ourRM_bo5, gpt-4o_ourRM_bo1}.
2. Router: DeBERTa-v3-small pairwise ranker trained with ``prob_nlabels``
   loss on query text + candidate-response pairs; quality signal is armoRM /
   proxy-RM score of each sampled response.
3. Inference mode: "bubble" — evaluate from cheapest action upward, escalate
   when the router predicts a sufficiently higher expected reward.
4. Budget constraint: configurable via ``--match_t`` and ``--candidate_model``
   flags in ``train_router.py``.

---

Compatibility Adaptation (BESTRouteAdaptedBaseline) — Deviation Notes
-----------------------------------------------------------------------
This repo studies single-model, binary cheap-vs-revise routing.  The full
official BEST-Route pipeline requires:

  (a) Multiple LLM backends (e.g., LLaMA-31-8B and GPT-4o).
  (b) 20 pre-sampled responses per query per model.
  (c) armoRM oracle scoring + proxy reward model training.
  (d) DeBERTa-v3-small router fine-tuning.

None of these are feasible without new API calls and external model
downloads.  ``BESTRouteAdaptedBaseline`` maps the BEST-Route *framework* to
this repo's setting with the following documented deviations:

  Faithful elements:
  - Action space: binary {reasoning_greedy (cost 1), direct_plus_revise (cost 2)},
    analogous to {cheap_model_bo1, expensive_model_bo1} in the official code.
  - "Bubble" inference: apply cheap action first; escalate only when the
    routing score is sufficiently high (matches official ``inference_mode=bubble``).
  - Budget awareness: ``n_samples`` arg controls available compute per query
    (matches official budget-constrained routing objective).

  Deviations:
  DEV-1  ACTION SPACE: Binary instead of ≥6 actions.  Official selects from
         (model × best-of-n) Cartesian product; this adaptation uses exactly
         two strategies from this repo's action catalogue.
  DEV-2  ROUTER: Feature-based heuristic instead of trained DeBERTa ranker.
         Official trains ``microsoft/deberta-v3-small`` on 8 000 labelled
         examples; training data would require multi-model response sampling
         and reward-model scoring — both infeasible without new API calls.
         This adaptation uses ``difficulty_proxy + (1 − confidence_proxy)``
         derived from ``src/evaluation/strong_baselines_eval._difficulty_score``
         and ``._confidence_from_first_reasoning``, which are string/regex
         features computed offline.
  DEV-3  QUALITY SIGNAL: Binary correctness instead of continuous armoRM
         scores.  In offline evaluation the adapted baseline computes
         ``correct`` = predicted == ground_truth; no reward model is used.
  DEV-4  SINGLE MODEL: One backend instead of multiple LLMs.
         All inference uses the ``Model`` passed at construction time.

See ``docs/BEST_ROUTE_INTEGRATION.md`` for the full integration document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.baselines.base import Baseline, BaselineResult
from src.baselines.external.base import ExternalBaseline
from src.models.base import Model

_REPO_DIR = Path(__file__).resolve().parents[3] / "external" / "best_route" / ".repo"


# ---------------------------------------------------------------------------
# Official-code wrapper (blocked until external repo is cloned)
# ---------------------------------------------------------------------------


class BESTRouteBaseline(ExternalBaseline):
    """Adapter for the *official* BEST-Route code.

    Status: **blocked** — requires cloning
    ``https://github.com/microsoft/best-route-llm`` into
    ``external/best_route/.repo`` and completing the multi-stage data
    preparation pipeline described in ``external/best_route/README.md``.

    For an immediately-runnable compatibility adaptation, use
    ``BESTRouteAdaptedBaseline`` instead.
    """

    def __init__(self, model: Model | None = None) -> None:
        super().__init__(model)

    @property
    def name(self) -> str:
        return "best_route"

    def _check_installation(self) -> bool:
        return _REPO_DIR.exists()

    def solve(
        self, query_id: str, question: str, ground_truth: str, n_samples: int
    ) -> BaselineResult:
        if not self.installed:
            raise RuntimeError(
                f"BEST-Route official code is not installed. "
                f"Clone https://github.com/microsoft/best-route-llm into {_REPO_DIR} "
                "and follow the setup instructions in external/best_route/README.md."
            )
        raise NotImplementedError(
            "Official BEST-Route bridge is not yet implemented. "
            "The multi-stage pipeline (response sampling → armoRM scoring → "
            "proxy-RM training → router training) must be completed before "
            "this wrapper can delegate to the official code. "
            "See docs/BEST_ROUTE_INTEGRATION.md for the integration roadmap."
        )


# ---------------------------------------------------------------------------
# Compatibility adaptation (immediately runnable)
# ---------------------------------------------------------------------------


class BESTRouteAdaptedBaseline(Baseline):
    """Documented compatibility adaptation of BEST-Route for binary routing.

    Implements the BEST-Route framework (Ding et al., ICML 2025) mapped to
    this repository's binary cheap-vs-revise routing setting.

    All design decisions are traced in the module docstring above.
    See ``docs/BEST_ROUTE_INTEGRATION.md`` for the full integration document.

    Parameters
    ----------
    model:
        The inference backend.  Used for both the cheap first pass and the
        optional costly revision.
    threshold:
        Routing score threshold in [0, 2].  Queries whose routing score
        (difficulty_proxy + 1 − confidence_proxy) meets or exceeds this
        value are escalated to ``direct_plus_revise``.
        Default 0.5 is conservative; tune on a held-out validation set.
        In the official code this is implicitly set by the trained router.
    """

    #: Routing score threshold default.  Queries above this are escalated.
    DEFAULT_THRESHOLD: float = 0.5

    def __init__(
        self,
        model: Model,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        super().__init__(model)
        if not (0.0 <= threshold <= 2.0):
            raise ValueError(
                f"threshold must be in [0, 2]; got {threshold}. "
                "The routing score (difficulty + 1-conf) lies in this range."
            )
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "best_route_adapted"

    # ------------------------------------------------------------------
    # Feature-based routing score (DEV-2 deviation from official DeBERTa)
    # ------------------------------------------------------------------

    @staticmethod
    def _routing_score(question: str, first_pass_raw: str, parsed: str) -> float:
        """Compute a routing score in [0, 2].

        Higher score means the query is more likely to benefit from revision.

        Derivation (DEV-2):
        Official BEST-Route uses a trained DeBERTa-v3-small pairwise ranker.
        This adaptation uses:
            score = difficulty_proxy(question)
                  + (1 − confidence_proxy(question, first_pass_raw, parsed))
        where both components are computed by the string/regex feature
        functions already used in this repo's strong-baselines evaluation.
        The score range is [0, 2] because each component is in [0, 1].
        """
        from src.evaluation.strong_baselines_eval import (
            _confidence_from_first_reasoning,
            _difficulty_score,
        )

        diff = _difficulty_score(question)
        conf = _confidence_from_first_reasoning(question, first_pass_raw, parsed)
        return diff + (1.0 - conf)

    @staticmethod
    def _revision_prompt(question: str, first_pass: str) -> str:
        """Build the revision prompt.

        Mirrors the ``direct_plus_revise`` strategy already in this repo:
        present the first-pass output and ask the model to review and correct.
        """
        return (
            f"{question}\n\n"
            "Here is an initial attempt at solving this problem:\n\n"
            f"{first_pass}\n\n"
            "Please carefully review the solution above. "
            "Identify and correct any errors, then provide your final answer."
        )

    # ------------------------------------------------------------------
    # Baseline interface
    # ------------------------------------------------------------------

    def solve(
        self,
        query_id: str,
        question: str,
        ground_truth: str,
        n_samples: int = 2,
    ) -> BaselineResult:
        """Run the adapted BEST-Route baseline on a single query.

        Implements the *bubble* inference mode from the official code:
        apply the cheapest action first, then escalate if the routing
        score indicates likely quality improvement and budget permits.

        Parameters
        ----------
        n_samples:
            Per-query compute budget.  If ``n_samples < 2`` there is no
            budget to escalate; the cheap action is always used.  This
            mirrors BEST-Route's budget-constrained routing objective.
        """
        from src.utils.answer_extraction import extract_numeric_answer

        metadata: dict[str, Any] = {
            "adaptation": "binary_best_route",
            "deviation_notes": "DEV-1 DEV-2 DEV-3 DEV-4 (see module docstring)",
            "bubble_mode": True,
            "threshold": self.threshold,
        }

        # --- Bubble step 1: apply cheapest action (reasoning_greedy) ---
        raw_first = self.model.generate(question)
        parsed_first = extract_numeric_answer(raw_first)

        if n_samples < 2:
            # Budget exhausted: stay with cheap action (no escalation possible).
            # This matches BEST-Route's behaviour when the budget allows only
            # one forward pass.
            return BaselineResult(
                query_id=query_id,
                question=question,
                candidates=[raw_first],
                final_answer=parsed_first,
                ground_truth=ground_truth,
                correct=(parsed_first == ground_truth),
                samples_used=1,
                metadata={
                    **metadata,
                    "action": "reasoning_greedy",
                    "routing_score": None,
                    "budget_exceeded": True,
                },
            )

        # --- Bubble step 2: compute routing score and decide whether to escalate ---
        score = self._routing_score(question, raw_first, parsed_first)
        metadata["routing_score"] = round(score, 4)

        if score >= self.threshold:
            # Escalate to the costly action (direct_plus_revise).
            # In official BEST-Route this corresponds to routing to a higher
            # best-of-n tier or a more capable model.
            prompt2 = self._revision_prompt(question, raw_first)
            raw_revised = self.model.generate(prompt2)
            parsed_final = extract_numeric_answer(raw_revised)
            return BaselineResult(
                query_id=query_id,
                question=question,
                candidates=[raw_first, raw_revised],
                final_answer=parsed_final,
                ground_truth=ground_truth,
                correct=(parsed_final == ground_truth),
                samples_used=2,
                metadata={**metadata, "action": "direct_plus_revise"},
            )
        else:
            # Stay with cheap action: routing score below threshold.
            return BaselineResult(
                query_id=query_id,
                question=question,
                candidates=[raw_first],
                final_answer=parsed_first,
                ground_truth=ground_truth,
                correct=(parsed_first == ground_truth),
                samples_used=1,
                metadata={**metadata, "action": "reasoning_greedy"},
            )
