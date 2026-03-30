"""Repository-grounded default artifact locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    """Paths under a repository root."""

    root: Path
    outputs: Path

    @classmethod
    def from_root(cls, root: Path) -> ArtifactPaths:
        root = root.resolve()
        return cls(root=root, outputs=root / "outputs")

    @property
    def simulated_sweep(self) -> Path:
        return self.outputs / "simulated_sweep"

    @property
    def baselines(self) -> Path:
        return self.outputs / "baselines"

    @property
    def cross_regime(self) -> Path:
        return self.outputs / "cross_regime_comparison"

    @property
    def oracle_routing_eval(self) -> Path:
        return self.outputs / "oracle_routing_eval"

    @property
    def oracle_subset_eval(self) -> Path:
        return self.outputs / "oracle_subset_eval"

    @property
    def next_stage_eval(self) -> Path:
        return self.outputs / "next_stage_eval"

    @property
    def budget_sweep(self) -> Path:
        return self.outputs / "budget_sweep"

    @property
    def paper_tables(self) -> Path:
        return self.outputs / "paper_tables"

    @property
    def paper_figures(self) -> Path:
        return self.outputs / "paper_figures"


# Policy eval summaries used by cross-regime docs (accuracy/cost tradeoff table).
REAL_POLICY_SUMMARY_FILES: tuple[str, ...] = (
    "outputs/real_policy_eval/summary.json",
    "outputs/real_math500_policy_eval/summary.json",
    "outputs/real_hard_gsm8k_policy_eval/summary.json",
    "outputs/real_hard_gsm8k_b2_policy_eval/summary.json",
)
