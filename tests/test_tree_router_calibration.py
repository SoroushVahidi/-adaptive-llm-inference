from __future__ import annotations

import numpy as np

from src.routing_hybrid.calibration import ProbabilityCalibrator


def test_calibration_hooks() -> None:
    s = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    y = np.array([0, 0, 1, 1], dtype=int)
    for method in ["none", "sigmoid", "isotonic"]:
        c = ProbabilityCalibrator(method=method).fit(s, y)
        out = c.transform(s)
        assert len(out) == len(s)
