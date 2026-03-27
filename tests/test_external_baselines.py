"""Tests for external baseline wrapper infrastructure."""

import pytest

from src.baselines.external.best_route_wrapper import BESTRouteBaseline
from src.baselines.external.tale_wrapper import TALEBaseline


def test_tale_not_installed():
    baseline = TALEBaseline()
    assert baseline.name == "tale"
    assert not baseline.installed


def test_best_route_not_installed():
    baseline = BESTRouteBaseline()
    assert baseline.name == "best_route"
    assert not baseline.installed


def test_tale_solve_raises_without_install():
    baseline = TALEBaseline()
    with pytest.raises(RuntimeError, match="TALE is not installed"):
        baseline.solve("q1", "What is 2+2?", "4", 1)


def test_best_route_solve_raises_without_install():
    baseline = BESTRouteBaseline()
    with pytest.raises(RuntimeError, match="BEST-Route is not installed"):
        baseline.solve("q1", "What is 2+2?", "4", 1)
