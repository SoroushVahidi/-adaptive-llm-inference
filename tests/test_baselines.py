from src.baselines.best_of_n import BestOfNBaseline
from src.baselines.greedy import GreedyBaseline
from src.baselines.self_consistency import SelfConsistencyBaseline
from src.models.dummy import DummyModel


def _make_model(correct_prob: float = 1.0) -> DummyModel:
    model = DummyModel(correct_prob=correct_prob, seed=0)
    model.set_ground_truth("42")
    return model


def test_greedy_correct():
    model = _make_model(correct_prob=1.0)
    baseline = GreedyBaseline(model)
    result = baseline.solve("q1", "What is 6*7?", "42")
    assert result.correct
    assert result.samples_used == 1


def test_best_of_n():
    model = _make_model(correct_prob=0.6)
    baseline = BestOfNBaseline(model)
    result = baseline.solve("q1", "What is 6*7?", "42", n_samples=10)
    assert result.samples_used == 10
    assert len(result.candidates) == 10


def test_self_consistency():
    model = _make_model(correct_prob=0.6)
    baseline = SelfConsistencyBaseline(model)
    result = baseline.solve("q1", "What is 6*7?", "42", n_samples=10)
    assert result.samples_used == 10
    assert len(result.candidates) == 10


def test_baseline_names():
    model = _make_model()
    assert GreedyBaseline(model).name == "greedy"
    assert BestOfNBaseline(model).name == "best_of_n"
    assert SelfConsistencyBaseline(model).name == "self_consistency"
