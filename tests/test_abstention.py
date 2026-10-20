import numpy as np
import pytest

from src.abstention import apply_cutoff, aurc, cutoff_for_coverage, cutoff_for_risk, risk_at, risk_curve


def test_risk_curve_by_hand():
    prob = np.array([0.9, 0.2, 0.6, 0.95])
    y = np.array([1, 1, 0, 1])
    # confidence 0.40, 0.30, 0.10, 0.45 -> order 0.95 (right), 0.9 (right), 0.2 (wrong), 0.6 (wrong)
    assert np.allclose(risk_curve(prob, y), [0, 0, 1 / 3, 2 / 4])
    assert np.allclose(risk_at(prob, y, [0.5, 1.0]), [0, 0.5])


def test_e_aurc_is_zero_when_errors_are_the_least_confident():
    prob = np.array([0.99, 0.02, 0.9, 0.45, 0.55])
    y = np.array([1, 0, 1, 1, 0])           # the two errors have confidence 0.05
    value, excess = aurc(prob, y)
    assert excess == pytest.approx(0.0) and value > 0


@pytest.fixture
def val():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    prob = np.clip(0.5 + (2 * y - 1) * rng.normal(0.2, 0.15, 5000), 0, 1)
    return prob, y


def test_coverage_cutoff_answers_the_target_share(val):
    prob, y = val
    coverage, _ = apply_cutoff(prob, y, cutoff_for_coverage(prob, 0.8))
    assert coverage == pytest.approx(0.8, abs=0.002)


def test_risk_cutoff_meets_the_target_with_the_largest_coverage(val):
    prob, y = val
    cutoff = cutoff_for_risk(prob, y, 0.02)
    coverage, risk = apply_cutoff(prob, y, cutoff)
    assert risk <= 0.02
    assert risk_at(prob, y, [coverage + 0.01])[0] > 0.02   # answering 1% more would exceed the target


def test_answering_nothing_gives_zero_coverage_and_undefined_risk():
    prob, y = np.array([0.9, 0.1]), np.array([0, 1])   # every patch confidently wrong
    cutoff = cutoff_for_risk(prob, y, 0.01)
    coverage, risk = apply_cutoff(prob, y, cutoff)
    assert cutoff == np.inf and coverage == 0 and np.isnan(risk)
