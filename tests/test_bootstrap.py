import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.bootstrap import cluster_arrays, draw_weights, weighted_metrics


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    n, n_clusters = 3000, 6
    cluster = rng.integers(0, n_clusters, n)
    y = (rng.random(n) < 0.2 + 0.1 * cluster).astype(int)            # tumor share differs by patient
    prob = np.round(np.clip(0.3 * y + 0.7 * rng.random(n), 0, 1), 2)  # rounding creates many ties
    return prob.astype(np.float32), y, cluster, n_clusters


def test_unit_weights_give_the_usual_metrics(sample):
    prob, y, cluster, n_clusters = sample
    metrics = weighted_metrics(cluster_arrays(prob, y, cluster, n_clusters), np.ones(n_clusters))
    assert metrics["auc"][0] == pytest.approx(roc_auc_score(y, prob))
    assert metrics["accuracy"][0] == pytest.approx(np.mean((prob > 0.5) == y))


def test_drawing_a_patient_twice_equals_duplicating_its_patches(sample):
    prob, y, cluster, n_clusters = sample
    weights = np.array([2, 0, 1, 1, 0, 2], dtype=float)
    repeat = weights[cluster].astype(int)
    expected_auc = roc_auc_score(np.repeat(y, repeat), np.repeat(prob, repeat))
    metrics = weighted_metrics(cluster_arrays(prob, y, cluster, n_clusters), weights)
    assert metrics["auc"][0] == pytest.approx(expected_auc)
    assert metrics["accuracy"][0] == pytest.approx(np.mean(np.repeat((prob > 0.5) == y, repeat)))


def test_draws_are_reproducible_and_keep_the_number_of_patients():
    weights = draw_weights(9, 50, seed=3)
    assert np.array_equal(weights, draw_weights(9, 50, seed=3))
    assert weights.shape == (50, 9) and np.all(weights.sum(axis=1) == 9)
