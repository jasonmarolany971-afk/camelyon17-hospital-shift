import numpy as np
import torch

from src.evaluate import binary_metrics, threshold_at_specificity
from src.models import build_model
from src.utils import set_seed


def test_threshold_keeps_specificity_and_maximizes_sensitivity():
    negatives = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    positives = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    y = np.r_[np.zeros(10), np.ones(6)]
    scores = np.r_[negatives, positives]

    # 90% specificity on 10 negatives allows one false positive (0.95):
    # the lowest valid threshold is 0.9, which catches 0.9 and 0.99.
    threshold = threshold_at_specificity(y, scores, specificity=0.9)
    metrics = binary_metrics(y, scores, threshold)

    assert threshold == 0.9
    assert metrics["specificity_at_threshold"] == 0.9
    assert metrics["sensitivity_at_threshold"] == 2 / 6


def test_binary_metrics_on_perfect_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_metrics(y, scores, threshold=0.5)
    assert metrics == {
        "n": 4, "accuracy": 1.0, "auc": 1.0,
        "sensitivity_at_threshold": 1.0, "specificity_at_threshold": 1.0,
        "predicted_tumor_share": 0.5,
    }


def test_auc_is_undefined_with_a_single_class():
    metrics = binary_metrics(np.array([1, 1]), np.array([0.3, 0.7]), threshold=0.5)
    assert metrics["auc"] is None and metrics["specificity_at_threshold"] is None


def test_model_initialization_depends_only_on_the_seed():
    set_seed(0)
    first = build_model("densenet121").state_dict()
    set_seed(0)
    same = build_model("densenet121").state_dict()
    set_seed(1)
    other = build_model("densenet121").state_dict()
    assert all(torch.equal(first[k], same[k]) for k in first)
    assert not torch.equal(first["classifier.weight"], other["classifier.weight"])


def test_model_outputs_two_logits_for_96px_patches():
    model = build_model("densenet121").eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 96, 96)).shape == (2, 2)
