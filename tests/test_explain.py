import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.explain import GradCAM, most_confident_errors, overlap_table, shared_errors


def frame(probs, y):
    n = len(probs)
    return pd.DataFrame({"index": range(n), "y": y, "prob": probs, "hospital": 2, "slide": 20})


def test_most_confident_errors():
    f = frame([0.05, 0.4, 0.9, 0.6, 0.99, 0.2], [1, 1, 1, 0, 0, 0])
    assert list(most_confident_errors(f, "fn", 2)["index"]) == [0, 1]   # tumors called normal, lowest first
    assert list(most_confident_errors(f, "fp", 5)["index"]) == [4, 3]   # normals called tumor, highest first


def test_shared_errors_and_overlap():
    y = [1, 1, 0, 0]
    frames = {"a": frame([0.1, 0.9, 0.8, 0.1], y), "b": frame([0.3, 0.2, 0.7, 0.2], y)}
    shared = shared_errors(frames)
    assert list(shared["index"]) == [0, 2]       # both wrong; patch 0 (mean 0.2, 0.30 from 0.5) before 2 (mean 0.75)
    table = overlap_table({"test": frames})
    assert table.set_index("group").loc["all_methods_wrong", "errors"] == 2
    assert table.set_index("group").loc["b", "errors"] == 3


class CenterOnly(nn.Module):
    """Tumor logit computed from the central cell of a 3 x 3 feature map only."""

    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        nn.init.constant_(self.features.weight, 1.0)
        self.head = nn.Linear(2, 2)
        nn.init.constant_(self.head.weight, 1.0)

    def forward(self, x):
        return self.head(self.features(x)[:, :, 1, 1])


def test_grad_cam_points_at_the_cell_that_drives_the_score():
    model = CenterOnly()
    x = torch.rand(2, 3, 3, 3) + 0.1                 # positive everywhere
    maps, probs = GradCAM(model, model.features)(x)
    assert maps.shape == (2, 3, 3) and probs.shape == (2,)
    # Weights are spatial means of the gradient (nonzero only at the center), activations are positive
    # everywhere: the map is positive everywhere, and the explained score depends on the center alone.
    assert torch.all(maps > 0)
    x_center_only = torch.zeros(2, 3, 3, 3)
    x_center_only[:, :, 1, 1] = 1.0
    maps, _ = GradCAM(model, model.features)(x_center_only)
    assert torch.all(maps[:, 1, 1] > 0) and maps.sum() == maps[:, 1, 1].sum()
