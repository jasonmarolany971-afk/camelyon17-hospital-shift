"""The in-browser demo computes Grad-CAM in closed form: it must equal PyTorch autograd, and ONNX must equal PyTorch."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.explain import GradCAM
from src.export_web_demo import DecisionWithCam
from src.models import build_model


class DecisionOnly(torch.nn.Module):
    """Same network returning both logits, so that GradCAM can explain z = logit(tumor) - logit(normal)."""

    def __init__(self, model):
        super().__init__()
        self.model, self.features = model, model.features

    def forward(self, x):
        logits = self.model(x)
        return torch.stack([torch.zeros_like(logits[:, 0]), logits[:, 1] - logits[:, 0]], dim=1)


def test_closed_form_grad_cam_equals_autograd():
    torch.manual_seed(0)
    model = build_model("densenet121").eval()
    x = torch.randn(2, 3, 96, 96)
    z, cam = DecisionWithCam(model)(x)
    reference = DecisionOnly(model)
    maps, _ = GradCAM(reference, reference.features)(x)
    logits = model(x)
    assert torch.allclose(z, logits[:, 1] - logits[:, 0], atol=1e-5)
    assert torch.allclose(cam, maps, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not Path("web/model.onnx").exists(), reason="run python -m src.export_web_demo first")
def test_onnx_model_runs_and_returns_a_3x3_map():
    import onnxruntime

    session = onnxruntime.InferenceSession("web/model.onnx")
    z, cam = session.run(None, {"image": np.random.default_rng(0).normal(size=(1, 3, 96, 96)).astype(np.float32)})
    assert z.shape == (1,) and cam.shape == (1, 3, 3) and (cam >= 0).all()
