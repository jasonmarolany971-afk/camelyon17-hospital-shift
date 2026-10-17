"""Model definitions."""

import torch.nn as nn
import torchvision


def build_model(name: str, num_classes: int = 2) -> nn.Module:
    """Torchvision CNN trained from scratch, with a new final layer (as in WILDS)."""
    if name != "densenet121":
        raise ValueError(f"Unsupported model {name!r}")
    model = torchvision.models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
