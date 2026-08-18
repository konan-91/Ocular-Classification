"""The classifier: a pretrained ResNet with a new two class head.

The dataset holds tens of thousands of topoplots but only a few dozen
participants, so the effective sample size is much smaller than the image count
suggests. Fine tuning a pretrained backbone works better than training from
scratch at that scale.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from .channels import LABELS

ARCHITECTURES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}

DEFAULT_ARCHITECTURE = "resnet34"


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build(architecture: str = DEFAULT_ARCHITECTURE, pretrained: bool = True) -> nn.Module:
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {architecture!r}, "
            f"choose from {sorted(ARCHITECTURES)}"
        )

    factory, weights = ARCHITECTURES[architecture]
    model = factory(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze everything except the classification head."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


def save(model: nn.Module, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "metadata": metadata, "labels": list(LABELS)},
        path,
    )


def load(path: Path, device: torch.device | None = None) -> tuple[nn.Module, dict]:
    device = device or pick_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = checkpoint.get("metadata", {})

    model = build(
        metadata.get("architecture", DEFAULT_ARCHITECTURE), pretrained=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, metadata
