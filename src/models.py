"""Model construction utilities."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

NUM_CLASSES = 10


def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build a supported ImageNet-pretrained model adapted to CIFAR-10."""
    normalized = name.lower().replace("_", "-")
    if normalized == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if normalized in {"mobilenetv3-small", "mobilenet-v3-small"}:
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {name}")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Estimate model parameter size in MiB using parameter dtypes."""
    total_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    total_bytes += sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    return total_bytes / (1024 ** 2)


def model_summary(model: nn.Module) -> Tuple[int, float]:
    """Return trainable parameter count and estimated in-memory size in MiB."""
    return count_parameters(model), model_size_mb(model)
