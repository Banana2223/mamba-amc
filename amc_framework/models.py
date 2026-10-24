"""Small model registry shared by all baseline experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


ModelFactory = Callable[..., nn.Module]
_MODEL_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Model name cannot be empty")

    def decorator(factory: ModelFactory) -> ModelFactory:
        if normalized in _MODEL_REGISTRY:
            raise KeyError(f"Model is already registered: {normalized}")
        _MODEL_REGISTRY[normalized] = factory
        return factory

    return decorator


def create_model(name: str, **kwargs: Any) -> nn.Module:
    normalized = name.strip().lower()
    if normalized not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise KeyError(f"Unknown model {name!r}; available models: {available}")
    return _MODEL_REGISTRY[normalized](**kwargs)


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_REGISTRY))


@register_model("tiny_probe")
class TinyAMCProbe(nn.Module):
    """Engineering-only CNN used to verify the reusable framework."""

    def __init__(self, num_classes: int = 24) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected input [B, 2, L], got {tuple(iq.shape)}")
        return self.classifier(self.features(iq).squeeze(-1))


class ConvBlock1d(nn.Sequential):
    """Convolution, normalization, activation, and temporal downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


@register_model("lightweight_cnn")
class LightweightCNN(nn.Module):
    """Compact 1-D CNN baseline for raw I/Q modulation recognition.

    The model treats I and Q as two input channels. Four convolutional stages
    reduce the 1024-sample sequence to local feature maps. Adaptive pooling to
    more than one temporal bin deliberately preserves coarse temporal layout;
    this makes the network a stronger baseline than a global-average-only
    engineering probe while keeping its parameter count modest.
    """

    def __init__(
        self,
        num_classes: int = 24,
        base_channels: int = 48,
        pooled_length: int = 8,
        classifier_width: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must exceed one")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if pooled_length <= 0:
            raise ValueError("pooled_length must be positive")
        if classifier_width <= 0:
            raise ValueError("classifier_width must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        stage_channels = (
            base_channels,
            base_channels,
            base_channels * 2,
            base_channels * 4,
        )
        kernels = (7, 5, 5, 3)
        blocks: list[nn.Module] = []
        in_channels = 2
        for out_channels, kernel_size in zip(stage_channels, kernels, strict=True):
            blocks.append(
                ConvBlock1d(in_channels, out_channels, kernel_size=kernel_size)
            )
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.temporal_pool = nn.AdaptiveAvgPool1d(pooled_length)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(stage_channels[-1] * pooled_length, classifier_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_width, num_classes),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected input [B, 2, L], got {tuple(iq.shape)}")
        if iq.shape[-1] < 16:
            raise ValueError("Input sequence must contain at least 16 samples")
        features = self.features(iq)
        return self.classifier(self.temporal_pool(features))

