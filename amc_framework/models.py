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


class RadioResidualUnit(nn.Module):
    """Two-convolution residual unit used by the RadioML ResNet baseline."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.activation(self.conv1(features)))
        return features + residual


class RadioResidualStack(nn.Module):
    """Projection, residual units, and x2 temporal downsampling."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        units: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if units <= 0:
            raise ValueError("units must be positive")
        self.projection = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.residual_units = nn.Sequential(
            *[
                RadioResidualUnit(channels, kernel_size=kernel_size)
                for _ in range(units)
            ]
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.projection(features)
        return self.pool(self.residual_units(projected))


@register_model("radio_resnet")
class RadioResNet(nn.Module):
    """Six-stack 1-D ResNet adapted from the RadioML 2018 baseline.

    The reference architecture maps ``(2, 1024)`` raw I/Q samples through six
    residual stacks, producing ``(32, 16)`` features before two 128-unit SELU
    layers and a 24-way classifier. This implementation keeps that topology
    while exposing its dimensions for controlled project experiments.
    """

    def __init__(
        self,
        num_classes: int = 24,
        input_length: int = 1024,
        channels: int = 32,
        num_stacks: int = 6,
        units_per_stack: int = 2,
        kernel_size: int = 3,
        classifier_width: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must exceed one")
        if input_length <= 0:
            raise ValueError("input_length must be positive")
        if num_stacks <= 0:
            raise ValueError("num_stacks must be positive")
        downsampling_factor = 2**num_stacks
        if input_length % downsampling_factor != 0:
            raise ValueError(
                "input_length must be divisible by 2 ** num_stacks"
            )
        if classifier_width <= 0:
            raise ValueError("classifier_width must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        stacks: list[nn.Module] = []
        in_channels = 2
        for _ in range(num_stacks):
            stacks.append(
                RadioResidualStack(
                    in_channels=in_channels,
                    channels=channels,
                    units=units_per_stack,
                    kernel_size=kernel_size,
                )
            )
            in_channels = channels
        self.input_length = input_length
        self.features = nn.Sequential(*stacks)
        flattened_features = channels * (input_length // downsampling_factor)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, classifier_width),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(classifier_width, classifier_width),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(classifier_width, num_classes),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                fan_in = module.in_channels * module.kernel_size[0]
                nn.init.normal_(module.weight, mean=0.0, std=fan_in**-0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=module.in_features**-0.5,
                )
                nn.init.zeros_(module.bias)
        # Start every residual branch at zero so each unit initially behaves
        # like an identity map. This prevents activation growth through the 12
        # additions while allowing the residual paths to learn immediately.
        for module in self.modules():
            if isinstance(module, RadioResidualUnit):
                nn.init.zeros_(module.conv2.weight)
                if module.conv2.bias is not None:
                    nn.init.zeros_(module.conv2.bias)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError(f"Expected input [B, 2, L], got {tuple(iq.shape)}")
        if iq.shape[-1] != self.input_length:
            raise ValueError(
                f"Expected sequence length {self.input_length}, got {iq.shape[-1]}"
            )
        return self.classifier(self.features(iq))

