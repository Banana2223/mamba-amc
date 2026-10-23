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

