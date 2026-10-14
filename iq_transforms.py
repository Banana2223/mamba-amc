"""Configurable IQ preprocessing and channel augmentation for AMC experiments.

All operations work on a real tensor shaped ``(2, length)`` where channel 0 is
I and channel 1 is Q. Random operations use PyTorch's worker-local RNG, so a
seeded DataLoader remains reproducible with multiple workers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch


NormalizationMode = Literal["none", "rms"]
CropMode = Literal["center", "random"]


@dataclass(frozen=True)
class IQTransformConfig:
    """Configuration for preprocessing and label-preserving channel effects."""

    sequence_length: int = 1024
    crop_mode: CropMode = "center"
    normalization: NormalizationMode = "rms"
    epsilon: float = 1e-8

    awgn_probability: float = 0.0
    awgn_snr_db_min: float = 15.0
    awgn_snr_db_max: float = 30.0

    phase_probability: float = 0.0
    phase_max_radians: float = math.pi

    frequency_probability: float = 0.0
    frequency_max_abs_cycles_per_sample: float = 0.005

    multipath_probability: float = 0.0
    multipath_max_echoes: int = 2
    multipath_max_delay: int = 16
    multipath_max_gain: float = 0.35
    multipath_decay: float = 0.6

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.crop_mode not in ("center", "random"):
            raise ValueError(f"Unsupported crop mode: {self.crop_mode}")
        if self.normalization not in ("none", "rms"):
            raise ValueError(f"Unsupported normalization: {self.normalization}")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

        probabilities = {
            "awgn_probability": self.awgn_probability,
            "phase_probability": self.phase_probability,
            "frequency_probability": self.frequency_probability,
            "multipath_probability": self.multipath_probability,
        }
        for name, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        if self.awgn_snr_db_min > self.awgn_snr_db_max:
            raise ValueError("awgn_snr_db_min cannot exceed awgn_snr_db_max")
        if not 0.0 <= self.phase_max_radians <= math.pi:
            raise ValueError("phase_max_radians must be in [0, pi]")
        if not 0.0 <= self.frequency_max_abs_cycles_per_sample <= 0.5:
            raise ValueError(
                "frequency_max_abs_cycles_per_sample must be in [0, 0.5]"
            )
        if self.multipath_max_echoes < 1:
            raise ValueError("multipath_max_echoes must be at least 1")
        if self.multipath_max_delay < 1:
            raise ValueError("multipath_max_delay must be at least 1")
        if self.multipath_max_gain < 0:
            raise ValueError("multipath_max_gain cannot be negative")
        if not 0 < self.multipath_decay <= 1:
            raise ValueError("multipath_decay must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IQTransform:
    """Apply crop, optional channel effects, and final normalization."""

    def __init__(self, config: IQTransformConfig) -> None:
        self.config = config

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        if low == high:
            return float(low)
        return float(torch.empty(()).uniform_(low, high).item())

    @staticmethod
    def _should_apply(probability: float) -> bool:
        return probability >= 1.0 or (
            probability > 0.0 and float(torch.rand(()).item()) < probability
        )

    def _crop(self, iq: torch.Tensor) -> torch.Tensor:
        source_length = int(iq.shape[-1])
        target_length = self.config.sequence_length
        if target_length > source_length:
            raise ValueError(
                f"sequence_length={target_length} exceeds input length={source_length}"
            )
        if target_length == source_length:
            return iq
        if self.config.crop_mode == "center":
            start = (source_length - target_length) // 2
        else:
            start = int(
                torch.randint(0, source_length - target_length + 1, ()).item()
            )
        return iq[:, start : start + target_length]

    def _phase_rotate(self, signal: torch.Tensor) -> torch.Tensor:
        phase = self._uniform(
            -self.config.phase_max_radians,
            self.config.phase_max_radians,
        )
        return signal * torch.polar(
            torch.ones((), dtype=signal.real.dtype),
            torch.tensor(phase, dtype=signal.real.dtype),
        )

    def _frequency_shift(self, signal: torch.Tensor) -> torch.Tensor:
        max_offset = self.config.frequency_max_abs_cycles_per_sample
        offset = self._uniform(-max_offset, max_offset)
        samples = torch.arange(len(signal), dtype=signal.real.dtype)
        phase = 2.0 * math.pi * offset * samples
        return signal * torch.polar(torch.ones_like(phase), phase)

    def _multipath(self, signal: torch.Tensor) -> torch.Tensor:
        output = signal.clone()
        echo_count = int(
            torch.randint(1, self.config.multipath_max_echoes + 1, ()).item()
        )
        max_delay = min(self.config.multipath_max_delay, len(signal) - 1)
        if max_delay < 1:
            return output

        for echo_number in range(echo_count):
            delay = int(torch.randint(1, max_delay + 1, ()).item())
            max_gain = self.config.multipath_max_gain * (
                self.config.multipath_decay**echo_number
            )
            gain = self._uniform(0.0, max_gain)
            phase = self._uniform(-math.pi, math.pi)
            coefficient = torch.polar(
                torch.tensor(gain, dtype=signal.real.dtype),
                torch.tensor(phase, dtype=signal.real.dtype),
            )
            output[delay:] += coefficient * signal[:-delay]
        return output

    def _add_awgn(self, signal: torch.Tensor) -> torch.Tensor:
        snr_db = self._uniform(
            self.config.awgn_snr_db_min,
            self.config.awgn_snr_db_max,
        )
        signal_power = torch.mean(torch.abs(signal).square())
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        component_std = torch.sqrt(noise_power / 2.0)
        noise = component_std * torch.complex(
            torch.randn_like(signal.real), torch.randn_like(signal.real)
        )
        return signal + noise

    def _normalize(self, iq: torch.Tensor) -> torch.Tensor:
        if self.config.normalization == "none":
            return iq
        # One scale for the complex signal preserves I/Q geometry and does not
        # remove the carrier/DC component that may distinguish AM subclasses.
        complex_power = torch.mean(iq[0].square() + iq[1].square())
        scale = torch.sqrt(complex_power.clamp_min(self.config.epsilon))
        return iq / scale

    def __call__(self, iq: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(iq, dtype=torch.float32).clone()
        if tensor.ndim != 2 or tensor.shape[0] != 2:
            raise ValueError(f"Expected IQ shape (2, length), got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("IQ input contains NaN or Inf")

        tensor = self._crop(tensor)
        signal = torch.complex(tensor[0], tensor[1])
        if self._should_apply(self.config.phase_probability):
            signal = self._phase_rotate(signal)
        if self._should_apply(self.config.frequency_probability):
            signal = self._frequency_shift(signal)
        if self._should_apply(self.config.multipath_probability):
            signal = self._multipath(signal)
        if self._should_apply(self.config.awgn_probability):
            signal = self._add_awgn(signal)

        output = torch.stack((signal.real, signal.imag), dim=0)
        output = self._normalize(output).contiguous()
        if not torch.isfinite(output).all():
            raise RuntimeError("IQ transform produced NaN or Inf")
        return output


def build_train_transform(
    sequence_length: int = 1024,
    enable_augmentation: bool = False,
) -> IQTransform:
    """Build the conservative training transform used by default.

    Augmentation is opt-in so the first baselines can establish an unaugmented
    reference. When enabled, all parameter ranges remain deliberately mild.
    """
    return IQTransform(
        IQTransformConfig(
            sequence_length=sequence_length,
            crop_mode="random" if sequence_length < 1024 else "center",
            normalization="rms",
            awgn_probability=0.25 if enable_augmentation else 0.0,
            awgn_snr_db_min=15.0,
            awgn_snr_db_max=30.0,
            phase_probability=0.50 if enable_augmentation else 0.0,
            phase_max_radians=math.pi,
            frequency_probability=0.25 if enable_augmentation else 0.0,
            frequency_max_abs_cycles_per_sample=0.005,
            multipath_probability=0.20 if enable_augmentation else 0.0,
            multipath_max_echoes=2,
            multipath_max_delay=16,
            multipath_max_gain=0.35,
            multipath_decay=0.6,
        )
    )


def build_eval_transform(sequence_length: int = 1024) -> IQTransform:
    """Build deterministic validation/test preprocessing (never augmented)."""
    return IQTransform(
        IQTransformConfig(
            sequence_length=sequence_length,
            crop_mode="center",
            normalization="rms",
        )
    )

