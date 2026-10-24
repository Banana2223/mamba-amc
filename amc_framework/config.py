"""Strict JSON configuration for baseline training experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    name: str = "tiny_probe"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    hdf5_path: str = "data/raw/RML2018.01a/GOLD_XYZ_OSC.0001_1024.hdf5"
    splits_path: str = "artifacts/P1-T7/split_indices.npz"
    classes_path: str = "data/raw/RML2018.01a/classes-fixed.json"
    sequence_length: int = 1024
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    train_augmentation: bool = False


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    name: str = "none"
    minimum_learning_rate: float = 0.0
    factor: float = 0.5
    patience: int = 3


@dataclass
class TrainingConfig:
    seed: int = 2026
    device: str = "auto"
    max_epochs: int = 1
    min_epochs: int = 1
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.0
    amp: bool = True
    deterministic: bool = True
    max_train_batches: int | None = None
    max_validation_batches: int | None = None
    checkpoint_metric: str = "macro_f1"
    checkpoint_mode: str = "max"
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.0


@dataclass
class EvaluationConfig:
    num_classes: int = 24
    low_snr_min_db: int = -20
    low_snr_max_db: int = 0


@dataclass
class OutputConfig:
    root: str = "outputs/P2-T2"
    run_name: str | None = None


@dataclass
class ExperimentConfig:
    experiment_name: str = "amc-baseline"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty")
        if self.data.sequence_length <= 0:
            raise ValueError("data.sequence_length must be positive")
        if self.data.batch_size <= 0:
            raise ValueError("data.batch_size must be positive")
        if self.data.num_workers < 0:
            raise ValueError("data.num_workers cannot be negative")
        if self.data.prefetch_factor <= 0:
            raise ValueError("data.prefetch_factor must be positive")
        if self.optimizer.learning_rate <= 0:
            raise ValueError("optimizer.learning_rate must be positive")
        if self.optimizer.weight_decay < 0:
            raise ValueError("optimizer.weight_decay cannot be negative")
        if self.optimizer.name.lower() not in {"adam", "adamw", "sgd"}:
            raise ValueError("optimizer.name must be adam, adamw, or sgd")
        if self.scheduler.name.lower() not in {"none", "cosine", "plateau"}:
            raise ValueError("scheduler.name must be none, cosine, or plateau")
        if self.training.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("training.device must be auto, cpu, or cuda")
        if self.training.max_epochs <= 0:
            raise ValueError("training.max_epochs must be positive")
        if self.training.min_epochs <= 0:
            raise ValueError("training.min_epochs must be positive")
        if self.training.min_epochs > self.training.max_epochs:
            raise ValueError("training.min_epochs cannot exceed max_epochs")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError(
                "training.gradient_accumulation_steps must be positive"
            )
        for name, value in (
            ("max_train_batches", self.training.max_train_batches),
            ("max_validation_batches", self.training.max_validation_batches),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"training.{name} must be positive or null")
        if self.training.checkpoint_metric not in {
            "loss",
            "accuracy",
            "macro_f1",
            "low_snr_accuracy",
        }:
            raise ValueError("unsupported training.checkpoint_metric")
        if self.training.checkpoint_mode not in {"min", "max"}:
            raise ValueError("training.checkpoint_mode must be min or max")
        if self.training.early_stopping_patience < 0:
            raise ValueError("training.early_stopping_patience cannot be negative")
        if self.evaluation.num_classes <= 1:
            raise ValueError("evaluation.num_classes must exceed one")
        if self.evaluation.low_snr_min_db > self.evaluation.low_snr_max_db:
            raise ValueError("low_snr_min_db cannot exceed low_snr_max_db")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS: dict[str, type[Any]] = {
    "model": ModelConfig,
    "data": DataConfig,
    "optimizer": OptimizerConfig,
    "scheduler": SchedulerConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
    "output": OutputConfig,
}


def _strict_dataclass(cls: type[Any], values: dict[str, Any], prefix: str) -> Any:
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown configuration key(s) in {prefix}: {names}")
    return cls(**values)


def experiment_config_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    allowed = {"experiment_name", *_SECTIONS}
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown top-level configuration key(s): {names}")

    kwargs: dict[str, Any] = {}
    if "experiment_name" in values:
        kwargs["experiment_name"] = values["experiment_name"]
    for section_name, cls in _SECTIONS.items():
        section_values = values.get(section_name, {})
        if not isinstance(section_values, dict):
            raise TypeError(f"Configuration section {section_name} must be an object")
        kwargs[section_name] = _strict_dataclass(
            cls, section_values, section_name
        )
    config = ExperimentConfig(**kwargs)
    config.validate()
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("The experiment configuration must be a JSON object")
    return experiment_config_from_dict(values)

