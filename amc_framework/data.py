"""DataLoader construction for the fixed RadioML 2018.01A protocol."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from iq_transforms import build_eval_transform, build_train_transform
from rml2018_dataset import RadioML2018Dataset

from .config import DataConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_class_names(path: str | Path, expected_count: int) -> list[str]:
    resolved = resolve_project_path(path)
    values = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"Class file must contain a JSON string list: {resolved}")
    if len(values) != expected_count or len(set(values)) != expected_count:
        raise ValueError(
            f"Expected {expected_count} unique class names, got {len(values)}"
        )
    return values


@dataclass
class DataBundle:
    train_dataset: RadioML2018Dataset
    validation_dataset: RadioML2018Dataset
    test_dataset: RadioML2018Dataset
    train_loader: DataLoader[dict[str, torch.Tensor]]
    validation_loader: DataLoader[dict[str, torch.Tensor]]
    test_loader: DataLoader[dict[str, torch.Tensor]]
    train_generator: torch.Generator
    class_names: list[str]

    def close(self) -> None:
        self.train_dataset.close()
        self.validation_dataset.close()
        self.test_dataset.close()


def _make_loader(
    dataset: RadioML2018Dataset,
    config: DataConfig,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader[dict[str, torch.Tensor]]:
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory and torch.cuda.is_available(),
        "persistent_workers": config.persistent_workers and config.num_workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(**kwargs)  # type: ignore[arg-type]


def build_data_bundle(
    config: DataConfig,
    seed: int,
    num_classes: int,
) -> DataBundle:
    hdf5_path = resolve_project_path(config.hdf5_path)
    splits_path = resolve_project_path(config.splits_path)
    class_names = load_class_names(config.classes_path, num_classes)

    train_transform = build_train_transform(
        sequence_length=config.sequence_length,
        enable_augmentation=config.train_augmentation,
    )
    evaluation_transform = build_eval_transform(config.sequence_length)

    train_dataset = RadioML2018Dataset(
        "train", hdf5_path, splits_path, transform=train_transform
    )
    validation_dataset = RadioML2018Dataset(
        "val", hdf5_path, splits_path, transform=evaluation_transform
    )
    test_dataset = RadioML2018Dataset(
        "test", hdf5_path, splits_path, transform=evaluation_transform
    )

    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    test_generator = torch.Generator().manual_seed(seed + 2)
    return DataBundle(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=_make_loader(
            train_dataset, config, shuffle=True, generator=train_generator
        ),
        validation_loader=_make_loader(
            validation_dataset,
            config,
            shuffle=False,
            generator=validation_generator,
        ),
        test_loader=_make_loader(
            test_dataset, config, shuffle=False, generator=test_generator
        ),
        train_generator=train_generator,
        class_names=class_names,
    )

