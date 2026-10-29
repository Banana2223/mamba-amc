"""Reusable fixed-sample helpers for baseline overfit verification."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from iq_transforms import build_eval_transform
from rml2018_dataset import RadioML2018Dataset


def select_balanced_train_positions(
    hdf5_path: Path,
    splits_path: Path,
    samples_per_class: int,
    num_classes: int = 24,
) -> list[int]:
    """Select equal samples per class using contiguous HDF5 metadata reads."""
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    with np.load(splits_path, allow_pickle=False) as archive:
        train_indices = np.asarray(archive["train"], dtype=np.uint32)

    selected: list[list[int]] = [[] for _ in range(num_classes)]
    chunk_size = 131_072
    with h5py.File(hdf5_path, mode="r") as file:
        total_rows = len(file["Y"])
        for raw_start in range(0, total_rows, chunk_size):
            raw_stop = min(raw_start + chunk_size, total_rows)
            position_start = int(np.searchsorted(train_indices, raw_start, side="left"))
            position_stop = int(np.searchsorted(train_indices, raw_stop, side="left"))
            positions = np.arange(position_start, position_stop, dtype=np.int64)
            if len(positions) == 0:
                continue
            raw_indices = train_indices[positions].astype(np.int64, copy=False)
            labels = np.argmax(
                file["Y"][raw_start:raw_stop][raw_indices - raw_start], axis=1
            )
            for position, label in zip(positions.tolist(), labels.tolist(), strict=True):
                if len(selected[label]) < samples_per_class:
                    selected[label].append(position)
            if all(len(items) == samples_per_class for items in selected):
                break

    missing = [index for index, items in enumerate(selected) if not items]
    if missing:
        raise RuntimeError(f"Could not select samples for classes: {missing}")
    return [position for items in selected for position in items]


def load_fixed_balanced_batch(
    hdf5_path: Path,
    splits_path: Path,
    samples_per_class: int,
    sequence_length: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    positions = select_balanced_train_positions(
        hdf5_path,
        splits_path,
        samples_per_class=samples_per_class,
    )
    dataset = RadioML2018Dataset(
        "train",
        hdf5_path,
        splits_path,
        transform=build_eval_transform(sequence_length=sequence_length),
    )
    try:
        samples = [dataset[position] for position in positions]
    finally:
        dataset.close()
    return (
        torch.stack([sample["iq"] for sample in samples]),
        torch.stack([sample["label"] for sample in samples]),
        [int(sample["index"].item()) for sample in samples],
    )
