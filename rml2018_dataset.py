"""Reusable PyTorch dataset for RadioML 2018.01A.

The 20 GiB HDF5 file is never loaded into memory as a whole.  Every process
opens its own read-only HDF5 handle lazily, which makes the dataset safe to use
with ``torch.utils.data.DataLoader(num_workers > 0)``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HDF5 = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "RML2018.01a"
    / "GOLD_XYZ_OSC.0001_1024.hdf5"
)
DEFAULT_SPLITS = PROJECT_ROOT / "artifacts" / "P1-T7" / "split_indices.npz"
VALID_SPLITS = ("train", "val", "test")


class RadioML2018Dataset(Dataset[dict[str, torch.Tensor]]):
    """Read one fixed RadioML split from HDF5 on demand.

    Returned samples are dictionaries with:

    - ``iq``: float32 tensor shaped ``(2, 1024)``;
    - ``label``: int64 scalar class index;
    - ``snr``: int64 scalar SNR in dB;
    - ``index``: int64 scalar row number in the original HDF5 file.

    No normalization or augmentation is performed here.  Those transformations
    belong to P1-T8 and can be supplied through ``transform`` later.
    """

    def __init__(
        self,
        split: str,
        hdf5_path: str | Path = DEFAULT_HDF5,
        splits_path: str | Path = DEFAULT_SPLITS,
        transform: Any | None = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

        self.split = split
        self.hdf5_path = Path(hdf5_path).expanduser().resolve()
        self.splits_path = Path(splits_path).expanduser().resolve()
        self.transform = transform

        if not self.hdf5_path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")
        if not self.splits_path.is_file():
            raise FileNotFoundError(
                f"Split file not found: {self.splits_path}. "
                "Run P1-T7-build_splits.py first."
            )

        with np.load(self.splits_path, allow_pickle=False) as archive:
            if split not in archive.files:
                raise KeyError(f"Split {split!r} is absent from {self.splits_path}")
            # Keep only the compact row-index array in RAM (about 7 MiB for train).
            self.indices = np.asarray(archive[split], dtype=np.uint32)

        if self.indices.ndim != 1:
            raise ValueError(f"Split indices must be one-dimensional: {self.indices.shape}")

        self._file: h5py.File | None = None
        self._owner_pid: int | None = None

    def __len__(self) -> int:
        return int(len(self.indices))

    def _ensure_open(self) -> h5py.File:
        current_pid = os.getpid()
        if self._file is None or self._owner_pid != current_pid:
            self.close()
            self._file = h5py.File(self.hdf5_path, mode="r")
            self._owner_pid = current_pid
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._owner_pid = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Never pickle an HDF5 handle into a DataLoader worker."""
        state = self.__dict__.copy()
        state["_file"] = None
        state["_owner_pid"] = None
        return state

    @staticmethod
    def _as_sample(
        iq: np.ndarray,
        one_hot: np.ndarray,
        snr: np.ndarray | np.generic | int,
        raw_index: int,
        transform: Any | None,
    ) -> dict[str, torch.Tensor]:
        # HDF5 stores (time, I/Q); PyTorch models use (channel, time).
        iq_array = np.ascontiguousarray(iq.T, dtype=np.float32)
        if transform is not None:
            iq_array = transform(iq_array)
        iq_tensor = (
            iq_array
            if isinstance(iq_array, torch.Tensor)
            else torch.from_numpy(np.ascontiguousarray(iq_array))
        )
        return {
            "iq": iq_tensor,
            "label": torch.tensor(int(np.argmax(one_hot)), dtype=torch.long),
            "snr": torch.tensor(int(np.asarray(snr).reshape(-1)[0]), dtype=torch.long),
            "index": torch.tensor(raw_index, dtype=torch.long),
        }

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        raw_index = int(self.indices[position])
        file = self._ensure_open()
        return self._as_sample(
            file["X"][raw_index],
            file["Y"][raw_index],
            file["Z"][raw_index],
            raw_index,
            self.transform,
        )

    def __getitems__(
        self, positions: Sequence[int]
    ) -> list[dict[str, torch.Tensor]]:
        """Batch HDF5 reads while preserving DataLoader's requested order."""
        if not positions:
            return []

        requested = self.indices[np.asarray(positions, dtype=np.int64)]
        order = np.argsort(requested)
        sorted_indices = requested[order].astype(np.int64, copy=False)
        if np.any(sorted_indices[1:] == sorted_indices[:-1]):
            # Duplicate positions are legal for custom samplers; h5py fancy
            # indexing is not. Fall back to scalar reads in that rare case.
            return [self[position] for position in positions]

        file = self._ensure_open()
        x_sorted = file["X"][sorted_indices]
        y_sorted = file["Y"][sorted_indices]
        z_sorted = file["Z"][sorted_indices]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))

        return [
            self._as_sample(
                x_sorted[inverse[item]],
                y_sorted[inverse[item]],
                z_sorted[inverse[item]],
                int(requested[item]),
                self.transform,
            )
            for item in range(len(requested))
        ]

    def __enter__(self) -> "RadioML2018Dataset":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

