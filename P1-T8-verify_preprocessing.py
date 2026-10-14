"""P1-T8: verify IQ preprocessing and configurable channel augmentation."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from iq_transforms import (
    IQTransform,
    IQTransformConfig,
    build_eval_transform,
    build_train_transform,
)
from rml2018_dataset import DEFAULT_HDF5, DEFAULT_SPLITS, RadioML2018Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "P1-T8"
JSON_REPORT = OUTPUT_DIR / "preprocessing_report.json"
MARKDOWN_REPORT = OUTPUT_DIR / "preprocessing_report.md"
FIGURE_PATH = OUTPUT_DIR / "preprocessing_examples.png"
PLOT_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SEED = 2026
TARGET_CLASS = 4  # QPSK in classes-fixed.json
TARGET_SNR = 20


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def complex_power(iq: torch.Tensor) -> float:
    return float(torch.mean(iq[0].square() + iq[1].square()).item())


def find_example_index(
    hdf5_path: Path,
    split_indices: np.ndarray,
    class_index: int,
    snr: int,
    chunk_size: int = 131_072,
) -> int:
    """Find a readable example belonging to the train split without assumptions."""
    membership = np.zeros(2_555_904, dtype=bool)
    membership[split_indices] = True
    with h5py.File(hdf5_path, mode="r") as file:
        total = len(file["Y"])
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            y = file["Y"][start:stop]
            z = file["Z"][start:stop, 0]
            labels = np.argmax(y, axis=1)
            local = np.flatnonzero(
                membership[start:stop] & (labels == class_index) & (z == snr)
            )
            if len(local):
                return start + int(local[0])
    raise RuntimeError(f"No train example found for class={class_index}, SNR={snr}")


def force_all_augmentations(sequence_length: int = 1024) -> IQTransform:
    """Use every augmentation once for validation and visualization."""
    return IQTransform(
        IQTransformConfig(
            sequence_length=sequence_length,
            crop_mode="center",
            normalization="rms",
            awgn_probability=1.0,
            awgn_snr_db_min=20.0,
            awgn_snr_db_max=20.0,
            phase_probability=1.0,
            phase_max_radians=math.pi,
            frequency_probability=1.0,
            frequency_max_abs_cycles_per_sample=0.005,
            multipath_probability=1.0,
            multipath_max_echoes=2,
            multipath_max_delay=16,
            multipath_max_gain=0.35,
            multipath_decay=0.6,
        )
    )


def plot_examples(raw: torch.Tensor, normalized: torch.Tensor, augmented: torch.Tensor) -> None:
    rows = [
        ("Raw QPSK at 20 dB", raw),
        ("RMS normalized", normalized),
        ("Forced channel augmentations", augmented),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(13, 10))
    for row, (title, iq) in enumerate(rows):
        wave_axis, constellation_axis = axes[row]
        time_limit = min(256, iq.shape[-1])
        wave_axis.plot(iq[0, :time_limit], linewidth=0.9, label="I")
        wave_axis.plot(iq[1, :time_limit], linewidth=0.9, label="Q", alpha=0.8)
        wave_axis.set_title(title)
        wave_axis.set_xlabel("Time sample")
        wave_axis.set_ylabel("Amplitude")
        wave_axis.grid(alpha=0.25)
        wave_axis.legend(loc="upper right")

        constellation_axis.scatter(iq[0], iq[1], s=7, alpha=0.4)
        constellation_axis.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        constellation_axis.axvline(0, color="black", linewidth=0.5, alpha=0.4)
        constellation_axis.set_title(f"{title} constellation")
        constellation_axis.set_xlabel("I")
        constellation_axis.set_ylabel("Q")
        constellation_axis.set_aspect("equal", adjustable="box")
        constellation_axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def first_augmented_batch(
    hdf5_path: Path,
    splits_path: Path,
    batch_size: int,
    num_workers: int,
) -> dict[str, torch.Tensor]:
    """Create one deterministically seeded augmented training batch."""
    dataset = RadioML2018Dataset(
        "train",
        hdf5_path,
        splits_path,
        transform=build_train_transform(enable_augmentation=True),
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        persistent_workers=False,
    )
    batch = next(iter(loader))
    copied = {name: value.clone() for name, value in batch.items()}
    dataset.close()
    return copied


def write_markdown(report: dict[str, Any]) -> None:
    checks = report["checks"]
    lines = [
        "# P1-T8 IQ preprocessing and augmentation report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Validation passed: `{report['passed']}`",
        f"- Example: QPSK at `{report['example']['snr_db']}` dB, HDF5 row "
        f"`{report['example']['raw_index']}`",
        "",
        "## Fixed preprocessing protocol",
        "",
        "- Input representation: real tensor `(2, length)`, channel order I then Q.",
        "- Default sequence length: 1024; shorter experiments use random train crop "
        "and center validation/test crop.",
        "- Normalization: per-sample complex RMS, with one shared scale for I and Q.",
        "- No per-channel centering: carrier/DC information that may distinguish AM "
        "subclasses is retained.",
        "- Validation and test transforms are deterministic and never augmented.",
        "- Training augmentation is disabled for the first baseline and enabled only "
        "in explicitly marked experiments.",
        "",
        "## Optional training augmentations",
        "",
        "- AWGN: probability 0.25, added-noise SNR sampled from 15 to 30 dB.",
        "- Constant phase rotation: probability 0.50, angle up to +/-pi.",
        "- Carrier frequency offset: probability 0.25, up to +/-0.005 cycles/sample.",
        "- Multipath echoes: probability 0.20, at most 2 echoes and 16-sample delay.",
        "",
        "The original dataset SNR metadata is retained as provenance. After synthetic "
        "channel augmentation it must not be interpreted as the exact effective SNR.",
        "",
        "## Verification",
        "",
    ]
    for name, value in checks.items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(
        [
            "",
            f"- Raw complex power: `{report['example']['raw_complex_power']:.6f}`",
            f"- Normalized complex power: "
            f"`{report['example']['normalized_complex_power']:.6f}`",
            f"- Augmented/normalized complex power: "
            f"`{report['example']['augmented_complex_power']:.6f}`",
            f"- Figure: `{report['figure']}`",
            "",
        ]
    )
    MARKDOWN_REPORT.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    hdf5_path = args.hdf5.expanduser().resolve()
    splits_path = args.splits.expanduser().resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(splits_path, allow_pickle=False) as archive:
        train_indices = np.asarray(archive["train"], dtype=np.uint32)
    raw_index = find_example_index(
        hdf5_path, train_indices, TARGET_CLASS, TARGET_SNR
    )
    with h5py.File(hdf5_path, mode="r") as file:
        raw = torch.from_numpy(
            np.ascontiguousarray(file["X"][raw_index].T, dtype=np.float32)
        )

    eval_transform = build_eval_transform()
    normalized_first = eval_transform(raw)
    normalized_second = eval_transform(raw)

    forced_transform = force_all_augmentations()
    torch.manual_seed(SEED)
    augmented_first = forced_transform(raw)
    torch.manual_seed(SEED)
    augmented_repeat = forced_transform(raw)
    torch.manual_seed(SEED + 1)
    augmented_different_seed = forced_transform(raw)

    short_eval = build_eval_transform(sequence_length=512)(raw)
    torch.manual_seed(SEED)
    short_train = build_train_transform(sequence_length=512)(raw)

    loader_dataset = RadioML2018Dataset(
        "val", hdf5_path, splits_path, transform=eval_transform
    )
    loader = DataLoader(loader_dataset, batch_size=32, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    loader_dataset.close()

    augmented_batch_first = first_augmented_batch(
        hdf5_path, splits_path, args.batch_size, args.num_workers
    )
    augmented_batch_repeat = first_augmented_batch(
        hdf5_path, splits_path, args.batch_size, args.num_workers
    )
    batch_complex_power = torch.mean(
        batch["iq"][:, 0].square() + batch["iq"][:, 1].square(), dim=1
    )

    checks = {
        "eval_is_deterministic": bool(torch.equal(normalized_first, normalized_second)),
        "seeded_augmentation_is_reproducible": bool(
            torch.equal(augmented_first, augmented_repeat)
        ),
        "seeded_dataloader_augmentation_is_reproducible": all(
            torch.equal(augmented_batch_first[name], augmented_batch_repeat[name])
            for name in ("iq", "label", "snr", "index")
        ),
        "different_seed_changes_augmentation": bool(
            not torch.equal(augmented_first, augmented_different_seed)
        ),
        "all_outputs_are_finite": bool(
            torch.isfinite(normalized_first).all()
            and torch.isfinite(augmented_first).all()
            and torch.isfinite(batch["iq"]).all()
        ),
        "full_length_shape_is_2x1024": tuple(normalized_first.shape) == (2, 1024),
        "short_crop_shape_is_2x512": (
            tuple(short_eval.shape) == (2, 512)
            and tuple(short_train.shape) == (2, 512)
        ),
        "normalized_power_is_one": abs(complex_power(normalized_first) - 1.0) < 1e-5,
        "augmented_power_is_one": abs(complex_power(augmented_first) - 1.0) < 1e-5,
        "loader_batch_shape_is_32x2x1024": tuple(batch["iq"].shape)
        == (32, 2, 1024),
        "loader_metadata_shapes_are_valid": all(
            tuple(batch[name].shape) == (32,) for name in ("label", "snr", "index")
        ),
        "loader_samples_have_unit_power": bool(
            torch.allclose(batch_complex_power, torch.ones_like(batch_complex_power), atol=1e-5)
        ),
    }
    passed = all(checks.values())
    plot_examples(raw, normalized_first, augmented_first)

    report: dict[str, Any] = {
        "generated_at": now_iso(),
        "passed": passed,
        "seed": SEED,
        "protocol": {
            "train_without_augmentation": build_train_transform(
                enable_augmentation=False
            ).config.to_dict(),
            "train_with_augmentation": build_train_transform(
                enable_augmentation=True
            ).config.to_dict(),
            "validation_and_test": eval_transform.config.to_dict(),
        },
        "example": {
            "class_index": TARGET_CLASS,
            "class_name": "QPSK",
            "snr_db": TARGET_SNR,
            "raw_index": raw_index,
            "raw_complex_power": complex_power(raw),
            "normalized_complex_power": complex_power(normalized_first),
            "augmented_complex_power": complex_power(augmented_first),
        },
        "loader_batch": {
            "iq_shape": list(batch["iq"].shape),
            "iq_dtype": str(batch["iq"].dtype),
            "label_shape": list(batch["label"].shape),
            "snr_shape": list(batch["snr"].shape),
            "index_shape": list(batch["index"].shape),
            "augmentation_reproducibility_batch_size": args.batch_size,
            "augmentation_reproducibility_num_workers": args.num_workers,
        },
        "checks": checks,
        "snr_metadata_note": (
            "Original SNR is retained as provenance; after augmentation it is not "
            "the exact effective SNR."
        ),
        "figure": str(FIGURE_PATH.relative_to(PROJECT_ROOT)),
    }
    JSON_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(report)
    print(f"JSON report:     {JSON_REPORT}")
    print(f"Markdown report: {MARKDOWN_REPORT}")
    print(f"Figure:          {FIGURE_PATH}")
    print(f"Validation:      {'passed' if passed else 'failed'}")
    if not passed:
        raise RuntimeError(f"P1-T8 validation failed: {checks}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify RML2018.01A preprocessing and augmentation."
    )
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
