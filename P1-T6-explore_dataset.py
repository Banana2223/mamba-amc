"""P1-T6: explore and quality-check the RadioML 2018.01A dataset.

The script keeps the 20 GiB IQ array on disk. Label/SNR metadata are scanned
in chunks, and ``--mode full`` scans X sequentially in bounded-size batches.
It produces machine-readable and human-readable reports plus basic figures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
PLOT_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "RML2018.01a"
DEFAULT_HDF5 = DATASET_DIR / "GOLD_XYZ_OSC.0001_1024.hdf5"
DEFAULT_CLASSES = DATASET_DIR / "classes-fixed.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "P1-T6"
JSON_REPORT = OUTPUT_DIR / "data_quality_report.json"
MARKDOWN_REPORT = OUTPUT_DIR / "data_quality_report.md"

EXPECTED_SHAPES = {
    "X": (2_555_904, 1024, 2),
    "Y": (2_555_904, 24),
    "Z": (2_555_904, 1),
}
EXPECTED_SNRS = np.arange(-20, 32, 2, dtype=np.int64)
EXPECTED_CLASSES = 24
EXPECTED_SAMPLES_PER_GROUP = 4096
EXPECTED_GROUPS = EXPECTED_CLASSES * len(EXPECTED_SNRS)
DEFAULT_METADATA_CHUNK = 131_072
DEFAULT_X_CHUNK = 8192
DEFAULT_SAMPLES_PER_GROUP = 4
PREFERRED_PLOT_CLASSES = ("BPSK", "QPSK", "16QAM", "FM")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_classes(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Class-order file not found: {path}")
    classes = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(classes, list) or not all(
        isinstance(name, str) for name in classes
    ):
        raise ValueError(f"Class-order file is not a JSON string list: {path}")
    if len(classes) != EXPECTED_CLASSES or len(set(classes)) != len(classes):
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} unique classes, got {len(classes)}"
        )
    return classes


def describe_hdf5(file: h5py.File) -> dict[str, Any]:
    description: dict[str, Any] = {"root_keys": sorted(file.keys())}
    for name in ("X", "Y", "Z"):
        if name not in file:
            continue
        dataset = file[name]
        description[name] = {
            "shape": list(dataset.shape),
            "dtype": str(dataset.dtype),
            "chunks": list(dataset.chunks) if dataset.chunks else None,
            "compression": dataset.compression,
        }
    return description


def scan_metadata(
    file: h5py.File,
    samples_per_group: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Scan Y/Z fully and keep a small deterministic IQ sample index set."""
    y_dataset = file["Y"]
    z_dataset = file["Z"]
    total_samples = int(y_dataset.shape[0])
    snr_to_position = {int(snr): idx for idx, snr in enumerate(EXPECTED_SNRS)}
    distribution = np.zeros(
        (EXPECTED_CLASSES, len(EXPECTED_SNRS)), dtype=np.int64
    )
    invalid_onehot_rows = 0
    unknown_snr_rows = 0
    selected_by_group: list[list[int]] = [
        [] for _ in range(EXPECTED_GROUPS)
    ]

    for start in range(0, total_samples, chunk_size):
        stop = min(start + chunk_size, total_samples)
        y = y_dataset[start:stop]
        z = z_dataset[start:stop, 0].astype(np.int64, copy=False)

        valid_binary = np.logical_or(y == 0, y == 1).all(axis=1)
        valid_sum = y.sum(axis=1) == 1
        valid = valid_binary & valid_sum
        invalid_onehot_rows += int((~valid).sum())
        labels = np.argmax(y, axis=1).astype(np.int64, copy=False)

        snr_positions = np.full(len(z), -1, dtype=np.int64)
        for snr, position in snr_to_position.items():
            snr_positions[z == snr] = position
        unknown_snr_rows += int((snr_positions < 0).sum())

        valid_group = (
            valid
            & (labels >= 0)
            & (labels < EXPECTED_CLASSES)
            & (snr_positions >= 0)
        )
        pair_ids = labels[valid_group] * len(EXPECTED_SNRS) + snr_positions[valid_group]
        distribution += np.bincount(
            pair_ids, minlength=EXPECTED_GROUPS
        ).reshape(EXPECTED_CLASSES, len(EXPECTED_SNRS))

        local_valid_indices = np.flatnonzero(valid_group)
        local_pair_ids = labels[local_valid_indices] * len(EXPECTED_SNRS) + snr_positions[
            local_valid_indices
        ]
        for pair_id in np.unique(local_pair_ids):
            selected = selected_by_group[int(pair_id)]
            needed = samples_per_group - len(selected)
            if needed <= 0:
                continue
            candidates = local_valid_indices[local_pair_ids == pair_id][:needed]
            selected.extend((start + candidates).astype(int).tolist())

        if start == 0 or stop == total_samples or stop % (chunk_size * 5) == 0:
            print(f"Metadata scan: {stop:,}/{total_samples:,}")

    selected_indices = np.array(
        sorted(index for group in selected_by_group for index in group),
        dtype=np.int64,
    )
    missing_sample_groups = [
        pair_id
        for pair_id, group in enumerate(selected_by_group)
        if len(group) < samples_per_group
    ]

    return {
        "distribution": distribution,
        "invalid_onehot_rows": invalid_onehot_rows,
        "unknown_snr_rows": unknown_snr_rows,
        "selected_indices": selected_indices,
        "missing_sample_groups": missing_sample_groups,
    }


def read_selected_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read sorted indices as contiguous runs to avoid inefficient fancy I/O."""
    if len(indices) == 0:
        return np.empty((0,) + dataset.shape[1:], dtype=dataset.dtype)

    output = np.empty((len(indices),) + dataset.shape[1:], dtype=dataset.dtype)
    run_start = 0
    for position in range(1, len(indices) + 1):
        run_ended = position == len(indices) or indices[position] != indices[position - 1] + 1
        if not run_ended:
            continue
        source_start = int(indices[run_start])
        source_stop = int(indices[position - 1]) + 1
        output[run_start:position] = dataset[source_start:source_stop]
        run_start = position
    return output


def finite_channel_stats(values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(finite_values.size),
        "mean": float(np.mean(finite_values, dtype=np.float64)),
        "std": float(np.std(finite_values, dtype=np.float64)),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
    }


def sample_iq_statistics(
    x: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
) -> dict[str, Any]:
    finite_mask = np.isfinite(x)
    sample_has_nonfinite = ~finite_mask.all(axis=(1, 2))
    all_zero_samples = np.all(x == 0, axis=(1, 2))
    sample_power = np.mean(np.square(x, dtype=np.float64), axis=(1, 2))
    power_by_snr = {}
    for snr in EXPECTED_SNRS:
        mask = snrs == snr
        if mask.any():
            power_by_snr[str(int(snr))] = {
                "sample_count": int(mask.sum()),
                "mean_power": float(np.mean(sample_power[mask])),
                "std_power": float(np.std(sample_power[mask])),
            }

    return {
        "scope": "stratified_sample",
        "sample_count": int(len(x)),
        "class_count_in_sample": int(len(np.unique(labels))),
        "snr_count_in_sample": int(len(np.unique(snrs))),
        "nonfinite_value_count": int((~finite_mask).sum()),
        "samples_with_nonfinite": int(sample_has_nonfinite.sum()),
        "all_zero_samples": int(all_zero_samples.sum()),
        "I": finite_channel_stats(x[:, :, 0]),
        "Q": finite_channel_stats(x[:, :, 1]),
        "power_by_snr": power_by_snr,
    }


def scan_full_iq(
    file: h5py.File,
    chunk_size: int,
) -> dict[str, Any]:
    """Sequentially scan all X values with bounded memory."""
    x_dataset = file["X"]
    z_dataset = file["Z"]
    total_samples = int(x_dataset.shape[0])

    channel_count = np.zeros(2, dtype=np.int64)
    channel_sum = np.zeros(2, dtype=np.float64)
    channel_sum_sq = np.zeros(2, dtype=np.float64)
    channel_min = np.full(2, np.inf, dtype=np.float64)
    channel_max = np.full(2, -np.inf, dtype=np.float64)
    nonfinite_value_count = 0
    samples_with_nonfinite = 0
    all_zero_samples = 0
    snr_power_sum = np.zeros(len(EXPECTED_SNRS), dtype=np.float64)
    snr_power_sum_sq = np.zeros(len(EXPECTED_SNRS), dtype=np.float64)
    snr_power_count = np.zeros(len(EXPECTED_SNRS), dtype=np.int64)
    snr_to_position = {int(snr): idx for idx, snr in enumerate(EXPECTED_SNRS)}

    start_time = time.perf_counter()
    for start in range(0, total_samples, chunk_size):
        stop = min(start + chunk_size, total_samples)
        x = x_dataset[start:stop]
        z = z_dataset[start:stop, 0].astype(np.int64, copy=False)
        finite = np.isfinite(x)
        nonfinite_value_count += int((~finite).sum())
        samples_with_nonfinite += int((~finite.all(axis=(1, 2))).sum())
        all_zero_samples += int(np.all(x == 0, axis=(1, 2)).sum())

        for channel in range(2):
            values = x[:, :, channel]
            channel_finite = finite[:, :, channel]
            finite_values = values[channel_finite]
            if finite_values.size:
                channel_count[channel] += finite_values.size
                channel_sum[channel] += np.sum(finite_values, dtype=np.float64)
                channel_sum_sq[channel] += np.einsum(
                    "i,i->", finite_values, finite_values, dtype=np.float64
                )
                channel_min[channel] = min(
                    channel_min[channel], float(np.min(finite_values))
                )
                channel_max[channel] = max(
                    channel_max[channel], float(np.max(finite_values))
                )

        safe_x = np.where(finite, x, 0.0)
        sample_power = np.einsum(
            "ijk,ijk->i", safe_x, safe_x, dtype=np.float64
        ) / (x.shape[1] * x.shape[2])
        for snr, position in snr_to_position.items():
            mask = z == snr
            if not mask.any():
                continue
            values = sample_power[mask]
            snr_power_count[position] += len(values)
            snr_power_sum[position] += np.sum(values, dtype=np.float64)
            snr_power_sum_sq[position] += np.einsum(
                "i,i->", values, values, dtype=np.float64
            )

        if start == 0 or stop == total_samples or stop % (chunk_size * 20) == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"Full IQ scan: {stop:,}/{total_samples:,} "
                f"({100 * stop / total_samples:.1f}%, {elapsed:.1f}s)"
            )

    channels = {}
    for index, name in enumerate(("I", "Q")):
        count = int(channel_count[index])
        mean = channel_sum[index] / count
        variance = max(channel_sum_sq[index] / count - mean**2, 0.0)
        channels[name] = {
            "count": count,
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
            "min": float(channel_min[index]),
            "max": float(channel_max[index]),
        }

    power_by_snr = {}
    for position, snr in enumerate(EXPECTED_SNRS):
        count = int(snr_power_count[position])
        mean = snr_power_sum[position] / count
        variance = max(snr_power_sum_sq[position] / count - mean**2, 0.0)
        power_by_snr[str(int(snr))] = {
            "sample_count": count,
            "mean_power": float(mean),
            "std_power": float(math.sqrt(variance)),
        }

    return {
        "scope": "full_dataset",
        "sample_count": total_samples,
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
        "nonfinite_value_count": nonfinite_value_count,
        "samples_with_nonfinite": samples_with_nonfinite,
        "all_zero_samples": all_zero_samples,
        **channels,
        "power_by_snr": power_by_snr,
    }


def choose_plot_samples(
    selected_indices: np.ndarray,
    selected_x: np.ndarray,
    selected_labels: np.ndarray,
    selected_snrs: np.ndarray,
    classes: list[str],
    plot_snr: int,
) -> list[dict[str, Any]]:
    chosen = []
    for class_name in PREFERRED_PLOT_CLASSES:
        if class_name not in classes:
            continue
        class_index = classes.index(class_name)
        matches = np.flatnonzero(
            (selected_labels == class_index) & (selected_snrs == plot_snr)
        )
        if not len(matches):
            continue
        position = int(matches[0])
        chosen.append(
            {
                "class_name": class_name,
                "class_index": class_index,
                "snr": plot_snr,
                "dataset_index": int(selected_indices[position]),
                "x": selected_x[position],
            }
        )
    return chosen


def plot_distribution(
    distribution: np.ndarray,
    classes: list[str],
) -> Path:
    path = OUTPUT_DIR / "class_snr_distribution.png"
    figure, axis = plt.subplots(figsize=(13, 8))
    image = axis.imshow(distribution, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(EXPECTED_SNRS)), EXPECTED_SNRS, rotation=45)
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Modulation class")
    axis.set_title("RML2018.01A sample count by class and SNR")
    figure.colorbar(image, ax=axis, label="Samples")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_waveforms(chosen: list[dict[str, Any]]) -> Path | None:
    if not chosen:
        return None
    path = OUTPUT_DIR / "sample_waveforms.png"
    figure, axes = plt.subplots(
        len(chosen), 1, figsize=(13, 2.7 * len(chosen)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, sample in zip(axes, chosen):
        x = sample["x"]
        axis.plot(x[:, 0], label="I", linewidth=0.9)
        axis.plot(x[:, 1], label="Q", linewidth=0.9, alpha=0.8)
        axis.set_title(
            f"{sample['class_name']} at {sample['snr']} dB "
            f"(index {sample['dataset_index']})"
        )
        axis.set_ylabel("Amplitude")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time sample")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_constellations(chosen: list[dict[str, Any]]) -> Path | None:
    if not chosen:
        return None
    path = OUTPUT_DIR / "sample_constellations.png"
    columns = 2
    rows = math.ceil(len(chosen) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(10, 5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, sample in zip(axes, chosen):
        x = sample["x"]
        axis.scatter(x[:, 0], x[:, 1], s=7, alpha=0.45)
        axis.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        axis.axvline(0, color="black", linewidth=0.5, alpha=0.4)
        axis.set_title(f"{sample['class_name']} at {sample['snr']} dB")
        axis.set_xlabel("I")
        axis.set_ylabel("Q")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    for axis in axes[len(chosen):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_iq_histogram(x: np.ndarray) -> Path:
    path = OUTPUT_DIR / "iq_value_distribution.png"
    i_values = x[:, :, 0].ravel()
    q_values = x[:, :, 1].ravel()
    max_values = 500_000
    stride = max(1, math.ceil(len(i_values) / max_values))
    i_values = i_values[::stride]
    q_values = q_values[::stride]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(i_values, bins=160, density=True, alpha=0.55, label="I")
    axis.hist(q_values, bins=160, density=True, alpha=0.55, label="Q")
    axis.set_xlabel("Amplitude")
    axis.set_ylabel("Density")
    axis.set_title("Stratified IQ amplitude distribution")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_power_by_snr(iq_statistics: dict[str, Any]) -> Path:
    path = OUTPUT_DIR / "signal_power_by_snr.png"
    values = iq_statistics["power_by_snr"]
    means = [values[str(int(snr))]["mean_power"] for snr in EXPECTED_SNRS]
    stds = [values[str(int(snr))]["std_power"] for snr in EXPECTED_SNRS]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.errorbar(EXPECTED_SNRS, means, yerr=stds, marker="o", capsize=2)
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Mean I/Q power")
    axis.set_title(f"Signal power by SNR ({iq_statistics['scope']})")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def relative(path: Path | None) -> str | None:
    return str(path.relative_to(PROJECT_ROOT)) if path is not None else None


def build_validation(
    structure: dict[str, Any],
    distribution: np.ndarray,
    metadata: dict[str, Any],
    iq_statistics: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    errors = []
    for name, expected_shape in EXPECTED_SHAPES.items():
        actual = tuple(structure.get(name, {}).get("shape", []))
        if actual != expected_shape:
            errors.append(f"{name} shape: expected {expected_shape}, got {actual}")
    if metadata["invalid_onehot_rows"]:
        errors.append(f"Invalid one-hot rows: {metadata['invalid_onehot_rows']}")
    if metadata["unknown_snr_rows"]:
        errors.append(f"Rows with unknown SNR: {metadata['unknown_snr_rows']}")
    if np.any(distribution != EXPECTED_SAMPLES_PER_GROUP):
        errors.append("Not every class-SNR group contains 4096 samples")
    if iq_statistics["nonfinite_value_count"]:
        errors.append(
            f"Non-finite IQ values: {iq_statistics['nonfinite_value_count']}"
        )
    if iq_statistics["all_zero_samples"]:
        errors.append(f"All-zero IQ samples: {iq_statistics['all_zero_samples']}")

    passed = not errors
    return {
        "passed": passed,
        "completion_ready": passed and mode == "full",
        "mode": mode,
        "errors": errors,
        "note": (
            "Full X scan completed; P1-T6 can be closed."
            if passed and mode == "full"
            else "Quick sample check only; run --mode full before closing P1-T6."
            if passed
            else "Resolve validation errors before continuing."
        ),
    }


def write_markdown(report: dict[str, Any]) -> None:
    distribution = report["distribution"]
    quality = report["iq_statistics"]
    validation = report["validation"]
    figures = report["figures"]

    lines = [
        "# P1-T6 RML2018.01A data exploration and quality report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Scan mode: `{report['mode']}`",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Validation passed: `{validation['passed']}`",
        f"- P1-T6 completion ready: `{validation['completion_ready']}`",
        "",
        "## Structure",
        "",
        "| Dataset | Shape | Dtype |",
        "|---|---:|---|",
    ]
    for name in ("X", "Y", "Z"):
        item = report["hdf5"][name]
        lines.append(f"| {name} | `{tuple(item['shape'])}` | `{item['dtype']}` |")

    lines.extend(
        [
            "",
            "## Distribution",
            "",
            f"- Total samples: `{distribution['total_samples']:,}`",
            f"- Classes: `{distribution['class_count']}`",
            f"- SNR levels: `{distribution['snr_count']}`",
            f"- Class-SNR groups: `{distribution['group_count']}`",
            f"- Samples per group: `{distribution['group_count_min']}` to "
            f"`{distribution['group_count_max']}`",
            f"- Invalid one-hot rows: `{distribution['invalid_onehot_rows']}`",
            "",
            "## IQ quality",
            "",
            f"- Scope: `{quality['scope']}`",
            f"- Samples checked: `{quality['sample_count']:,}`",
            f"- Non-finite values: `{quality['nonfinite_value_count']}`",
            f"- Samples containing non-finite values: "
            f"`{quality['samples_with_nonfinite']}`",
            f"- All-zero samples: `{quality['all_zero_samples']}`",
            f"- I range: `{quality['I']['min']:.6g}` to `{quality['I']['max']:.6g}`",
            f"- Q range: `{quality['Q']['min']:.6g}` to `{quality['Q']['max']:.6g}`",
            "",
            "## Figures",
            "",
        ]
    )
    for name, path in figures.items():
        if path:
            lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            validation["note"],
            "",
            "The dataset is synthetic historical research data. Passing this "
            "quality check does not remove the need to report dataset errata and "
            "the lack of real over-the-air validation as limitations.",
            "",
        ]
    )
    MARKDOWN_REPORT.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    hdf5_path = args.hdf5.expanduser().resolve()
    classes_path = args.classes.expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    classes = load_classes(classes_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with h5py.File(hdf5_path, "r") as file:
        structure = describe_hdf5(file)
        missing = [name for name in ("X", "Y", "Z") if name not in file]
        if missing:
            raise RuntimeError(f"Missing HDF5 datasets: {missing}")

        metadata = scan_metadata(
            file,
            samples_per_group=args.samples_per_group,
            chunk_size=args.metadata_chunk,
        )
        distribution = metadata["distribution"]
        selected_indices = metadata["selected_indices"]
        selected_x = read_selected_rows(file["X"], selected_indices)
        selected_y = read_selected_rows(file["Y"], selected_indices)
        selected_z = read_selected_rows(file["Z"], selected_indices)[:, 0].astype(
            np.int64, copy=False
        )
        selected_labels = np.argmax(selected_y, axis=1).astype(np.int64, copy=False)

        sampled_statistics = sample_iq_statistics(
            selected_x, selected_labels, selected_z
        )
        iq_statistics = (
            scan_full_iq(file, chunk_size=args.x_chunk)
            if args.mode == "full"
            else sampled_statistics
        )

    chosen = choose_plot_samples(
        selected_indices,
        selected_x,
        selected_labels,
        selected_z,
        classes,
        args.plot_snr,
    )
    figures = {
        "class_snr_distribution": relative(plot_distribution(distribution, classes)),
        "sample_waveforms": relative(plot_waveforms(chosen)),
        "sample_constellations": relative(plot_constellations(chosen)),
        "iq_value_distribution": relative(plot_iq_histogram(selected_x)),
        "signal_power_by_snr": relative(plot_power_by_snr(iq_statistics)),
    }

    class_counts = distribution.sum(axis=1)
    snr_counts = distribution.sum(axis=0)
    validation = build_validation(
        structure, distribution, metadata, iq_statistics, args.mode
    )
    report: dict[str, Any] = {
        "generated_at": now_iso(),
        "mode": args.mode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "dataset_path": str(hdf5_path.relative_to(PROJECT_ROOT)),
        "classes_path": str(classes_path.relative_to(PROJECT_ROOT)),
        "hdf5": structure,
        "classes": classes,
        "snrs_db": EXPECTED_SNRS.astype(int).tolist(),
        "distribution": {
            "total_samples": int(distribution.sum()),
            "class_count": len(classes),
            "snr_count": len(EXPECTED_SNRS),
            "group_count": int(np.count_nonzero(distribution)),
            "group_count_min": int(distribution.min()),
            "group_count_max": int(distribution.max()),
            "class_counts": {
                name: int(class_counts[index]) for index, name in enumerate(classes)
            },
            "snr_counts": {
                str(int(snr)): int(snr_counts[index])
                for index, snr in enumerate(EXPECTED_SNRS)
            },
            "class_snr_matrix": distribution.astype(int).tolist(),
            "invalid_onehot_rows": metadata["invalid_onehot_rows"],
            "unknown_snr_rows": metadata["unknown_snr_rows"],
        },
        "stratified_sample": {
            "samples_per_group": args.samples_per_group,
            "selected_sample_count": int(len(selected_indices)),
            "missing_sample_groups": metadata["missing_sample_groups"],
            "iq_statistics": sampled_statistics,
            "plot_examples": [
                {
                    key: value
                    for key, value in sample.items()
                    if key != "x"
                }
                for sample in chosen
            ],
        },
        "iq_statistics": iq_statistics,
        "figures": figures,
        "validation": validation,
    }

    JSON_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(report)
    print(f"JSON report:     {JSON_REPORT}")
    print(f"Markdown report: {MARKDOWN_REPORT}")
    print(f"Validation:      {validation}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore and quality-check RadioML 2018.01A without loading it all into RAM."
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=DEFAULT_HDF5,
        help="Path to GOLD_XYZ_OSC.0001_1024.hdf5.",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=DEFAULT_CLASSES,
        help="Corrected JSON class-order file.",
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="quick samples X; full sequentially scans all IQ values.",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=DEFAULT_SAMPLES_PER_GROUP,
        help="IQ samples retained from each class-SNR group for plots.",
    )
    parser.add_argument(
        "--metadata-chunk",
        type=int,
        default=DEFAULT_METADATA_CHUNK,
        help="Y/Z rows per metadata scan batch.",
    )
    parser.add_argument(
        "--x-chunk",
        type=int,
        default=DEFAULT_X_CHUNK,
        help="X samples per full-scan batch.",
    )
    parser.add_argument(
        "--plot-snr",
        type=int,
        choices=EXPECTED_SNRS.astype(int).tolist(),
        default=30,
        help="SNR used for waveform and constellation examples.",
    )
    args = parser.parse_args()
    if args.samples_per_group <= 0:
        parser.error("--samples-per-group must be positive")
    if args.metadata_chunk <= 0 or args.x_chunk <= 0:
        parser.error("chunk sizes must be positive")
    return args


def main() -> int:
    try:
        report = run(parse_args())
        return 0 if report["validation"]["passed"] else 1
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
