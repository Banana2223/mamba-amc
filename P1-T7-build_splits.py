"""P1-T7: build and verify fixed stratified RadioML 2018.01A splits.

The script scans only Y/Z metadata, independently shuffles every
``modulation class x SNR`` group, writes compact uint32 row indices, checks
index-level leakage and coverage, and smoke-tests the reusable PyTorch loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from rml2018_dataset import RadioML2018Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "RML2018.01a"
DEFAULT_HDF5 = DATASET_DIR / "GOLD_XYZ_OSC.0001_1024.hdf5"
DEFAULT_CLASSES = DATASET_DIR / "classes-fixed.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "P1-T7"
DEFAULT_SPLITS = OUTPUT_DIR / "split_indices.npz"
MANIFEST_PATH = OUTPUT_DIR / "split_manifest.json"
REPORT_PATH = OUTPUT_DIR / "split_report.md"

EXPECTED_SAMPLES = 2_555_904
EXPECTED_CLASSES = 24
EXPECTED_SNRS = np.arange(-20, 32, 2, dtype=np.int64)
EXPECTED_GROUP_SIZE = 4096
SPLIT_NAMES = ("train", "val", "test")
SPLIT_PER_GROUP = {"train": 2868, "val": 614, "test": 614}
DEFAULT_SEED = 2026
DEFAULT_CHUNK_SIZE = 131_072


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_classes(path: Path) -> list[str]:
    classes = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(classes, list)
        or len(classes) != EXPECTED_CLASSES
        or len(set(classes)) != EXPECTED_CLASSES
        or not all(isinstance(item, str) for item in classes)
    ):
        raise ValueError(f"Expected {EXPECTED_CLASSES} unique class names in {path}")
    return classes


def collect_group_indices(
    file: h5py.File,
    chunk_size: int,
) -> list[np.ndarray]:
    """Collect compact uint32 row indices for all class-SNR strata."""
    y_dataset = file["Y"]
    z_dataset = file["Z"]
    total = int(len(y_dataset))
    if total != EXPECTED_SAMPLES or len(z_dataset) != total:
        raise ValueError(f"Unexpected dataset length: Y={total}, Z={len(z_dataset)}")

    snr_to_position = {int(snr): i for i, snr in enumerate(EXPECTED_SNRS)}
    group_count = EXPECTED_CLASSES * len(EXPECTED_SNRS)
    fragments: list[list[np.ndarray]] = [[] for _ in range(group_count)]

    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        y = y_dataset[start:stop]
        z = z_dataset[start:stop, 0].astype(np.int64, copy=False)
        valid = np.logical_or(y == 0, y == 1).all(axis=1) & (y.sum(axis=1) == 1)
        if not valid.all():
            raise ValueError(f"Invalid one-hot labels in rows {start}:{stop}")
        labels = np.argmax(y, axis=1).astype(np.int64, copy=False)

        snr_positions = np.full(len(z), -1, dtype=np.int64)
        for snr, position in snr_to_position.items():
            snr_positions[z == snr] = position
        if np.any(snr_positions < 0):
            unknown = np.unique(z[snr_positions < 0]).tolist()
            raise ValueError(f"Unknown SNR values in rows {start}:{stop}: {unknown}")

        pair_ids = labels * len(EXPECTED_SNRS) + snr_positions
        raw_indices = np.arange(start, stop, dtype=np.uint32)
        for pair_id in np.unique(pair_ids):
            fragments[int(pair_id)].append(raw_indices[pair_ids == pair_id])

        if start == 0 or stop == total or stop % (chunk_size * 5) == 0:
            print(f"Metadata scan: {stop:,}/{total:,}")

    groups = [
        np.concatenate(parts) if parts else np.empty(0, dtype=np.uint32)
        for parts in fragments
    ]
    bad_sizes = {
        group_id: int(len(indices))
        for group_id, indices in enumerate(groups)
        if len(indices) != EXPECTED_GROUP_SIZE
    }
    if bad_sizes:
        raise ValueError(f"Unexpected class-SNR group sizes: {bad_sizes}")
    return groups


def stratified_split(
    groups: list[np.ndarray],
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Shuffle each stratum independently and assign exact integer counts."""
    split_parts: dict[str, list[np.ndarray]] = {name: [] for name in SPLIT_NAMES}
    counts = np.zeros(
        (len(SPLIT_NAMES), EXPECTED_CLASSES, len(EXPECTED_SNRS)), dtype=np.int64
    )

    for group_id, group_indices in enumerate(groups):
        # A group-specific SeedSequence makes each stratum reproducible and
        # independent of traversal order.
        rng = np.random.default_rng(np.random.SeedSequence([seed, group_id]))
        shuffled = rng.permutation(group_indices)
        cursor = 0
        class_id, snr_position = divmod(group_id, len(EXPECTED_SNRS))
        for split_position, name in enumerate(SPLIT_NAMES):
            count = SPLIT_PER_GROUP[name]
            selected = shuffled[cursor : cursor + count]
            split_parts[name].append(selected)
            counts[split_position, class_id, snr_position] = len(selected)
            cursor += count
        if cursor != len(group_indices):
            raise AssertionError(f"Group {group_id} was not fully assigned")

    splits = {
        name: np.sort(np.concatenate(parts)).astype(np.uint32, copy=False)
        for name, parts in split_parts.items()
    }
    return splits, counts


def verify_splits(splits: dict[str, np.ndarray]) -> dict[str, Any]:
    """Prove unique assignment, pairwise disjointness, and full coverage."""
    ownership = np.full(EXPECTED_SAMPLES, -1, dtype=np.int8)
    duplicate_counts: dict[str, int] = {}
    overlap_count = 0

    for split_id, name in enumerate(SPLIT_NAMES):
        indices = splits[name]
        if len(indices) == 0 or int(indices[-1]) >= EXPECTED_SAMPLES:
            raise ValueError(f"{name} contains an out-of-range or empty index set")
        duplicate_counts[name] = int(len(indices) - len(np.unique(indices)))
        already_owned = ownership[indices] != -1
        overlap_count += int(already_owned.sum())
        ownership[indices] = split_id

    missing_count = int((ownership == -1).sum())
    expected_sizes = {
        name: SPLIT_PER_GROUP[name] * EXPECTED_CLASSES * len(EXPECTED_SNRS)
        for name in SPLIT_NAMES
    }
    size_ok = all(len(splits[name]) == expected_sizes[name] for name in SPLIT_NAMES)
    passed = (
        size_ok
        and not any(duplicate_counts.values())
        and overlap_count == 0
        and missing_count == 0
    )
    return {
        "passed": passed,
        "size_ok": size_ok,
        "duplicate_counts": duplicate_counts,
        "cross_split_overlap_count": overlap_count,
        "unassigned_index_count": missing_count,
        "full_coverage": missing_count == 0,
    }


def save_splits(path: Path, splits: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **splits)
    os.replace(temporary, path)


def smoke_test_loader(
    hdf5_path: Path,
    splits_path: Path,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        dataset = RadioML2018Dataset(name, hdf5_path, splits_path)
        generator = torch.Generator().manual_seed(DEFAULT_SEED)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            generator=generator,
            persistent_workers=num_workers > 0,
        )
        started = time.perf_counter()
        batch = next(iter(loader))
        elapsed = time.perf_counter() - started
        shape_ok = tuple(batch["iq"].shape) == (batch_size, 2, 1024)
        dtype_ok = batch["iq"].dtype == torch.float32
        metadata_ok = all(
            len(batch[key]) == batch_size for key in ("label", "snr", "index")
        )
        snr_ok = set(batch["snr"].tolist()).issubset(set(EXPECTED_SNRS.tolist()))
        result[name] = {
            "passed": bool(shape_ok and dtype_ok and metadata_ok and snr_ok),
            "dataset_length": len(dataset),
            "batch_iq_shape": list(batch["iq"].shape),
            "batch_iq_dtype": str(batch["iq"].dtype),
            "batch_label_shape": list(batch["label"].shape),
            "batch_snr_shape": list(batch["snr"].shape),
            "batch_index_shape": list(batch["index"].shape),
            "first_batch_seconds": round(elapsed, 4),
        }
        dataset.close()
    return result


def write_report(manifest: dict[str, Any]) -> None:
    lines = [
        "# P1-T7 RML2018.01A split and loading report",
        "",
        f"- Generated: `{manifest['generated_at']}`",
        f"- Random seed: `{manifest['seed']}`",
        f"- Validation passed: `{manifest['validation']['passed']}`",
        f"- Loader smoke test passed: `{manifest['loader_smoke_test']['passed']}`",
        "",
        "## Fixed split",
        "",
        "| Split | Samples per class-SNR group | Total samples | Actual ratio |",
        "|---|---:|---:|---:|",
    ]
    for name in SPLIT_NAMES:
        item = manifest["splits"][name]
        lines.append(
            f"| {name} | {item['per_group']} | {item['sample_count']:,} | "
            f"{100 * item['ratio']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "Each of the 624 class-SNR strata is shuffled independently before "
            "assignment. All models must reuse this split file and seed.",
            "",
            "## Leakage and coverage checks",
            "",
            f"- Duplicate indices: `{manifest['validation']['duplicate_counts']}`",
            f"- Cross-split overlaps: "
            f"`{manifest['validation']['cross_split_overlap_count']}`",
            f"- Unassigned rows: `{manifest['validation']['unassigned_index_count']}`",
            f"- Full coverage: `{manifest['validation']['full_coverage']}`",
            "",
            "These checks establish index-level separation. RML2018.01A does not "
            "provide transmitter/session identity, so session-level leakage cannot "
            "be tested and must not be claimed.",
            "",
            "## DataLoader smoke test",
            "",
        ]
    )
    for name in SPLIT_NAMES:
        item = manifest["loader_smoke_test"][name]
        lines.append(
            f"- {name}: `{item['batch_iq_shape']}` `{item['batch_iq_dtype']}`, "
            f"first batch `{item['first_batch_seconds']:.4f}s`, passed `{item['passed']}`"
        )
    lines.extend(
        [
            "",
            "The loader opens HDF5 lazily per process and returns IQ as `(2, 1024)`. "
            "Normalization and augmentation are intentionally deferred to P1-T8.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    hdf5_path = args.hdf5.expanduser().resolve()
    classes_path = args.classes.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    if not classes_path.is_file():
        raise FileNotFoundError(f"Class-order file not found: {classes_path}")
    classes = load_classes(classes_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with h5py.File(hdf5_path, mode="r") as file:
        missing = [key for key in ("X", "Y", "Z") if key not in file]
        if missing:
            raise KeyError(f"Missing HDF5 datasets: {missing}")
        groups = collect_group_indices(file, args.chunk_size)

    splits, distribution = stratified_split(groups, args.seed)
    validation = verify_splits(splits)
    if not validation["passed"]:
        raise RuntimeError(f"Split validation failed: {validation}")
    save_splits(output_path, splits)

    loader_results = smoke_test_loader(
        hdf5_path,
        output_path,
        args.batch_size,
        args.num_workers,
    )
    loader_passed = all(item["passed"] for item in loader_results.values())
    if not loader_passed:
        raise RuntimeError(f"DataLoader smoke test failed: {loader_results}")

    manifest: dict[str, Any] = {
        "generated_at": now_iso(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "dataset_path": str(hdf5_path.relative_to(PROJECT_ROOT)),
        "classes_path": str(classes_path.relative_to(PROJECT_ROOT)),
        "split_file": str(output_path.relative_to(PROJECT_ROOT)),
        "split_file_sha256": sha256_file(output_path),
        "seed": args.seed,
        "strategy": "independent random permutation within each class-SNR stratum",
        "classes": classes,
        "snrs_db": EXPECTED_SNRS.astype(int).tolist(),
        "stratum_count": EXPECTED_CLASSES * len(EXPECTED_SNRS),
        "source_samples_per_stratum": EXPECTED_GROUP_SIZE,
        "splits": {
            name: {
                "per_group": SPLIT_PER_GROUP[name],
                "sample_count": int(len(splits[name])),
                "ratio": len(splits[name]) / EXPECTED_SAMPLES,
                "indices_sha256": hashlib.sha256(splits[name].tobytes()).hexdigest(),
                "group_count_min": int(distribution[position].min()),
                "group_count_max": int(distribution[position].max()),
            }
            for position, name in enumerate(SPLIT_NAMES)
        },
        "validation": validation,
        "leakage_scope": (
            "Index-level only; transmitter/session identifiers are unavailable "
            "in RML2018.01A."
        ),
        "loader_smoke_test": {
            "passed": loader_passed,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            **loader_results,
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(manifest)

    print(f"Split file:      {output_path}")
    print(f"Manifest:        {MANIFEST_PATH}")
    print(f"Markdown report: {REPORT_PATH}")
    print(f"Validation:      {validation}")
    print(f"Loader passed:   {loader_passed}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed class-SNR-stratified RML2018.01A splits."
    )
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
