"""P1-T9: minimal end-to-end training-chain verification.

This is an engineering smoke test, not a baseline experiment.  It connects the
fixed split, HDF5 on-demand loading, deterministic RMS preprocessing, a tiny
1-D CNN, cross-entropy loss, backpropagation, an optimizer update, and
validation inference.  No checkpoint or research accuracy is produced.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from iq_transforms import build_eval_transform, build_train_transform
from rml2018_dataset import DEFAULT_HDF5, DEFAULT_SPLITS, RadioML2018Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "P1-T9"
JSON_REPORT = OUTPUT_DIR / "minimal_chain_report.json"
MARKDOWN_REPORT = OUTPUT_DIR / "minimal_chain_report.md"

NUM_CLASSES = 24
SEQUENCE_LENGTH = 1024
DEFAULT_SEED = 2026


class TinyAMCProbe(nn.Module):
    """Small probe network used only to verify the training pipeline."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
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
        features = self.features(iq).squeeze(-1)
        return self.classifier(features)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def make_loader(
    split: str,
    hdf5_path: Path,
    splits_path: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
) -> tuple[RadioML2018Dataset, DataLoader[dict[str, torch.Tensor]]]:
    transform = (
        build_train_transform(enable_augmentation=False)
        if split == "train"
        else build_eval_transform()
    )
    dataset = RadioML2018Dataset(
        split,
        hdf5_path=hdf5_path,
        splits_path=splits_path,
        transform=transform,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )
    return dataset, loader


def get_first_train_batch(
    hdf5_path: Path,
    splits_path: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    dataset, loader = make_loader(
        "train",
        hdf5_path,
        splits_path,
        batch_size,
        num_workers,
        seed,
        shuffle=True,
    )
    batch = next(iter(loader))
    copied = {name: value.clone() for name, value in batch.items()}
    del loader
    dataset.close()
    return copied


def tensor_is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def batch_checks(batch: dict[str, torch.Tensor], batch_size: int) -> dict[str, bool]:
    iq = batch["iq"]
    labels = batch["label"]
    snrs = batch["snr"]
    indices = batch["index"]
    powers = torch.mean(iq[:, 0].square() + iq[:, 1].square(), dim=1)
    valid_snrs = set(range(-20, 32, 2))
    return {
        "iq_shape": tuple(iq.shape) == (batch_size, 2, SEQUENCE_LENGTH),
        "iq_dtype": iq.dtype == torch.float32,
        "iq_finite": tensor_is_finite(iq),
        "unit_complex_power": bool(
            torch.allclose(powers, torch.ones_like(powers), atol=1e-5)
        ),
        "label_shape": tuple(labels.shape) == (batch_size,),
        "label_dtype": labels.dtype == torch.int64,
        "label_range": bool(((labels >= 0) & (labels < NUM_CLASSES)).all()),
        "snr_shape": tuple(snrs.shape) == (batch_size,),
        "snr_values": set(snrs.tolist()).issubset(valid_snrs),
        "index_shape": tuple(indices.shape) == (batch_size,),
        "unique_indices": int(torch.unique(indices).numel()) == batch_size,
    }


def gradient_statistics(model: nn.Module) -> dict[str, Any]:
    squared_norm = 0.0
    tensor_count = 0
    all_finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        tensor_count += 1
        all_finite = all_finite and tensor_is_finite(parameter.grad)
        squared_norm += float(parameter.grad.detach().float().square().sum().item())
    return {
        "tensor_count": tensor_count,
        "global_l2_norm": math.sqrt(squared_norm),
        "all_finite": all_finite,
        "nonzero": squared_norm > 0.0,
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# P1-T9 minimal end-to-end chain report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Validation passed: `{report['passed']}`",
        f"- Device: `{report['runtime']['device']}`",
        f"- Probe model parameters: `{report['model']['parameter_count']:,}`",
        f"- Train steps: `{report['training']['steps']}`",
        f"- Total elapsed: `{report['runtime']['elapsed_seconds']:.4f}` seconds",
        "",
        "## Verified chain",
        "",
        "`fixed split -> HDF5 on-demand read -> RMS preprocessing -> DataLoader -> "
        "TinyAMCProbe -> cross-entropy -> backward -> optimizer update -> validation`",
        "",
        "## Checks",
        "",
    ]
    for name, value in report["checks"].items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(
        [
            "",
            "## Tensor contract",
            "",
            f"- Input batch: `{report['tensors']['input_shape']}` "
            f"`{report['tensors']['input_dtype']}`",
            f"- Labels: `{report['tensors']['label_shape']}` "
            f"`{report['tensors']['label_dtype']}`",
            f"- Logits: `{report['tensors']['logits_shape']}`",
            "",
            "## Training smoke values",
            "",
            f"- Losses: `{report['training']['losses']}`",
            f"- Last gradient L2 norm: "
            f"`{report['training']['last_gradient']['global_l2_norm']:.6f}`",
            f"- Validation loss on one batch: "
            f"`{report['validation']['one_batch_loss']:.6f}`",
            "",
            "These loss values are only software sanity checks. They are not model "
            "performance results and must not be used in the thesis comparison.",
            "",
        ]
    )
    if report["runtime"]["cuda_peak_memory_mib"] is not None:
        lines.append(
            f"- CUDA peak memory: "
            f"`{report['runtime']['cuda_peak_memory_mib']:.2f}` MiB"
        )
        lines.append("")
    MARKDOWN_REPORT.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    hdf5_path = args.hdf5.expanduser().resolve()
    splits_path = args.splits.expanduser().resolve()
    device = resolve_device(args.device)
    set_reproducible_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    first_batch = get_first_train_batch(
        hdf5_path, splits_path, args.batch_size, args.num_workers, args.seed
    )
    repeated_batch = get_first_train_batch(
        hdf5_path, splits_path, args.batch_size, args.num_workers, args.seed
    )
    reproducible_batch = all(
        torch.equal(first_batch[name], repeated_batch[name])
        for name in ("iq", "label", "snr", "index")
    )
    input_checks = batch_checks(first_batch, args.batch_size)

    train_dataset, train_loader = make_loader(
        "train",
        hdf5_path,
        splits_path,
        args.batch_size,
        args.num_workers,
        args.seed,
        shuffle=True,
    )
    val_dataset, val_loader = make_loader(
        "val",
        hdf5_path,
        splits_path,
        args.batch_size,
        args.num_workers,
        args.seed,
        shuffle=False,
    )

    model = TinyAMCProbe().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    tracked_parameter = next(model.parameters())
    parameter_before = tracked_parameter.detach().clone()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    losses: list[float] = []
    last_logits: torch.Tensor | None = None
    last_gradient: dict[str, Any] = {}
    train_iterator = iter(train_loader)
    for _ in range(args.steps):
        batch = next(train_iterator)
        iq = batch["iq"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(iq)
        loss = criterion(logits, labels)
        if not tensor_is_finite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")
        loss.backward()
        last_gradient = gradient_statistics(model)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        last_logits = logits.detach()

    parameter_changed = bool(
        not torch.equal(parameter_before, tracked_parameter.detach())
    )

    model.eval()
    with torch.inference_mode():
        val_batch = next(iter(val_loader))
        val_iq = val_batch["iq"].to(device, non_blocking=True)
        val_labels = val_batch["label"].to(device, non_blocking=True)
        val_logits = model(val_iq)
        val_loss = criterion(val_logits, val_labels)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        cuda_peak_memory_mib: float | None = (
            torch.cuda.max_memory_allocated(device) / 1024**2
        )
        device_name = torch.cuda.get_device_name(device)
    else:
        cuda_peak_memory_mib = None
        device_name = "CPU"

    assert last_logits is not None
    checks = {
        **{f"input_{name}": value for name, value in input_checks.items()},
        "same_seed_first_batch_reproducible": reproducible_batch,
        "logits_shape_is_batch_x_24": tuple(last_logits.shape)
        == (args.batch_size, NUM_CLASSES),
        "train_logits_finite": tensor_is_finite(last_logits),
        "all_train_losses_finite": all(math.isfinite(loss) for loss in losses),
        "gradient_tensors_exist": last_gradient["tensor_count"] > 0,
        "gradients_finite": bool(last_gradient["all_finite"]),
        "gradients_nonzero": bool(last_gradient["nonzero"]),
        "optimizer_changed_parameters": parameter_changed,
        "validation_logits_shape_is_batch_x_24": tuple(val_logits.shape)
        == (args.batch_size, NUM_CLASSES),
        "validation_logits_finite": tensor_is_finite(val_logits),
        "validation_loss_finite": tensor_is_finite(val_loss),
    }
    passed = all(checks.values())
    elapsed = time.perf_counter() - started

    report: dict[str, Any] = {
        "generated_at": now_iso(),
        "passed": passed,
        "purpose": "engineering smoke test; not a baseline result",
        "seed": args.seed,
        "data": {
            "hdf5_path": str(hdf5_path.relative_to(PROJECT_ROOT)),
            "splits_path": str(splits_path.relative_to(PROJECT_ROOT)),
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "augmentation_enabled": False,
            "preprocessing": "per-sample complex RMS normalization",
        },
        "model": {
            "name": "TinyAMCProbe",
            "role": "pipeline verification only",
            "parameter_count": parameter_count,
            "num_classes": NUM_CLASSES,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "losses": losses,
            "last_gradient": last_gradient,
        },
        "validation": {
            "batches": 1,
            "one_batch_loss": float(val_loss.item()),
        },
        "tensors": {
            "input_shape": list(first_batch["iq"].shape),
            "input_dtype": str(first_batch["iq"].dtype),
            "label_shape": list(first_batch["label"].shape),
            "label_dtype": str(first_batch["label"].dtype),
            "logits_shape": list(last_logits.shape),
        },
        "runtime": {
            "device": str(device),
            "device_name": device_name,
            "num_workers": args.num_workers,
            "elapsed_seconds": round(elapsed, 4),
            "cuda_peak_memory_mib": (
                round(cuda_peak_memory_mib, 3)
                if cuda_peak_memory_mib is not None
                else None
            ),
        },
        "checks": checks,
    }

    JSON_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(report)

    del train_loader, val_loader
    train_dataset.close()
    val_dataset.close()

    print(f"JSON report:     {JSON_REPORT}")
    print(f"Markdown report: {MARKDOWN_REPORT}")
    print(f"Device:          {device} ({device_name})")
    print(f"Train losses:    {[round(loss, 6) for loss in losses]}")
    print(f"Validation:      {'passed' if passed else 'failed'}")
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"P1-T9 validation failed: {failed}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the minimal RadioML training chain end to end."
    )
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
