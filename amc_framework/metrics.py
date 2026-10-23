"""Streaming AMC metrics that do not retain model outputs in memory."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))


class ClassificationAccumulator:
    """Accumulate confusion, per-SNR accuracy, and loss incrementally."""

    def __init__(
        self,
        num_classes: int,
        low_snr_min_db: int,
        low_snr_max_db: int,
    ) -> None:
        self.num_classes = num_classes
        self.low_snr_min_db = low_snr_min_db
        self.low_snr_max_db = low_snr_max_db
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.snr_correct: dict[int, int] = defaultdict(int)
        self.snr_total: dict[int, int] = defaultdict(int)
        self.loss_sum = 0.0
        self.sample_count = 0

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        snrs: torch.Tensor,
        loss: float,
    ) -> None:
        predictions = logits.detach().argmax(dim=1).cpu().numpy().astype(np.int64)
        true_labels = labels.detach().cpu().numpy().astype(np.int64)
        snr_values = snrs.detach().cpu().numpy().astype(np.int64)
        if not (
            len(predictions) == len(true_labels) == len(snr_values)
        ):
            raise ValueError("logits, labels, and snrs have different batch sizes")
        if np.any(true_labels < 0) or np.any(true_labels >= self.num_classes):
            raise ValueError("labels contain an out-of-range class index")
        if np.any(predictions < 0) or np.any(predictions >= self.num_classes):
            raise ValueError("predictions contain an out-of-range class index")

        pair_ids = true_labels * self.num_classes + predictions
        self.confusion += np.bincount(
            pair_ids, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
        correct = predictions == true_labels
        for snr in np.unique(snr_values):
            selected = snr_values == snr
            self.snr_total[int(snr)] += int(selected.sum())
            self.snr_correct[int(snr)] += int(correct[selected].sum())
        batch_size = len(true_labels)
        self.loss_sum += float(loss) * batch_size
        self.sample_count += batch_size

    def compute(self) -> dict[str, Any]:
        if self.sample_count == 0:
            raise RuntimeError("Cannot compute metrics without samples")
        true_positive = np.diag(self.confusion).astype(np.float64)
        actual = self.confusion.sum(axis=1).astype(np.float64)
        predicted = self.confusion.sum(axis=0).astype(np.float64)
        recall = np.divide(
            true_positive,
            actual,
            out=np.zeros_like(true_positive),
            where=actual > 0,
        )
        precision = np.divide(
            true_positive,
            predicted,
            out=np.zeros_like(true_positive),
            where=predicted > 0,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(true_positive),
            where=(precision + recall) > 0,
        )
        per_snr_accuracy = {
            str(snr): self.snr_correct[snr] / self.snr_total[snr]
            for snr in sorted(self.snr_total)
        }
        low_snrs = [
            snr
            for snr in self.snr_total
            if self.low_snr_min_db <= snr <= self.low_snr_max_db
        ]
        low_total = sum(self.snr_total[snr] for snr in low_snrs)
        low_correct = sum(self.snr_correct[snr] for snr in low_snrs)
        return {
            "loss": self.loss_sum / self.sample_count,
            "accuracy": float(true_positive.sum() / self.sample_count),
            "macro_f1": float(f1.mean()),
            "per_class_recall": recall.tolist(),
            "per_snr_accuracy": per_snr_accuracy,
            "low_snr_accuracy": (
                float(low_correct / low_total) if low_total > 0 else None
            ),
            "low_snr_range_db": [self.low_snr_min_db, self.low_snr_max_db],
            "sample_count": self.sample_count,
            "confusion_matrix": self.confusion.tolist(),
        }


def save_evaluation_plots(
    metrics: dict[str, Any],
    class_names: list[str],
    output_dir: str | Path,
    prefix: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    row_sum = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_sum,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_sum > 0,
    )

    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    axis.set_title(f"{prefix} normalized confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
    axis.set_yticks(range(len(class_names)), class_names, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    confusion_path = destination / f"{prefix}_confusion_matrix.png"
    figure.savefig(confusion_path, dpi=160)
    plt.close(figure)

    snr_items = sorted(
        (int(snr), float(value))
        for snr, value in metrics["per_snr_accuracy"].items()
    )
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [item[0] for item in snr_items],
        [item[1] for item in snr_items],
        marker="o",
    )
    axis.set_title(f"{prefix} accuracy by SNR")
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    snr_path = destination / f"{prefix}_accuracy_by_snr.png"
    figure.savefig(snr_path, dpi=160)
    plt.close(figure)
    return [confusion_path, snr_path]

