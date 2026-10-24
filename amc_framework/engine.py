"""Reusable training, validation, checkpoint, resume, and reporting engine."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig
from .data import PROJECT_ROOT, DataBundle, build_data_bundle, resolve_project_path
from .metrics import ClassificationAccumulator, save_evaluation_plots
from .models import create_model


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def set_reproducible_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}


def build_optimizer(
    model: nn.Module, config: ExperimentConfig
) -> torch.optim.Optimizer:
    values = config.optimizer
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = values.name.lower()
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=values.learning_rate,
            weight_decay=values.weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=values.learning_rate,
            weight_decay=values.weight_decay,
        )
    return torch.optim.SGD(
        parameters,
        lr=values.learning_rate,
        momentum=values.momentum,
        weight_decay=values.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    values = config.scheduler
    name = values.name.lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.max_epochs,
            eta_min=values.minimum_learning_rate,
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.training.checkpoint_mode,
        factor=values.factor,
        patience=values.patience,
        min_lr=values.minimum_learning_rate,
    )


def _make_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def atomic_json_write(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(values, temporary)
    os.replace(temporary, path)


class ExperimentRunner:
    """Run one baseline under the project's fixed data/evaluation protocol."""

    def __init__(
        self,
        config: ExperimentConfig,
        resume_path: str | Path | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.resume_path = (
            Path(resume_path).expanduser().resolve() if resume_path else None
        )
        self.device = resolve_device(config.training.device)
        set_reproducible_seed(
            config.training.seed, config.training.deterministic
        )
        self.output_dir = self._resolve_output_dir()
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.plot_dir = self.output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.data: DataBundle = build_data_bundle(
            config.data,
            seed=config.training.seed,
            num_classes=config.evaluation.num_classes,
        )
        model_kwargs = dict(config.model.kwargs)
        model_kwargs.setdefault("num_classes", config.evaluation.num_classes)
        self.model = create_model(config.model.name, **model_kwargs).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        self.amp_enabled = config.training.amp and self.device.type == "cuda"
        self.scaler = _make_grad_scaler(self.amp_enabled)
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = (
            math.inf if config.training.checkpoint_mode == "min" else -math.inf
        )
        self.best_epoch: int | None = None
        self.bad_epochs = 0
        self.resumed_from: str | None = None
        self.metrics_path = self.output_dir / "metrics.jsonl"

        self._write_run_metadata()
        if self.resume_path is not None:
            self._restore_checkpoint(self.resume_path)

    def _resolve_output_dir(self) -> Path:
        if self.resume_path is not None:
            if not self.resume_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {self.resume_path}")
            if self.resume_path.parent.name == "checkpoints":
                return self.resume_path.parent.parent
            return self.resume_path.parent
        root = resolve_project_path(self.config.output.root)
        run_name = self.config.output.run_name
        if run_name is None:
            timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
            run_name = f"{self.config.experiment_name}-{timestamp}"
        output_dir = root / run_name
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Run directory is not empty: {output_dir}. "
                "Choose another output.run_name or resume a checkpoint."
            )
        return output_dir

    def _write_run_metadata(self) -> None:
        config_values = self.config.to_dict()
        encoded = json.dumps(
            config_values, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        metadata = {
            "created_at": now_iso(),
            "config_sha256": hashlib.sha256(encoded).hexdigest(),
            "config": config_values,
            "runtime": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "amp_requested": self.config.training.amp,
                "amp_enabled": self.amp_enabled,
            },
        }
        target = (
            self.output_dir / "resume_config.json"
            if self.resume_path is not None
            else self.output_dir / "run_config.json"
        )
        atomic_json_write(target, metadata)

    def _autocast_context(self) -> Any:
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _planned_batches(self, loader: Any, maximum: int | None) -> int:
        available = len(loader)
        return min(available, maximum) if maximum is not None else available

    def _move_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        non_blocking = self.device.type == "cuda"
        return (
            batch["iq"].to(self.device, non_blocking=non_blocking),
            batch["label"].to(self.device, non_blocking=non_blocking),
            batch["snr"].to(self.device, non_blocking=non_blocking),
        )

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _run_training_epoch(self) -> dict[str, Any]:
        self.model.train()
        accumulator = ClassificationAccumulator(
            self.config.evaluation.num_classes,
            self.config.evaluation.low_snr_min_db,
            self.config.evaluation.low_snr_max_db,
        )
        accumulation_steps = self.config.training.gradient_accumulation_steps
        planned_batches = self._planned_batches(
            self.data.train_loader, self.config.training.max_train_batches
        )
        self.optimizer.zero_grad(set_to_none=True)
        self._synchronize_device()
        started = time.perf_counter()

        for batch_index, batch in enumerate(self.data.train_loader):
            if batch_index >= planned_batches:
                break
            iq, labels, snrs = self._move_batch(batch)
            with self._autocast_context():
                logits = self.model(iq)
                raw_loss = self.criterion(logits, labels)
                loss = raw_loss / accumulation_steps
            if not torch.isfinite(raw_loss):
                raise RuntimeError(f"Non-finite training loss: {raw_loss.item()}")
            self.scaler.scale(loss).backward()

            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == planned_batches
            )
            if should_step:
                if self.config.training.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            accumulator.update(logits, labels, snrs, float(raw_loss.item()))

        self._synchronize_device()
        metrics = accumulator.compute()
        elapsed = time.perf_counter() - started
        metrics["elapsed_seconds"] = elapsed
        metrics["samples_per_second"] = metrics["sample_count"] / elapsed
        metrics["end_to_end_milliseconds_per_sample"] = (
            1000.0 * elapsed / metrics["sample_count"]
        )
        metrics["optimizer_steps"] = self.global_step
        return metrics

    @torch.inference_mode()
    def evaluate_loader(
        self,
        loader: Any,
        maximum_batches: int | None,
    ) -> dict[str, Any]:
        self.model.eval()
        accumulator = ClassificationAccumulator(
            self.config.evaluation.num_classes,
            self.config.evaluation.low_snr_min_db,
            self.config.evaluation.low_snr_max_db,
        )
        planned_batches = self._planned_batches(loader, maximum_batches)
        self._synchronize_device()
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            if batch_index >= planned_batches:
                break
            iq, labels, snrs = self._move_batch(batch)
            with self._autocast_context():
                logits = self.model(iq)
                loss = self.criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite evaluation loss: {loss.item()}")
            accumulator.update(logits, labels, snrs, float(loss.item()))
        self._synchronize_device()
        metrics = accumulator.compute()
        elapsed = time.perf_counter() - started
        metrics["elapsed_seconds"] = elapsed
        metrics["samples_per_second"] = metrics["sample_count"] / elapsed
        metrics["end_to_end_milliseconds_per_sample"] = (
            1000.0 * elapsed / metrics["sample_count"]
        )
        return metrics

    def _metric_improved(self, current: float) -> bool:
        delta = self.config.training.early_stopping_min_delta
        if self.config.training.checkpoint_mode == "max":
            return current > self.best_metric + delta
        return current < self.best_metric - delta

    def _step_scheduler(self, validation_metric: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(validation_metric)
        else:
            self.scheduler.step()

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "saved_at": now_iso(),
            "epoch": epoch,
            "global_step": self.global_step,
            "model_name": self.config.model.name,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "train_generator_state": self.data.train_generator.get_state(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "config": self.config.to_dict(),
        }

    def _restore_checkpoint(self, path: Path) -> None:
        checkpoint = _torch_load(path, self.device)
        if checkpoint.get("model_name") != self.config.model.name:
            raise ValueError(
                "Checkpoint model does not match configuration: "
                f"{checkpoint.get('model_name')} != {self.config.model.name}"
            )
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler is not None and checkpoint.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.best_metric = float(checkpoint["best_metric"])
        self.best_epoch = checkpoint.get("best_epoch")
        self.bad_epochs = int(checkpoint.get("bad_epochs", 0))
        if checkpoint.get("train_generator_state") is not None:
            # ``torch.load(map_location=cuda)`` also moves serialized CPU RNG
            # states to CUDA. CPU generators only accept a CPU ByteTensor.
            self.data.train_generator.set_state(
                checkpoint["train_generator_state"].cpu()
            )
        if checkpoint.get("python_random_state") is not None:
            random.setstate(checkpoint["python_random_state"])
        if checkpoint.get("numpy_random_state") is not None:
            np.random.set_state(checkpoint["numpy_random_state"])
        if checkpoint.get("torch_random_state") is not None:
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_random_state"]]
            )
        self.resumed_from = str(path)

    def _load_model_checkpoint(self, path: Path) -> None:
        checkpoint = _torch_load(path, self.device)
        self.model.load_state_dict(checkpoint["model_state"])

    def _append_epoch_log(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def fit(self) -> dict[str, Any]:
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if self.start_epoch >= self.config.training.max_epochs:
            raise ValueError(
                f"Checkpoint already reached epoch {self.start_epoch}; "
                f"max_epochs={self.config.training.max_epochs}"
            )

        stopped_early = False
        epochs_completed = 0
        for epoch in range(self.start_epoch, self.config.training.max_epochs):
            train_metrics = self._run_training_epoch()
            validation_metrics = self.evaluate_loader(
                self.data.validation_loader,
                self.config.training.max_validation_batches,
            )
            raw_metric = validation_metrics[
                self.config.training.checkpoint_metric
            ]
            if raw_metric is None:
                raise RuntimeError(
                    "The checkpoint metric is unavailable for the evaluated samples: "
                    f"{self.config.training.checkpoint_metric}"
                )
            current_metric = float(raw_metric)
            self._step_scheduler(current_metric)
            improved = self._metric_improved(current_metric)
            if improved:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1

            record = {
                "timestamp": now_iso(),
                "epoch": epoch,
                "global_step": self.global_step,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "improved": improved,
                "best_metric": self.best_metric,
                "bad_epochs": self.bad_epochs,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            self._append_epoch_log(record)
            checkpoint = self._checkpoint_payload(epoch)
            if improved:
                atomic_torch_save(self.checkpoint_dir / "best.pt", checkpoint)
            atomic_torch_save(self.checkpoint_dir / "last.pt", checkpoint)
            epochs_completed += 1

            print(
                f"epoch={epoch + 1}/{self.config.training.max_epochs} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={validation_metrics['loss']:.6f} "
                f"val_accuracy={validation_metrics['accuracy']:.4f} "
                f"val_macro_f1={validation_metrics['macro_f1']:.4f} "
                f"best={self.best_metric:.4f}"
            )
            patience = self.config.training.early_stopping_patience
            minimum_reached = epoch + 1 >= self.config.training.min_epochs
            if minimum_reached and patience > 0 and self.bad_epochs >= patience:
                stopped_early = True
                break

        best_path = self.checkpoint_dir / "best.pt"
        if best_path.is_file():
            self._load_model_checkpoint(best_path)
        best_validation = self.evaluate_loader(
            self.data.validation_loader,
            self.config.training.max_validation_batches,
        )
        atomic_json_write(
            self.output_dir / "best_validation_metrics.json", best_validation
        )
        save_evaluation_plots(
            best_validation,
            self.data.class_names,
            self.plot_dir,
            "validation",
        )

        elapsed = time.perf_counter() - started
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_memory_mib: float | None = (
                torch.cuda.max_memory_allocated(self.device) / 1024**2
            )
            device_name = torch.cuda.get_device_name(self.device)
        else:
            peak_memory_mib = None
            device_name = platform.processor() or "CPU"

        summary = {
            "completed_at": now_iso(),
            "purpose": "framework verification" if self.config.model.name == "tiny_probe" else "baseline training",
            "output_dir": str(self.output_dir),
            "resumed_from": self.resumed_from,
            "epochs_completed_this_invocation": epochs_completed,
            "last_epoch": self.start_epoch + epochs_completed - 1,
            "stopped_early": stopped_early,
            "checkpoint_metric": self.config.training.checkpoint_metric,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "best_validation": best_validation,
            "model": {
                "name": self.config.model.name,
                "parameters": count_parameters(self.model),
            },
            "data": {
                "train_samples": len(self.data.train_dataset),
                "validation_samples": len(self.data.validation_dataset),
                "test_samples": len(self.data.test_dataset),
                "batch_size": self.config.data.batch_size,
                "effective_batch_size": (
                    self.config.data.batch_size
                    * self.config.training.gradient_accumulation_steps
                ),
                "augmentation_enabled": self.config.data.train_augmentation,
            },
            "runtime": {
                "device": str(self.device),
                "device_name": device_name,
                "amp_enabled": self.amp_enabled,
                "elapsed_seconds": elapsed,
                "peak_memory_mib": peak_memory_mib,
            },
        }
        atomic_json_write(self.output_dir / "summary.json", summary)
        return summary

    def evaluate_test(self) -> dict[str, Any]:
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
        self._load_model_checkpoint(best_path)
        metrics = self.evaluate_loader(self.data.test_loader, maximum_batches=None)
        atomic_json_write(self.output_dir / "test_metrics.json", metrics)
        save_evaluation_plots(
            metrics,
            self.data.class_names,
            self.plot_dir,
            "test",
        )
        return metrics

    def close(self) -> None:
        self.data.close()

    def __enter__(self) -> "ExperimentRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

