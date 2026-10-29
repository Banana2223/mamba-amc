"""Verify the P2-T4 RadioResNet with a class-balanced overfit check."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from amc_framework.data import resolve_project_path
from amc_framework.models import create_model
from amc_framework.verification import load_fixed_balanced_batch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HDF5 = PROJECT_ROOT / "data/raw/RML2018.01a/GOLD_XYZ_OSC.0001_1024.hdf5"
DEFAULT_SPLITS = PROJECT_ROOT / "artifacts/P1-T7/split_indices.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/P2-T4/resnet_verification.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shape, gradient, and fixed-small-set ResNet checks."
    )
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    overfit = result["overfit"]
    lines = [
        "# P2-T4 RadioResNet 工程验收报告",
        "",
        f"- 验收时间：`{result['generated_at']}`",
        f"- 验收结论：`{'通过' if result['passed'] else '未通过'}`",
        "- 验收性质：结构与训练路径验证，不是正式基准结果",
        "- 测试集状态：未读取、未评价",
        "",
        "## 验收结果",
        "",
        "| 检查项 | 结果 |",
        "|---|---|",
        f"| 输入形状 | `{result['input_shape']}` |",
        f"| 输出形状 | `{result['output_shape']}` |",
        f"| 模型参数量 | `{result['parameter_count']:,}` |",
        f"| 梯度有限且完整 | `{result['gradient_check_passed']}` |",
        f"| 固定小样本数 | `{overfit['sample_count']}`（24类均衡） |",
        f"| 初始损失 | `{overfit['initial_loss']:.6f}` |",
        f"| 最终损失 | `{overfit['final_loss']:.6f}` |",
        f"| 最终训练准确率 | `{overfit['final_accuracy']:.4f}` |",
        f"| 目标训练准确率 | `>= {overfit['target_accuracy']:.2f}` |",
        f"| 小样本过拟合检查 | `{overfit['passed']}` |",
        "",
        "该检查只证明模型、残差连接、损失和优化路径能够学习固定样本，不能代表泛化性能。",
        "正式训练仍须使用完整训练集和验证集，并继续封存测试集。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 < args.target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be in (0, 1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hdf5_path = resolve_project_path(args.hdf5)
    splits_path = resolve_project_path(args.splits)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    iq, labels, raw_indices = load_fixed_balanced_batch(
        hdf5_path,
        splits_path,
        samples_per_class=args.samples_per_class,
    )
    iq = iq.to(device)
    labels = labels.to(device)

    model = create_model(
        "radio_resnet",
        num_classes=24,
        input_length=1024,
        channels=32,
        num_stacks=6,
        units_per_stack=2,
        kernel_size=3,
        classifier_width=128,
        dropout=0.0,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    started = time.perf_counter()
    with torch.no_grad():
        initial_logits = model(iq)
        initial_loss = float(criterion(initial_logits, labels).item())

    final_loss = initial_loss
    final_accuracy = 0.0
    gradients_ok = False
    steps_completed = 0
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(iq)
        loss = criterion(logits, labels)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        gradients_ok = all(
            gradient is not None and torch.isfinite(gradient).all().item()
            for gradient in gradients
        )
        if not gradients_ok:
            raise RuntimeError(f"Invalid gradient detected at step {step + 1}")
        optimizer.step()
        steps_completed = step + 1
        with torch.no_grad():
            final_loss = float(loss.item())
            final_accuracy = float(
                (logits.argmax(dim=1) == labels).float().mean().item()
            )
        if final_accuracy >= args.target_accuracy and final_loss < initial_loss:
            break

    passed = bool(
        tuple(initial_logits.shape) == (len(labels), 24)
        and gradients_ok
        and np.isfinite(final_loss)
        and final_loss < initial_loss
        and final_accuracy >= args.target_accuracy
    )
    result: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": passed,
        "device": str(device),
        "model_name": "radio_resnet",
        "parameter_count": parameter_count,
        "architecture": {
            "input": [2, 1024],
            "channels": 32,
            "residual_stacks": 6,
            "residual_units_per_stack": 2,
            "classifier_width": 128,
            "classes": 24,
        },
        "input_shape": list(iq.shape),
        "output_shape": list(initial_logits.shape),
        "gradient_check_passed": gradients_ok,
        "test_split_evaluated": False,
        "overfit": {
            "sample_count": len(labels),
            "samples_per_class": args.samples_per_class,
            "class_count": len(torch.unique(labels)),
            "raw_indices": raw_indices,
            "steps_requested": args.steps,
            "steps_completed": steps_completed,
            "learning_rate": args.learning_rate,
            "target_accuracy": args.target_accuracy,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "final_accuracy": final_accuracy,
            "passed": passed,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(output_path.with_suffix(".md"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
