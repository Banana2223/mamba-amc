"""Aggregate the three P2-T3 formal validation runs without touching test data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs" / "P2-T3"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "P2-T3"
DEFAULT_CLASSES = PROJECT_ROOT / "data" / "raw" / "RML2018.01a" / "classes-fixed.json"
DEFAULT_SEEDS = (2026, 2027, 2028)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize P2-T3 validation metrics across formal seeds."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return values


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "sample_std": float(stdev(values)) if len(values) > 1 else 0.0,
    }


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only fields that must differ between repeated-seed runs."""
    normalized = copy.deepcopy(config)
    normalized["training"]["seed"] = "<training-seed>"
    normalized["output"]["run_name"] = "<run-name>"
    return normalized


def protocol_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_snr_plot(
    output: Path,
    per_seed: list[dict[str, Any]],
    aggregate: dict[str, dict[str, float]],
) -> None:
    snrs = sorted(int(value) for value in aggregate)
    means = [aggregate[str(snr)]["mean"] for snr in snrs]
    stds = [aggregate[str(snr)]["sample_std"] for snr in snrs]
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for run in per_seed:
        values = run["validation"]["per_snr_accuracy"]
        axis.plot(
            snrs,
            [values[str(snr)] for snr in snrs],
            alpha=0.35,
            linewidth=1.1,
            label=f"seed {run['seed']}",
        )
    axis.plot(snrs, means, color="black", linewidth=2.2, label="mean")
    axis.fill_between(
        snrs,
        np.asarray(means) - np.asarray(stds),
        np.asarray(means) + np.asarray(stds),
        color="black",
        alpha=0.14,
        label="mean +/- sample std",
    )
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Validation accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(snrs[::2])
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_learning_curves(output: Path, per_seed: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for run in per_seed:
        epochs = [record["epoch"] + 1 for record in run["epoch_records"]]
        axes[0].plot(
            epochs,
            [record["train"]["loss"] for record in run["epoch_records"]],
            linestyle="--",
            alpha=0.65,
            label=f"train {run['seed']}",
        )
        axes[0].plot(
            epochs,
            [record["validation"]["loss"] for record in run["epoch_records"]],
            label=f"val {run['seed']}",
        )
        axes[1].plot(
            epochs,
            [record["train"]["macro_f1"] for record in run["epoch_records"]],
            linestyle="--",
            alpha=0.65,
            label=f"train {run['seed']}",
        )
        axes[1].plot(
            epochs,
            [record["validation"]["macro_f1"] for record in run["epoch_records"]],
            label=f"val {run['seed']}",
        )
    axes[0].set_title("Loss")
    axes[1].set_title("Macro-F1")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("Value")
    axes[1].set_ylim(0.0, 1.0)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_class_recall_plot(
    output: Path,
    classes: list[str],
    aggregate: dict[str, dict[str, float]],
) -> None:
    means = [aggregate[name]["mean"] for name in classes]
    stds = [aggregate[name]["sample_std"] for name in classes]
    positions = np.arange(len(classes))
    figure, axis = plt.subplots(figsize=(12, 5.6))
    axis.bar(positions, means, yerr=stds, capsize=2, color="#4C78A8")
    axis.set_xticks(positions, classes, rotation=55, ha="right")
    axis.set_ylabel("Validation recall")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_confusion_plot(
    output: Path,
    classes: list[str],
    confusion: np.ndarray,
) -> None:
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_totals,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_totals != 0,
    )
    figure, axis = plt.subplots(figsize=(10.5, 9.2))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(classes)), classes, rotation=70, ha="right", fontsize=7)
    axis.set_yticks(np.arange(len(classes)), classes, fontsize=7)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Aggregated normalized validation confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(path: Path, result: dict[str, Any]) -> None:
    aggregate = result["aggregate"]
    lines = [
        "# P2-T3 轻量 1D CNN 三种子验证结果",
        "",
        f"- 汇总时间：`{result['generated_at']}`",
        f"- 随机种子：`{result['seeds']}`",
        f"- 可比性检查：`{'通过' if result['protocol_consistent'] else '未通过'}`",
        f"- 测试集状态：`{'未评价' if not result['test_split_evaluated'] else '已评价'}`",
        "- 标准差口径：三次运行的样本标准差（n-1）",
        "",
        "## 一、核心验证指标",
        "",
        "| 指标 | 均值 | 标准差 |",
        "|---|---:|---:|",
    ]
    labels = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro-F1",
        "low_snr_accuracy": "低 SNR Accuracy（-20～0 dB）",
        "loss": "Loss",
    }
    for key, label in labels.items():
        item = aggregate[key]
        lines.append(f"| {label} | {item['mean']:.6f} | {item['sample_std']:.6f} |")

    lines.extend(
        [
            "",
            "## 二、逐种子结果",
            "",
            "| Seed | 最佳 epoch | 实际 epoch | Accuracy | Macro-F1 | 低 SNR Accuracy | 早停 |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for run in result["runs"]:
        validation = run["validation"]
        lines.append(
            f"| {run['seed']} | {run['best_epoch']} | {run['epochs_completed']} | "
            f"{validation['accuracy']:.6f} | {validation['macro_f1']:.6f} | "
            f"{validation['low_snr_accuracy']:.6f} | {run['stopped_early']} |"
        )

    lines.extend(
        [
            "",
            "## 三、运行效率",
            "",
            "| 指标 | 均值 | 标准差 |",
            "|---|---:|---:|",
            f"| 总训练时间（分钟） | {aggregate['training_minutes']['mean']:.3f} | {aggregate['training_minutes']['sample_std']:.3f} |",
            f"| 峰值显存（MiB） | {aggregate['peak_memory_mib']['mean']:.3f} | {aggregate['peak_memory_mib']['sample_std']:.3f} |",
            f"| 验证吞吐率（样本/秒） | {aggregate['validation_samples_per_second']['mean']:.3f} | {aggregate['validation_samples_per_second']['sample_std']:.3f} |",
            f"| 验证单样本时延（毫秒） | {aggregate['validation_milliseconds_per_sample']['mean']:.6f} | {aggregate['validation_milliseconds_per_sample']['sample_std']:.6f} |",
            "",
            "## 四、分 SNR 验证准确率",
            "",
            "| SNR（dB） | 均值 | 标准差 |",
            "|---:|---:|---:|",
        ]
    )
    for snr in sorted(result["per_snr"], key=int):
        item = result["per_snr"][snr]
        lines.append(f"| {snr} | {item['mean']:.6f} | {item['sample_std']:.6f} |")

    lines.extend(
        [
            "",
            "## 五、逐类别召回率",
            "",
            "| 类别 | 均值 | 标准差 |",
            "|---|---:|---:|",
        ]
    )
    recalls = result["per_class_recall"]
    for name, item in sorted(recalls.items(), key=lambda pair: pair[1]["mean"]):
        lines.append(f"| {name} | {item['mean']:.6f} | {item['sample_std']:.6f} |")

    lines.extend(
        [
            "",
            "## 六、主要类别混淆",
            "",
            "| 真实类别 | 预测类别 | 三次合计样本数 | 占真实类别比例 |",
            "|---|---|---:|---:|",
        ]
    )
    for item in result["top_confusions"]:
        lines.append(
            f"| {item['true_class']} | {item['predicted_class']} | "
            f"{item['count']:,} | {item['rate']:.4%} |"
        )

    lines.extend(
        [
            "",
            "## 七、结论与边界",
            "",
            "- 三次运行采用相同数据划分、模型结构和公共训练协议，仅训练随机种子及运行名称不同。",
            "- 三次运行均由验证集 Macro-F1 早停，说明模型在当前结构和协议下已进入稳定平台区。",
            "- 该报告只汇总验证集；在模型与配置不再调整前，不应读取测试集。",
            "- 原始运行目录受 `.gitignore` 保护，本文档和对应图表用于版本化保存关键结果。",
            "",
            "## 八、图表",
            "",
            "- `validation_accuracy_by_snr_mean.png`：三种子分 SNR 曲线与均值±标准差；",
            "- `validation_learning_curves.png`：三种子训练/验证 Loss 和 Macro-F1；",
            "- `validation_per_class_recall.png`：逐类别召回率均值±标准差；",
            "- `validation_confusion_matrix_aggregate.png`：三种子聚合并按真实类别归一化的混淆矩阵。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    classes_path = args.classes.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError("Class file must contain a JSON string list")

    runs: list[dict[str, Any]] = []
    comparable_configs: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_dir = run_root / f"cnn-seed-{seed}"
        metadata = load_json(run_dir / "run_config.json")
        summary = load_json(run_dir / "summary.json")
        validation = load_json(run_dir / "best_validation_metrics.json")
        records = load_jsonl(run_dir / "metrics.jsonl")
        config = metadata["config"]
        recorded_seed = int(config["training"]["seed"])
        if recorded_seed != seed:
            raise ValueError(
                f"Seed mismatch for {run_dir}: expected {seed}, got {recorded_seed}"
            )
        if (run_dir / "test_metrics.json").exists():
            raise RuntimeError(
                f"Test metrics already exist in {run_dir}; this validation-only "
                "summary intentionally refuses to mix validation and test results."
            )
        comparable_configs.append(comparable_config(config))
        runs.append(
            {
                "seed": seed,
                "run_directory": str(run_dir.relative_to(PROJECT_ROOT)),
                "config_sha256": metadata["config_sha256"],
                "best_epoch": int(summary["best_epoch"]) + 1,
                "epochs_completed": int(summary["last_epoch"]) + 1,
                "stopped_early": bool(summary["stopped_early"]),
                "training_seconds": float(summary["runtime"]["elapsed_seconds"]),
                "peak_memory_mib": float(summary["runtime"]["peak_memory_mib"]),
                "validation": validation,
                "epoch_records": records,
            }
        )

    reference = comparable_configs[0]
    protocol_consistent = all(config == reference for config in comparable_configs[1:])
    if not protocol_consistent:
        raise ValueError("Formal run configurations differ beyond seed and run name")

    scalar_keys = ("loss", "accuracy", "macro_f1", "low_snr_accuracy")
    aggregate = {
        key: mean_std([float(run["validation"][key]) for run in runs])
        for key in scalar_keys
    }
    aggregate.update(
        {
            "training_minutes": mean_std(
                [run["training_seconds"] / 60.0 for run in runs]
            ),
            "peak_memory_mib": mean_std(
                [run["peak_memory_mib"] for run in runs]
            ),
            "validation_samples_per_second": mean_std(
                [
                    float(run["validation"]["samples_per_second"])
                    for run in runs
                ]
            ),
            "validation_milliseconds_per_sample": mean_std(
                [
                    float(run["validation"]["end_to_end_milliseconds_per_sample"])
                    for run in runs
                ]
            ),
        }
    )

    snr_keys = sorted(runs[0]["validation"]["per_snr_accuracy"], key=int)
    per_snr = {
        snr: mean_std(
            [float(run["validation"]["per_snr_accuracy"][snr]) for run in runs]
        )
        for snr in snr_keys
    }
    per_class_recall = {
        name: mean_std(
            [float(run["validation"]["per_class_recall"][index]) for run in runs]
        )
        for index, name in enumerate(classes)
    }

    confusion = np.sum(
        [np.asarray(run["validation"]["confusion_matrix"], dtype=np.int64) for run in runs],
        axis=0,
    )
    row_totals = confusion.sum(axis=1)
    confusion_items: list[dict[str, Any]] = []
    for true_index, row in enumerate(confusion):
        for predicted_index, count in enumerate(row):
            if true_index == predicted_index or count == 0:
                continue
            confusion_items.append(
                {
                    "true_class": classes[true_index],
                    "predicted_class": classes[predicted_index],
                    "count": int(count),
                    "rate": float(count / row_totals[true_index]),
                }
            )
    top_confusions = sorted(
        confusion_items, key=lambda item: item["count"], reverse=True
    )[:15]

    public_runs = []
    for run in runs:
        public_run = {key: value for key, value in run.items() if key != "epoch_records"}
        public_runs.append(public_run)
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "three-seed formal validation summary",
        "seeds": args.seeds,
        "protocol_consistent": protocol_consistent,
        "normalized_protocol_sha256": protocol_hash(reference),
        "test_split_evaluated": False,
        "standard_deviation": "sample standard deviation (n-1)",
        "model": {
            "name": "lightweight_cnn",
            "parameters": 291128,
        },
        "runs": public_runs,
        "aggregate": aggregate,
        "per_snr": per_snr,
        "per_class_recall": per_class_recall,
        "top_confusions": top_confusions,
    }

    json_path = output / "formal_validation_summary.json"
    report_path = output / "formal_validation_report.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(report_path, result)
    save_snr_plot(output / "validation_accuracy_by_snr_mean.png", runs, per_snr)
    save_learning_curves(output / "validation_learning_curves.png", runs)
    save_class_recall_plot(
        output / "validation_per_class_recall.png", classes, per_class_recall
    )
    save_confusion_plot(
        output / "validation_confusion_matrix_aggregate.png", classes, confusion
    )

    print(f"Protocol consistent: {protocol_consistent}")
    print(f"Validation Accuracy: {aggregate['accuracy']['mean']:.6f} "
          f"+/- {aggregate['accuracy']['sample_std']:.6f}")
    print(f"Validation Macro-F1: {aggregate['macro_f1']['mean']:.6f} "
          f"+/- {aggregate['macro_f1']['sample_std']:.6f}")
    print(f"Low-SNR Accuracy: {aggregate['low_snr_accuracy']['mean']:.6f} "
          f"+/- {aggregate['low_snr_accuracy']['sample_std']:.6f}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
