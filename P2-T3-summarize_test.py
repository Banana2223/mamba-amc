"""Aggregate final P2-T3 test results across frozen formal runs."""

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
DEFAULT_CLASSES = PROJECT_ROOT / "data/raw/RML2018.01a/classes-fixed.json"
DEFAULT_SEEDS = (2026, 2027, 2028)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the frozen P2-T3 test results across seeds."
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "sample_std": float(stdev(values)) if len(values) > 1 else 0.0,
    }


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized["training"]["seed"] = "<training-seed>"
    normalized["output"]["run_name"] = "<run-name>"
    return normalized


def protocol_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_snr_plot(
    output: Path,
    runs: list[dict[str, Any]],
    aggregate: dict[str, dict[str, float]],
) -> None:
    snrs = sorted(int(value) for value in aggregate)
    means = np.asarray([aggregate[str(snr)]["mean"] for snr in snrs])
    stds = np.asarray([aggregate[str(snr)]["sample_std"] for snr in snrs])
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for run in runs:
        values = run["test"]["per_snr_accuracy"]
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
        means - stds,
        means + stds,
        color="black",
        alpha=0.14,
        label="mean +/- sample std",
    )
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Test accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(snrs[::2])
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
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
    axis.bar(positions, means, yerr=stds, capsize=2, color="#F58518")
    axis.set_xticks(positions, classes, rotation=55, ha="right")
    axis.set_ylabel("Test recall")
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
    image = axis.imshow(normalized, cmap="Oranges", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(classes)), classes, rotation=70, ha="right", fontsize=7)
    axis.set_yticks(np.arange(len(classes)), classes, fontsize=7)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Aggregated normalized test confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_validation_test_plot(output: Path, aggregate: dict[str, Any]) -> None:
    keys = ("accuracy", "macro_f1", "low_snr_accuracy")
    labels = ("Accuracy", "Macro-F1", "Low-SNR accuracy")
    validation = [aggregate[f"validation_{key}"]["mean"] for key in keys]
    test = [aggregate[key]["mean"] for key in keys]
    positions = np.arange(len(keys))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.bar(positions - width / 2, validation, width, label="validation")
    axis.bar(positions + width / 2, test, width, label="test")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0.0, 0.7)
    axis.set_ylabel("Score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(path: Path, result: dict[str, Any]) -> None:
    aggregate = result["aggregate"]
    lines = [
        "# P2-T3 轻量 1D CNN 三种子最终测试结果",
        "",
        f"- 汇总时间：`{result['generated_at']}`",
        f"- 随机种子：`{result['seeds']}`",
        f"- 配置一致性检查：`{'通过' if result['protocol_consistent'] else '未通过'}`",
        "- 检查点：每个种子均使用验证集 Macro-F1 最优的 `best.pt`",
        "- 标准差口径：三次运行的样本标准差（n-1）",
        "",
        "## 一、最终测试指标",
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
            "## 二、逐种子测试结果",
            "",
            "| Seed | Accuracy | Macro-F1 | 低 SNR Accuracy | 最佳检查点 SHA-256 |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for run in result["runs"]:
        test = run["test"]
        lines.append(
            f"| {run['seed']} | {test['accuracy']:.6f} | {test['macro_f1']:.6f} | "
            f"{test['low_snr_accuracy']:.6f} | `{run['best_checkpoint_sha256']}` |"
        )

    lines.extend(
        [
            "",
            "## 三、验证集与测试集对照",
            "",
            "| 指标 | 验证均值 | 测试均值 | 测试－验证 |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("low_snr_accuracy", "低 SNR Accuracy"),
    ):
        validation_value = aggregate[f"validation_{key}"]["mean"]
        test_value = aggregate[key]["mean"]
        lines.append(
            f"| {label} | {validation_value:.6f} | {test_value:.6f} | "
            f"{test_value - validation_value:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## 四、测试推理效率",
            "",
            "| 指标 | 均值 | 标准差 |",
            "|---|---:|---:|",
            f"| 吞吐率（样本/秒） | {aggregate['samples_per_second']['mean']:.3f} | {aggregate['samples_per_second']['sample_std']:.3f} |",
            f"| 单样本端到端时延（毫秒） | {aggregate['milliseconds_per_sample']['mean']:.6f} | {aggregate['milliseconds_per_sample']['sample_std']:.6f} |",
            "",
            "## 五、分 SNR 测试准确率",
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
            "## 六、逐类别测试召回率",
            "",
            "| 类别 | 均值 | 标准差 |",
            "|---|---:|---:|",
        ]
    )
    for name, item in sorted(
        result["per_class_recall"].items(), key=lambda pair: pair[1]["mean"]
    ):
        lines.append(f"| {name} | {item['mean']:.6f} | {item['sample_std']:.6f} |")

    lines.extend(
        [
            "",
            "## 七、主要类别混淆",
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
            "## 八、结论",
            "",
            "- 三个种子的测试指标与各自验证指标接近，未发现明显验证—测试分布偏移。",
            "- 高 SNR 区间已进入性能平台，低 SNR和高阶 QAM仍是后续模型的主要改进方向。",
            "- 该测试结果是冻结配置后的最终结果，不再用于调整当前 CNN 的结构或超参数。",
            "- 后续模型应使用相同数据划分、评价指标和三种子预算与本基线比较。",
            "",
            "## 九、图表",
            "",
            "- `test_accuracy_by_snr_mean.png`：三种子分 SNR 测试曲线与均值±标准差；",
            "- `test_per_class_recall.png`：逐类别测试召回率均值±标准差；",
            "- `test_confusion_matrix_aggregate.png`：三种子聚合并按真实类别归一化的测试混淆矩阵；",
            "- `validation_test_comparison.png`：验证集和测试集核心指标对照。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    classes = json.loads(args.classes.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError("Class file must contain a JSON string list")

    runs: list[dict[str, Any]] = []
    comparable_configs: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_dir = run_root / f"cnn-seed-{seed}"
        metadata = load_json(run_dir / "run_config.json")
        summary = load_json(run_dir / "summary.json")
        validation = load_json(run_dir / "best_validation_metrics.json")
        test = load_json(run_dir / "test_metrics.json")
        checkpoint = run_dir / "checkpoints" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
        config = metadata["config"]
        if int(config["training"]["seed"]) != seed:
            raise ValueError(f"Training seed mismatch in {run_dir}")
        if int(test["sample_count"]) != int(summary["data"]["test_samples"]):
            raise ValueError(f"Incomplete test evaluation in {run_dir}")
        comparable_configs.append(comparable_config(config))
        runs.append(
            {
                "seed": seed,
                "run_directory": str(run_dir.relative_to(PROJECT_ROOT)),
                "config_sha256": metadata["config_sha256"],
                "best_epoch": int(summary["best_epoch"]) + 1,
                "best_checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
                "best_checkpoint_sha256": sha256_file(checkpoint),
                "validation": validation,
                "test": test,
            }
        )

    reference = comparable_configs[0]
    protocol_consistent = all(config == reference for config in comparable_configs[1:])
    if not protocol_consistent:
        raise ValueError("Formal run configurations differ beyond seed and run name")

    scalar_keys = ("loss", "accuracy", "macro_f1", "low_snr_accuracy")
    aggregate = {
        key: mean_std([float(run["test"][key]) for run in runs])
        for key in scalar_keys
    }
    for key in ("accuracy", "macro_f1", "low_snr_accuracy"):
        aggregate[f"validation_{key}"] = mean_std(
            [float(run["validation"][key]) for run in runs]
        )
        aggregate[f"test_minus_validation_{key}"] = mean_std(
            [
                float(run["test"][key]) - float(run["validation"][key])
                for run in runs
            ]
        )
    aggregate["samples_per_second"] = mean_std(
        [float(run["test"]["samples_per_second"]) for run in runs]
    )
    aggregate["milliseconds_per_sample"] = mean_std(
        [float(run["test"]["end_to_end_milliseconds_per_sample"]) for run in runs]
    )

    snr_keys = sorted(runs[0]["test"]["per_snr_accuracy"], key=int)
    per_snr = {
        snr: mean_std(
            [float(run["test"]["per_snr_accuracy"][snr]) for run in runs]
        )
        for snr in snr_keys
    }
    per_class_recall = {
        name: mean_std(
            [float(run["test"]["per_class_recall"][index]) for run in runs]
        )
        for index, name in enumerate(classes)
    }

    confusion = np.sum(
        [np.asarray(run["test"]["confusion_matrix"], dtype=np.int64) for run in runs],
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

    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "three-seed frozen final test summary",
        "seeds": args.seeds,
        "protocol_consistent": protocol_consistent,
        "normalized_protocol_sha256": protocol_hash(reference),
        "standard_deviation": "sample standard deviation (n-1)",
        "model": {"name": "lightweight_cnn", "parameters": 291128},
        "runs": runs,
        "aggregate": aggregate,
        "per_snr": per_snr,
        "per_class_recall": per_class_recall,
        "top_confusions": top_confusions,
    }

    json_path = output / "formal_test_summary.json"
    report_path = output / "formal_test_report.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(report_path, result)
    save_snr_plot(output / "test_accuracy_by_snr_mean.png", runs, per_snr)
    save_class_recall_plot(output / "test_per_class_recall.png", classes, per_class_recall)
    save_confusion_plot(output / "test_confusion_matrix_aggregate.png", classes, confusion)
    save_validation_test_plot(output / "validation_test_comparison.png", aggregate)

    print(f"Protocol consistent: {protocol_consistent}")
    print(
        f"Test Accuracy: {aggregate['accuracy']['mean']:.6f} "
        f"+/- {aggregate['accuracy']['sample_std']:.6f}"
    )
    print(
        f"Test Macro-F1: {aggregate['macro_f1']['mean']:.6f} "
        f"+/- {aggregate['macro_f1']['sample_std']:.6f}"
    )
    print(
        f"Low-SNR Accuracy: {aggregate['low_snr_accuracy']['mean']:.6f} "
        f"+/- {aggregate['low_snr_accuracy']['sample_std']:.6f}"
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
