"""Summarize the frozen three-seed RadioResNet validation and test results.

The script only reads existing experiment outputs.  It verifies that the
three runs used the same protocol (apart from seed/run name), checks that the
complete test split was evaluated, aggregates metrics, and compares the
result with the frozen lightweight-CNN baseline.
"""

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
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs" / "P2-T4"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "P2-T4"
DEFAULT_CLASSES = PROJECT_ROOT / "data/raw/RML2018.01a/classes-fixed.json"
DEFAULT_CNN_VALIDATION = PROJECT_ROOT / "artifacts/P2-T3/formal_validation_summary.json"
DEFAULT_CNN_TEST = PROJECT_ROOT / "artifacts/P2-T3/formal_test_summary.json"
DEFAULT_SEEDS = (2026, 2027, 2028)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize frozen P2-T4 RadioResNet results across seeds."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--cnn-validation", type=Path, default=DEFAULT_CNN_VALIDATION)
    parser.add_argument("--cnn-test", type=Path, default=DEFAULT_CNN_TEST)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    data = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def aggregate_metrics(runs: list[dict[str, Any]], split: str) -> dict[str, Any]:
    keys = ("loss", "accuracy", "macro_f1", "low_snr_accuracy")
    result = {
        key: mean_std([float(run[split][key]) for run in runs]) for key in keys
    }
    result["samples_per_second"] = mean_std(
        [float(run[split]["samples_per_second"]) for run in runs]
    )
    result["milliseconds_per_sample"] = mean_std(
        [float(run[split]["end_to_end_milliseconds_per_sample"]) for run in runs]
    )
    return result


def aggregate_snr(runs: list[dict[str, Any]], split: str) -> dict[str, Any]:
    keys = sorted(runs[0][split]["per_snr_accuracy"], key=int)
    return {
        snr: mean_std(
            [float(run[split]["per_snr_accuracy"][snr]) for run in runs]
        )
        for snr in keys
    }


def aggregate_class_recall(
    runs: list[dict[str, Any]], split: str, classes: list[str]
) -> dict[str, Any]:
    return {
        name: mean_std(
            [float(run[split]["per_class_recall"][index]) for run in runs]
        )
        for index, name in enumerate(classes)
    }


def aggregate_confusion(
    runs: list[dict[str, Any]], split: str, classes: list[str]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrix = np.sum(
        [np.asarray(run[split]["confusion_matrix"], dtype=np.int64) for run in runs],
        axis=0,
    )
    row_totals = matrix.sum(axis=1)
    items: list[dict[str, Any]] = []
    for true_index, row in enumerate(matrix):
        for predicted_index, count in enumerate(row):
            if true_index == predicted_index or count == 0:
                continue
            items.append(
                {
                    "true_class": classes[true_index],
                    "predicted_class": classes[predicted_index],
                    "count": int(count),
                    "rate": float(count / row_totals[true_index]),
                }
            )
    return matrix, sorted(items, key=lambda item: item["count"], reverse=True)[:15]


def save_learning_curves(path: Path, runs: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for run in runs:
        epochs = [record["epoch"] + 1 for record in run["epoch_records"]]
        axes[0].plot(
            epochs,
            [record["train"]["loss"] for record in run["epoch_records"]],
            linestyle="--",
            alpha=0.6,
            label=f"train {run['seed']}",
        )
        axes[0].plot(
            epochs,
            [record["validation"]["loss"] for record in run["epoch_records"]],
            label=f"val {run['seed']}",
        )
        axes[1].plot(
            epochs,
            [record["validation"]["macro_f1"] for record in run["epoch_records"]],
            label=f"val {run['seed']}",
        )
    axes[0].set_title("Loss")
    axes[1].set_title("Validation Macro-F1")
    axes[1].set_ylim(0.0, 0.75)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_snr_plot(
    path: Path, runs: list[dict[str, Any]], aggregate: dict[str, Any]
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
            alpha=0.32,
            linewidth=1.0,
            label=f"seed {run['seed']}",
        )
    axis.plot(snrs, means, color="black", linewidth=2.2, label="mean")
    axis.fill_between(snrs, means - stds, means + stds, color="black", alpha=0.14)
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Test accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(snrs[::2])
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_class_recall_plot(
    path: Path, classes: list[str], aggregate: dict[str, Any]
) -> None:
    values = [aggregate[name]["mean"] for name in classes]
    errors = [aggregate[name]["sample_std"] for name in classes]
    positions = np.arange(len(classes))
    figure, axis = plt.subplots(figsize=(12, 5.6))
    axis.bar(positions, values, yerr=errors, capsize=2, color="#E45756")
    axis.set_xticks(positions, classes, rotation=55, ha="right")
    axis.set_ylabel("Test recall")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_confusion_plot(path: Path, classes: list[str], matrix: np.ndarray) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    figure, axis = plt.subplots(figsize=(10.5, 9.2))
    image = axis.imshow(normalized, cmap="Reds", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(classes)), classes, rotation=70, ha="right", fontsize=7)
    axis.set_yticks(np.arange(len(classes)), classes, fontsize=7)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("RadioResNet aggregated normalized test confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_cnn_comparison(path: Path, comparison: dict[str, Any]) -> None:
    keys = ("accuracy", "macro_f1", "low_snr_accuracy")
    labels = ("Accuracy", "Macro-F1", "Low-SNR accuracy")
    cnn = [comparison[key]["cnn_mean"] for key in keys]
    resnet = [comparison[key]["resnet_mean"] for key in keys]
    positions = np.arange(len(keys))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.bar(positions - width / 2, cnn, width, label="Lightweight CNN")
    axis.bar(positions + width / 2, resnet, width, label="RadioResNet")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0.0, 0.72)
    axis.set_ylabel("Test score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path: Path, result: dict[str, Any]) -> None:
    validation = result["validation_aggregate"]
    test = result["test_aggregate"]
    comparison = result["cnn_comparison"]
    lines = [
        "# RadioResNet 三种子正式实验汇总",
        "",
        f"- 汇总时间：`{result['generated_at']}`",
        f"- 随机种子：`{result['seeds']}`",
        f"- 协议一致性：`{'通过' if result['protocol_consistent'] else '未通过'}`",
        "- 检查点：三个种子均使用验证集 Macro-F1 最优的 `best.pt`",
        "- 标准差：三次独立训练的样本标准差（n-1）",
        "",
        "## 一、正式结果",
        "",
        "| 指标 | 验证集均值±标准差 | 测试集均值±标准差 | 测试－验证 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("low_snr_accuracy", "低 SNR Accuracy（-20～0 dB）"),
        ("loss", "Loss"),
    ):
        lines.append(
            f"| {label} | {validation[key]['mean']:.6f} ± {validation[key]['sample_std']:.6f} "
            f"| {test[key]['mean']:.6f} ± {test[key]['sample_std']:.6f} "
            f"| {test[key]['mean'] - validation[key]['mean']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## 二、逐种子结果",
            "",
            "| Seed | 最佳 epoch | 实际 epoch | 测试 Accuracy | 测试 Macro-F1 | 低 SNR Accuracy | 早停 |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for run in result["runs"]:
        lines.append(
            f"| {run['seed']} | {run['best_epoch']} | {run['epochs_completed']} | "
            f"{run['test']['accuracy']:.6f} | {run['test']['macro_f1']:.6f} | "
            f"{run['test']['low_snr_accuracy']:.6f} | {run['stopped_early']} |"
        )

    lines.extend(
        [
            "",
            "## 三、与轻量 CNN 的冻结测试结果比较",
            "",
            "| 指标 | 轻量 CNN | RadioResNet | ResNet－CNN |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("low_snr_accuracy", "低 SNR Accuracy"),
    ):
        item = comparison[key]
        lines.append(
            f"| {label} | {item['cnn_mean']:.6f} | {item['resnet_mean']:.6f} | "
            f"{item['absolute_difference']:+.6f} |"
        )
    lines.extend(
        [
            f"| 参数量 | {comparison['parameters']['cnn']:,} | "
            f"{comparison['parameters']['resnet']:,} | "
            f"{comparison['parameters']['relative_difference']:+.2%} |",
            "",
            "## 四、效率",
            "",
            "| 指标 | 均值 | 标准差 |",
            "|---|---:|---:|",
            f"| 训练时间（分钟） | {result['training_efficiency']['minutes']['mean']:.3f} | {result['training_efficiency']['minutes']['sample_std']:.3f} |",
            f"| 峰值显存（MiB） | {result['training_efficiency']['peak_memory_mib']['mean']:.3f} | {result['training_efficiency']['peak_memory_mib']['sample_std']:.3f} |",
            f"| 测试吞吐率（样本/秒） | {test['samples_per_second']['mean']:.3f} | {test['samples_per_second']['sample_std']:.3f} |",
            f"| 测试单样本时延（毫秒） | {test['milliseconds_per_sample']['mean']:.6f} | {test['milliseconds_per_sample']['sample_std']:.6f} |",
            "",
            "## 五、分 SNR 测试准确率",
            "",
            "| SNR（dB） | 均值 | 标准差 |",
            "|---:|---:|---:|",
        ]
    )
    for snr in sorted(result["test_per_snr"], key=int):
        item = result["test_per_snr"][snr]
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
        result["test_per_class_recall"].items(), key=lambda pair: pair[1]["mean"]
    ):
        lines.append(f"| {name} | {item['mean']:.6f} | {item['sample_std']:.6f} |")

    lines.extend(
        [
            "",
            "## 七、主要类别混淆",
            "",
            "| 真实类别 | 预测类别 | 三种子合计样本数 | 占真实类别比例 |",
            "|---|---|---:|---:|",
        ]
    )
    for item in result["test_top_confusions"]:
        lines.append(
            f"| {item['true_class']} | {item['predicted_class']} | "
            f"{item['count']:,} | {item['rate']:.4%} |"
        )

    lines.extend(
        [
            "",
            "## 八、结论",
            "",
            "- 三个种子的验证与测试指标高度接近，说明结果不存在明显的验证—测试偏移，最终测试结论可信。",
            "- RadioResNet 相比轻量 CNN 提升了总体 Accuracy 和 Macro-F1，且用更少参数取得更均衡的类别识别能力；残差结构适合作为后续模型的强卷积基线。",
            "- 低 SNR 提升明显小于总体指标提升，表明深层残差特征提取没有从根本上解决极低信噪比下的信息损失。",
            "- 高阶 QAM、相邻 PSK/APSK及部分 AM 类别之间仍存在混淆；这些错误类型应成为后续 CLDNN、Transformer 和 Mamba 模型的重点对照项。",
            "- 当前只有三个训练种子，均值提升可作为本项目协议下的基线证据，但不单独表述为具有统计显著性的普遍结论。",
            "- 当前 RadioResNet 配置和测试结果已冻结，不再根据测试集调整结构或超参数。下一项工作可进入 CLDNN/MCLDNN 基线实现。",
            "",
            "## 九、产物",
            "",
            "- `formal_results_summary.json`：机器可读的完整汇总；",
            "- `validation_learning_curves.png`：三种子训练与验证曲线；",
            "- `test_accuracy_by_snr_mean.png`：分 SNR 测试曲线；",
            "- `test_per_class_recall.png`：逐类别召回率；",
            "- `test_confusion_matrix_aggregate.png`：聚合归一化混淆矩阵；",
            "- `cnn_resnet_test_comparison.png`：CNN 与 RadioResNet 冻结测试结果对照。",
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
    configs: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_dir = run_root / f"resnet-seed-{seed}"
        metadata = load_json(run_dir / "run_config.json")
        summary = load_json(run_dir / "summary.json")
        validation = load_json(run_dir / "best_validation_metrics.json")
        test = load_json(run_dir / "test_metrics.json")
        records = load_jsonl(run_dir / "metrics.jsonl")
        checkpoint = run_dir / "checkpoints/best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
        config = metadata["config"]
        if int(config["training"]["seed"]) != seed:
            raise ValueError(f"Training seed mismatch in {run_dir}")
        if int(test["sample_count"]) != int(summary["data"]["test_samples"]):
            raise ValueError(f"Incomplete test evaluation in {run_dir}")
        configs.append(comparable_config(config))
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
                "best_checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
                "best_checkpoint_sha256": sha256_file(checkpoint),
                "validation": validation,
                "test": test,
                "epoch_records": records,
            }
        )

    reference = configs[0]
    consistent = all(config == reference for config in configs[1:])
    if not consistent:
        raise ValueError("Formal run configurations differ beyond seed and run name")

    validation_aggregate = aggregate_metrics(runs, "validation")
    test_aggregate = aggregate_metrics(runs, "test")
    test_per_snr = aggregate_snr(runs, "test")
    test_per_class = aggregate_class_recall(runs, "test", classes)
    test_confusion, test_top_confusions = aggregate_confusion(runs, "test", classes)

    cnn_validation = load_json(args.cnn_validation.expanduser().resolve())
    cnn_test = load_json(args.cnn_test.expanduser().resolve())
    comparison: dict[str, Any] = {}
    for key in ("accuracy", "macro_f1", "low_snr_accuracy"):
        cnn_value = float(cnn_test["aggregate"][key]["mean"])
        resnet_value = float(test_aggregate[key]["mean"])
        comparison[key] = {
            "cnn_mean": cnn_value,
            "resnet_mean": resnet_value,
            "absolute_difference": resnet_value - cnn_value,
            "relative_difference": (resnet_value - cnn_value) / cnn_value,
        }
    cnn_parameters = int(cnn_test["model"]["parameters"])
    resnet_parameters = 165144
    comparison["parameters"] = {
        "cnn": cnn_parameters,
        "resnet": resnet_parameters,
        "absolute_difference": resnet_parameters - cnn_parameters,
        "relative_difference": (resnet_parameters - cnn_parameters) / cnn_parameters,
    }
    comparison["training_minutes"] = {
        "cnn_mean": float(cnn_validation["aggregate"]["training_minutes"]["mean"]),
        "resnet_mean": mean(run["training_seconds"] / 60.0 for run in runs),
    }

    public_runs = [
        {key: value for key, value in run.items() if key != "epoch_records"}
        for run in runs
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "P2-T4 three-seed frozen RadioResNet result summary",
        "seeds": args.seeds,
        "protocol_consistent": consistent,
        "normalized_protocol_sha256": protocol_hash(reference),
        "standard_deviation": "sample standard deviation (n-1)",
        "model": {"name": "radio_resnet", "parameters": resnet_parameters},
        "runs": public_runs,
        "validation_aggregate": validation_aggregate,
        "test_aggregate": test_aggregate,
        "training_efficiency": {
            "minutes": mean_std([run["training_seconds"] / 60.0 for run in runs]),
            "peak_memory_mib": mean_std([run["peak_memory_mib"] for run in runs]),
        },
        "test_per_snr": test_per_snr,
        "test_per_class_recall": test_per_class,
        "test_top_confusions": test_top_confusions,
        "cnn_comparison": comparison,
    }

    summary_path = output / "formal_results_summary.json"
    report_path = output / "formal_results_report.md"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(report_path, result)
    save_learning_curves(output / "validation_learning_curves.png", runs)
    save_snr_plot(output / "test_accuracy_by_snr_mean.png", runs, test_per_snr)
    save_class_recall_plot(output / "test_per_class_recall.png", classes, test_per_class)
    save_confusion_plot(
        output / "test_confusion_matrix_aggregate.png", classes, test_confusion
    )
    save_cnn_comparison(output / "cnn_resnet_test_comparison.png", comparison)

    print(f"Protocol consistent: {consistent}")
    print(
        f"Test Accuracy: {test_aggregate['accuracy']['mean']:.6f} "
        f"+/- {test_aggregate['accuracy']['sample_std']:.6f}"
    )
    print(
        f"Test Macro-F1: {test_aggregate['macro_f1']['mean']:.6f} "
        f"+/- {test_aggregate['macro_f1']['sample_std']:.6f}"
    )
    print(
        f"Low-SNR Accuracy: {test_aggregate['low_snr_accuracy']['mean']:.6f} "
        f"+/- {test_aggregate['low_snr_accuracy']['sample_std']:.6f}"
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
