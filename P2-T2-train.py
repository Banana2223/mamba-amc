"""P2-T2 command-line entry for unified AMC training and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from amc_framework import ExperimentRunner, load_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "P2-T2-smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an AMC model with the unified P2-T2 framework."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--output-root")
    parser.add_argument("--disable-amp", action="store_true")
    test_mode = parser.add_mutually_exclusive_group()
    test_mode.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Evaluate the fixed test split using the best checkpoint. "
            "Use only after the experiment configuration is frozen."
        ),
    )
    test_mode.add_argument(
        "--test-only",
        action="store_true",
        help=(
            "Skip training and evaluate the run's best checkpoint on the test "
            "split. Requires --resume and a frozen experiment configuration."
        ),
    )
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def apply_overrides(config: object, args: argparse.Namespace) -> None:
    # Attribute access stays explicit so command-line overrides cannot create
    # arbitrary configuration fields.
    if args.device is not None:
        config.training.device = args.device
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.max_epochs is not None:
        config.training.max_epochs = args.max_epochs
    if args.max_train_batches is not None:
        config.training.max_train_batches = args.max_train_batches
    if args.max_validation_batches is not None:
        config.training.max_validation_batches = args.max_validation_batches
    if args.run_name is not None:
        config.output.run_name = args.run_name
    if args.output_root is not None:
        config.output.root = args.output_root
    if args.disable_amp:
        config.training.amp = False
    config.validate()


def main() -> int:
    args = parse_args()
    if args.test_only and args.resume is None:
        raise ValueError("--test-only requires --resume to identify the run")
    config = load_experiment_config(args.config)
    apply_overrides(config, args)
    if args.print_config:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))

    with ExperimentRunner(config, resume_path=args.resume) as runner:
        if args.test_only:
            test_metrics = runner.evaluate_test()
            print(f"Run directory:   {runner.output_dir}")
            print(
                "Test-only evaluation completed: "
                f"accuracy={test_metrics['accuracy']:.6f}, "
                f"macro_f1={test_metrics['macro_f1']:.6f}"
            )
            return 0
        summary = runner.fit()
        print(f"Run directory:   {runner.output_dir}")
        print(f"Best epoch:      {summary['best_epoch']}")
        print(f"Best metric:     {summary['best_metric']:.6f}")
        if args.evaluate_test:
            test_metrics = runner.evaluate_test()
            print(
                "Test evaluation completed: "
                f"accuracy={test_metrics['accuracy']:.6f}, "
                f"macro_f1={test_metrics['macro_f1']:.6f}"
            )
        else:
            print("Test split was not evaluated (expected during development).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

