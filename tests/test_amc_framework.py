"""Fast unit checks for the P2-T2 framework (no RadioML file required)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from amc_framework.config import experiment_config_from_dict, load_experiment_config
from amc_framework.metrics import ClassificationAccumulator
from amc_framework.models import create_model


class ConfigurationTests(unittest.TestCase):
    def test_unknown_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps({"unknown": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_config(path)

    def test_minimum_epochs_cannot_exceed_maximum(self) -> None:
        with self.assertRaises(ValueError):
            experiment_config_from_dict(
                {"training": {"min_epochs": 11, "max_epochs": 10}}
            )


class MetricTests(unittest.TestCase):
    def test_streaming_metrics(self) -> None:
        accumulator = ClassificationAccumulator(
            num_classes=3,
            low_snr_min_db=-20,
            low_snr_max_db=0,
        )
        predictions = torch.tensor([0, 0, 1, 2])
        logits = torch.nn.functional.one_hot(predictions, num_classes=3).float()
        labels = torch.tensor([0, 1, 1, 2])
        snrs = torch.tensor([-20, -10, 2, 10])
        accumulator.update(logits, labels, snrs, loss=2.0)
        values = accumulator.compute()
        self.assertAlmostEqual(values["loss"], 2.0)
        self.assertAlmostEqual(values["accuracy"], 0.75)
        self.assertAlmostEqual(values["macro_f1"], 7.0 / 9.0)
        self.assertAlmostEqual(values["low_snr_accuracy"], 0.5)
        self.assertEqual(values["sample_count"], 4)


class ModelTests(unittest.TestCase):
    def test_probe_contract(self) -> None:
        model = create_model("tiny_probe", num_classes=24)
        output = model(torch.randn(2, 2, 1024))
        self.assertEqual(tuple(output.shape), (2, 24))

    def test_lightweight_cnn_contract_and_gradient(self) -> None:
        model = create_model(
            "lightweight_cnn",
            num_classes=24,
            base_channels=16,
            pooled_length=4,
            classifier_width=32,
            dropout=0.0,
        )
        output = model(torch.randn(2, 2, 1024))
        self.assertEqual(tuple(output.shape), (2, 24))
        output.sum().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_lightweight_cnn_rejects_wrong_input(self) -> None:
        model = create_model("lightweight_cnn", num_classes=24)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 1024))


if __name__ == "__main__":
    unittest.main()

