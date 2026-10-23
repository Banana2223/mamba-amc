"""Fast unit checks for the P2-T2 framework (no RadioML file required)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from amc_framework.config import load_experiment_config
from amc_framework.metrics import ClassificationAccumulator
from amc_framework.models import create_model


class ConfigurationTests(unittest.TestCase):
    def test_unknown_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps({"unknown": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_config(path)


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


if __name__ == "__main__":
    unittest.main()

