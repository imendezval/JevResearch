import random
import tempfile
import unittest
from pathlib import Path

from jevresearch.tasks.vision.cifar10 import CifarTask
from jevresearch.tasks.vision.cifar10.domain import CifarDomain


class GlobalDomainTests(unittest.TestCase):
    def test_mixed_round_trip_and_zero_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = CifarTask(Path(tmp) / "data", Path(tmp) / "run", fixture=True)
            domain = CifarDomain("cifar-mixed-v1")
            for optimizer in ("sgd", "adamw"):
                for params in ({"optimizer": optimizer, "wd_mode": "zero", "lr": 0.01},
                               {"optimizer": optimizer, "wd_mode": "positive", "lr": 0.01,
                                "weight_decay": 0.0001}):
                    config = domain.config(task, params)
                    self.assertEqual(domain.parameters(config), params)
                    task.validate(config)
            with self.assertRaises(ValueError):
                domain.config(task, {"optimizer": "adamw", "wd_mode": "zero", "lr": 0.01,
                                     "weight_decay": 0.0})

    def test_numeric_bounds_and_distinct_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = CifarTask(Path(tmp) / "data", Path(tmp) / "run", fixture=True)
            mixed = CifarDomain("cifar-mixed-v1")
            numeric = CifarDomain("cifar-sgd-numeric-v1")
            self.assertNotEqual(mixed.fingerprint, numeric.fingerprint)
            self.assertEqual(numeric.parameters(task.baseline()),
                             {"lr": 0.01, "weight_decay": 0.0001})
            for _ in range(100):
                params = numeric.random_parameters(random.Random(_))
                self.assertEqual(numeric.parameters(numeric.config(task, params)), params)
            with self.assertRaises(ValueError):
                numeric.config(task, {"lr": 0.01, "weight_decay": 0.0})


if __name__ == "__main__":
    unittest.main()
