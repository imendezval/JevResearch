import json
import tempfile
import unittest
from pathlib import Path

from jevresearch.cli import history
from jevresearch.controllers.random import RandomController
from jevresearch.core import ExperimentResult
from jevresearch.core.runner import Runner
from jevresearch.storage.history import Store
from jevresearch.tasks.vision.cifar10 import CifarTask


class FixedExecutor:
    kind = "test"

    def execute(self, _task, _spec, _trial_id):
        return ExperimentResult("completed", 0.2, {"validation_accuracy": 0.2}, 1.0)


class HistoryTests(unittest.TestCase):
    def test_fixture_export_has_spec_operator_seed_parent_and_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root / "history.sqlite")
            task = CifarTask(root / "data", root / "run", fixture=True)
            runner = Runner(store, task, RandomController(), FixedExecutor())
            sid = runner.start(2, 7)
            runner.run(sid)
            exported = history(store, sid)
            self.assertEqual((exported["task"], exported["fixture"]), ("cifar10_fixture", True))
            self.assertEqual([t["status"] for t in exported["trials"]], ["completed", "completed"])
            first, second = exported["trials"]
            self.assertEqual(first["operator"], "baseline")
            self.assertIsNone(first["parent_id"])
            self.assertEqual(second["parent_id"], 0)
            self.assertNotEqual(first["spec"]["config"], second["spec"]["config"])
            self.assertNotEqual(first["spec"]["seed"], second["spec"]["seed"])
            self.assertEqual(second["best_so_far"], 0.2)
            self.assertEqual(second["spec"]["data_split"], exported["settings"]["data_split"])
            self.assertIn("dataset_sha256", exported["settings"]["task_details"])
            store.close()


if __name__ == "__main__":
    unittest.main()
