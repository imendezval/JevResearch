import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from jevresearch.cli import compare, history, main
from jevresearch.controllers.random import RandomController
from jevresearch.controllers.jev import JevController
from jevresearch.core import ExperimentResult
from jevresearch.core.runner import Runner
from jevresearch.execution.subprocess import SubprocessExecutor
from jevresearch.storage.history import Store
from jevresearch.tasks.vision.cifar10 import CifarTask


class FixedExecutor:
    kind = "test"

    def execute(self, _task, _spec, _trial_id):
        return ExperimentResult("completed", 0.2, {"validation_accuracy": 0.2}, 1.0)


class FakeJev:
    kind = "fake"
    sdk_version = None
    timeout = 1
    max_retries = 0

    def invoke(self, body):
        options = list(body["questions"]["next_trial"]["criteria"])
        return {"model": "jev-1.13.0", "answers": {"next_trial": {
            "type": "choice", "choice": options[0], "confidence": 1.0,
            "probabilities": {id_: float(id_ == options[0]) for id_ in options}}},
            "usage": {"input_tokens": 10, "output_tokens": 2}}


class HistoryTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                         "vision dependencies are optional")
    def test_fake_jev_fixture_trains_two_real_process_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root / "history.sqlite")
            task = CifarTask(root / "data", root / "run", fixture=True, batch_size=16)
            runner = Runner(store, task, JevController(FakeJev()),
                            SubprocessExecutor(task.run_dir, 30))
            sid = runner.start(2, 7)
            runner.run(sid)
            exported = history(store, sid)
            self.assertEqual([t["status"] for t in exported["trials"]], ["completed", "completed"])
            self.assertEqual(exported["decision_attempts"][0]["status"], "selected")
            self.assertFalse(exported["controller_is_live"])
            self.assertEqual(exported["trials"][1]["parent_id"], 0)
            store.close()

    def test_comparison_marks_fake_controller_and_shows_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root / "history.sqlite")
            task = CifarTask(root / "data", root / "run", fixture=True)
            random_runner = Runner(store, task, RandomController(), FixedExecutor())
            jev_runner = Runner(store, task, JevController(FakeJev()), FixedExecutor())
            random_sid = random_runner.start(2, 7)
            jev_sid = jev_runner.start(2, 7)
            random_runner.run(random_sid)
            jev_runner.run(jev_sid)
            comparison = compare(history(store, random_sid), history(store, jev_sid))
            self.assertTrue(comparison["comparable_protocol"])
            self.assertFalse(comparison["jev"]["live"])
            self.assertEqual([x["attempted_trials"] for x in comparison["jev"]["trajectory"]], [1, 2])
            self.assertEqual(comparison["jev"]["controller_summary"]["logical_calls"], 1)
            self.assertEqual(comparison["jev"]["controller_summary"]["live_api_calls"], 0)
            changed_schedule = history(store, jev_sid)
            changed_schedule["settings"]["seed_schedule"] = "different"
            self.assertIn("seed_schedule", compare(history(store, random_sid),
                                                     changed_schedule)["differences"])
            store.close()

    def test_live_cli_requires_key_before_starting_session(self):
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "history.sqlite"
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}):
                with self.assertRaisesRegex(RuntimeError, "TYPESAFE_API_KEY"):
                    main(["run", "--db", str(db), "--budget", "2", "--controller", "jev"])
            self.assertFalse(db.exists())

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
