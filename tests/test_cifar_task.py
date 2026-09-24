import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from jevresearch.core.candidate_generator import CandidateGenerator, make_spec
from jevresearch.core import SearchState
from jevresearch.tasks.vision.cifar10.task import CifarTask, fixture_labels, stratified_indices


class CifarTaskTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                         "vision dependencies are optional")
    def test_worker_uses_saved_train_and_validation_only(self):
        from jevresearch.tasks.vision.cifar10.worker import _datasets, train
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = CifarTask(root / "data", root / "run", fixture=True, batch_size=16)
            train_set, val_set = _datasets(task.details(), task.split, 1)
            self.assertEqual(set(train_set.indices) & set(val_set.indices), set())
            self.assertEqual((len(train_set), len(val_set)), (80, 20))
            spec = make_spec(task, task.baseline(), 123, "source", None)
            result = train({"spec": vars(spec), "details": task.details(), "split": task.split})
            self.assertEqual(result.status, "completed")
            self.assertEqual((result.metrics["train_count"], result.metrics["validation_count"]), (80, 20))
            self.assertEqual(result.objective, result.metrics["validation_accuracy"])
            self.assertGreaterEqual(result.objective, 0)
            self.assertLessEqual(result.objective, 1)

    def test_stratified_split_is_saved_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = CifarTask(root / "data", root / "run", fixture=True)
            manifest = json.loads(task.split_path.read_text())
            train, val = manifest["train_indices"], manifest["val_indices"]
            self.assertEqual((len(train), len(val)), (80, 20))
            self.assertFalse(set(train) & set(val))
            self.assertEqual(set(train) | set(val), set(range(100)))
            self.assertEqual(stratified_indices(fixture_labels(), 1729, 8), (train, val))
            self.assertEqual(task.data_split, manifest["split_sha256"])
            self.assertEqual(CifarTask(root / "data", root / "run", fixture=True).details(), task.details())
            with self.assertRaises(ValueError):
                CifarTask(root / "data", root / "run", fixture=True, split_seed=1730)

    def test_candidates_are_complete_valid_and_novel(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = CifarTask(Path(tmp) / "data", Path(tmp) / "run", fixture=True)
            state = SearchState(1, 1, 3, 0, 0.1, task.baseline(), ())
            generator = CandidateGenerator()
            candidates = generator.generate(task, state, 5, "source")
            self.assertEqual(candidates, generator.generate(task, state, 5, "source"))
            self.assertEqual(len(candidates), len({c.spec.config_key for c in candidates}))
            self.assertTrue(all(c.parent_id == 0 for c in candidates))
            self.assertTrue(all(c.spec.seed == candidates[0].spec.seed for c in candidates))
            for candidate in candidates:
                task.validate(candidate.config)
                self.assertNotEqual(candidate.config, task.baseline())
            attempted = ({"config_key": candidates[0].spec.config_key},)
            state = SearchState(1, 2, 3, 0, 0.1, task.baseline(), attempted)
            self.assertNotIn(candidates[0].spec.config_key,
                             {c.spec.config_key for c in generator.generate(task, state, 5, "source")})


if __name__ == "__main__":
    unittest.main()
