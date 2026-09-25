import json
import tempfile
import unittest
from pathlib import Path

from jevresearch.controllers.single import SingleCandidateController
from jevresearch.core import ExperimentResult
from jevresearch.core.candidate_generator import trial_seed
from jevresearch.core.runner import Runner
from jevresearch.storage.history import Store
from jevresearch.tasks.vision.cifar10 import CifarTask
from jevresearch.tasks.vision.cifar10.global_search import GlobalCandidateGenerator


class CheapExecutor:
    kind = "cheap-global"

    def execute(self, task, spec, trial_id):
        value = 1.0 - abs(spec.config["lr"] - 0.02)
        return ExperimentResult("completed", value, {"validation_accuracy": value}, 0.001)


class GlobalSearchTests(unittest.TestCase):
    def _run(self, strategy, domain, budget, *, pause=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root / "history.sqlite")
            task = CifarTask(root / "data", root / "run", fixture=True)
            generator = GlobalCandidateGenerator(domain, strategy)
            runner = Runner(store, task, SingleCandidateController(), CheapExecutor(), generator)
            sid = runner.start(budget, 7)
            if pause:
                runner.run(sid, 3)
                runner = Runner(store, task, SingleCandidateController(), CheapExecutor(),
                                GlobalCandidateGenerator(domain, strategy))
            runner.run(sid)
            rows = store.proposal_history(sid)
            self.assertEqual(len(rows), budget)
            self.assertTrue(all(row["status"] == "completed" for row in rows))
            self.assertEqual(len({row["config_key"] for row in rows}), budget)
            self.assertEqual([row["spec"]["seed"] for row in rows],
                             [trial_seed(7, number) for number in range(budget)])
            self.assertTrue(all(row["spec"]["parent_id"] is None for row in rows))
            self.assertEqual(json.loads(store.session(sid)["settings"])["proposal_domain"], domain)
            keys = [row["config_key"] for row in rows]
            phases = [row["proposal"]["phase"] for row in rows[1:]]
            store.close()
            return keys, phases

    def test_tpe_replay_and_adaptive_phase(self):
        uninterrupted, phases = self._run("tpe", "cifar-mixed-v1", 12)
        resumed, _ = self._run("tpe", "cifar-mixed-v1", 12, pause=True)
        self.assertEqual(uninterrupted, resumed)
        self.assertIn("startup", phases)
        self.assertIn("adaptive", phases)

    def test_cma_numeric_replay_after_population(self):
        uninterrupted, phases = self._run("cmaes", "cifar-sgd-numeric-v1", 7)
        resumed, _ = self._run("cmaes", "cifar-sgd-numeric-v1", 7, pause=True)
        self.assertEqual(uninterrupted, resumed)
        self.assertIn("adaptive", phases)

    def test_global_random_and_pool_domain_boundary(self):
        self._run("global-random", "cifar-mixed-v1", 3)
        with self.assertRaises(ValueError):
            GlobalCandidateGenerator("cifar-mixed-v1", "cmaes")


if __name__ == "__main__":
    unittest.main()
