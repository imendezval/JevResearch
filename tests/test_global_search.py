import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jevresearch.controllers.single import SingleCandidateController
from jevresearch.controllers.random import RandomController
from jevresearch.controllers.jev import JevController
from jevresearch.core import ExperimentResult
from jevresearch.core.candidate_generator import trial_seed
from jevresearch.core.runner import Runner
from jevresearch.core.reporting import protocol_differences
from jevresearch.storage.history import Store
from jevresearch.tasks.vision.cifar10 import CifarTask
from jevresearch.tasks.vision.cifar10.global_search import GlobalCandidateGenerator


class CheapExecutor:
    kind = "cheap-global"

    def execute(self, task, spec, trial_id):
        value = 1.0 - abs(spec.config["lr"] - 0.02)
        return ExperimentResult("completed", value, {"validation_accuracy": value}, 0.001)


class FailingExecutor(CheapExecutor):
    def execute(self, task, spec, trial_id):
        if spec.parent_id is None and spec.config["lr"] != 0.01:
            return ExperimentResult("failed", None, {}, 0.001, "FixtureFailure", "injected")
        return super().execute(task, spec, trial_id)


class FakeJev:
    kind = "fake"
    sdk_version = None
    timeout = 1
    max_retries = 0

    def invoke(self, body):
        options = list(body["questions"]["next_trial"]["criteria"])
        selected = options[0]
        return {"model": "jev-1.13.0", "answers": {"next_trial": {
            "type": "choice", "choice": selected, "confidence": 1.0,
            "probabilities": {name: float(name == selected) for name in options}}}}


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
            settings = json.loads(store.session(sid)["settings"])
            self.assertEqual(settings["proposal_domain"], domain)
            self.assertIn("sampler_seed", settings)
            self.assertIn("sampler_options", settings)
            for row in store.trials(sid)[1:]:
                offer = store.offer(sid, row["offer_id"])
                offered = json.loads(offer["candidates"])
                self.assertEqual(sum(c["id"] == row["candidate_id"] for c in offered), 1)
                self.assertEqual(next(c["spec"] for c in offered if c["id"] == row["candidate_id"]),
                                 json.loads(row["spec"]))
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

    def test_failed_training_is_not_a_sampler_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root / "history.sqlite")
            task = CifarTask(root / "data", root / "run", fixture=True)
            runner = Runner(store, task, SingleCandidateController(), FailingExecutor(),
                            GlobalCandidateGenerator("cifar-mixed-v1", "tpe"))
            sid = runner.start(3, 7)
            runner.run(sid)
            rows = store.proposal_history(sid)
            self.assertEqual([row["status"] for row in rows], ["completed", "failed", "failed"])
            self.assertIsNone(rows[1]["result"]["objective"])
            self.assertEqual(rows[2]["proposal"]["completed_observations"], 1)
            store.close()

    def test_crash_before_offer_or_pending_replays_same_proposal(self):
        expected, _ = self._run("tpe", "cifar-mixed-v1", 4)
        for boundary in ("save_offer", "select"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store = Store(root / "history.sqlite")
                task = CifarTask(root / "data", root / "run", fixture=True)
                generator = GlobalCandidateGenerator("cifar-mixed-v1", "tpe")
                runner = Runner(store, task, SingleCandidateController(), CheapExecutor(), generator)
                sid = runner.start(4, 7)
                runner.run(sid, 1)
                with patch.object(store, boundary, side_effect=RuntimeError("injected crash")):
                    with self.assertRaisesRegex(RuntimeError, "injected crash"):
                        runner.run(sid)
                Runner(store, task, SingleCandidateController(), CheapExecutor(),
                       GlobalCandidateGenerator("cifar-mixed-v1", "tpe")).run(sid)
                self.assertEqual([row["config_key"] for row in store.proposal_history(sid)], expected)
                store.close()

    def test_random_and_fake_jev_receive_identical_global_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root / "history.sqlite")
            sessions = []
            for label, controller in (("random", RandomController()),
                                      ("fake_jev", JevController(FakeJev()))):
                task = CifarTask(root / "data", root / label, fixture=True)
                runner = Runner(store, task, controller, CheapExecutor(),
                                GlobalCandidateGenerator("cifar-mixed-v1", "global-pool"))
                sid = runner.start(2, 7)
                runner.run(sid)
                sessions.append(sid)
            pools = [json.loads(store.offers(sid)[0]["candidates"]) for sid in sessions]
            self.assertEqual(pools[0], pools[1])
            self.assertEqual(len(pools[0]), 8)
            self.assertEqual(store.decision_attempts(sessions[1])[0]["status"], "selected")
            store.close()

    def test_duplicate_resample_cap_stops_without_training_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root / "history.sqlite")
            task = CifarTask(root / "data", root / "run", fixture=True)
            generator = GlobalCandidateGenerator("cifar-sgd-numeric-v1", "global-random")
            runner = Runner(store, task, SingleCandidateController(), CheapExecutor(), generator)
            sid = runner.start(2, 7)
            baseline_params = generator.domain.parameters(task.baseline())
            with patch.object(type(generator.domain), "random_parameters", return_value=baseline_params):
                runner.run(sid)
            self.assertEqual(len(store.trials(sid)), 1)
            self.assertEqual(len(store.offers(sid)), 0)
            self.assertEqual(store.session(sid)["stop_reason"], "global proposal resample limit reached")
            store.close()

    def test_comparison_requires_same_domain_not_same_strategy(self):
        base = {"settings": {"task": "cifar10", "protocol": "official-train-v1",
                             "proposal_strategy": "global-random", "proposal_domain": "cifar-mixed-v1",
                             "domain_fingerprint": "mixed", "task_details": {"device": "cpu"}},
                "source": {"digest": "same"}}
        other = {"settings": {**base["settings"], "proposal_strategy": "tpe"},
                 "source": {"digest": "same"}}
        self.assertEqual(protocol_differences(base, other), [])
        other["settings"]["proposal_domain"] = "cifar-sgd-numeric-v1"
        other["settings"]["domain_fingerprint"] = "numeric"
        self.assertEqual(protocol_differences(base, other),
                         ["proposal_domain", "domain_fingerprint"])


if __name__ == "__main__":
    unittest.main()
