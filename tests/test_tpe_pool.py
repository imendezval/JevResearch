import json
import contextlib
import io
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from jevresearch.controllers.jev import JevController
from jevresearch.controllers.random import RandomController
from jevresearch.cli import main
from jevresearch.core import ExperimentResult, ExperimentSpec
from jevresearch.core.candidate_generator import trial_seed
from jevresearch.core.reporting import history, protocol_differences
from jevresearch.core.runner import Runner
from jevresearch.core.study_report import study_report
from jevresearch.core.study_runner import StudyRunner
from jevresearch.storage.history import InvariantError, Store
from jevresearch.tasks.vision.cifar10 import CifarTask
from jevresearch.tasks.vision.cifar10.global_search import GlobalCandidateGenerator


class CheapExecutor:
    kind = "cheap-pool"

    def execute(self, task, spec, trial_id):
        value = 1.0 - abs(spec.config["lr"] - 0.02)
        return ExperimentResult("completed", value, {"validation_accuracy": value}, 0.001)


class FakeJev:
    kind = "fake"
    sdk_version = None
    timeout = 1
    max_retries = 0

    def __init__(self, fail_once=False):
        self.fail_once = fail_once
        self.calls = 0

    def invoke(self, body):
        self.calls += 1
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("injected transport failure")
        options = list(body["questions"]["next_trial"]["criteria"])
        chosen = options[0]
        return {"model": "jev-1.13.0", "answers": {"next_trial": {
            "type": "choice", "choice": chosen, "confidence": 1.0,
            "probabilities": {name: float(name == chosen) for name in options}}}}


class TpePoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "history.sqlite"
        self.task = CifarTask(self.root / "data", self.root / "run", fixture=True)
        self.store = Store(self.path)
        self.addCleanup(lambda: self.store.close())

    def reopen(self):
        self.store.close()
        self.store = Store(self.path)

    def runner(self, controller=None, width=2):
        return Runner(self.store, self.task, controller or RandomController(), CheapExecutor(),
                      GlobalCandidateGenerator("cifar-mixed-v1", "tpe-pool", width))

    def offers(self, sid):
        return [json.loads(row["candidates"]) for row in self.store.offers(sid)]

    def test_pool_width_identity_and_first_offer_pairing(self):
        for width in (2, 8):
            sessions = []
            for controller in (RandomController(), JevController(FakeJev())):
                runner = self.runner(controller, width)
                sid = runner.start(2, 7)
                runner.run(sid)
                sessions.append(sid)
            first, second = (self.offers(sid)[0] for sid in sessions)
            self.assertEqual(first, second)
            self.assertEqual(len(first), width)
            self.assertEqual(len({c["spec"]["config"]["lr"] for c in first}), width)
            self.assertEqual({c["spec"]["seed"] for c in first}, {trial_seed(7, 1)})
            for sid in sessions:
                rows = self.store.proposal_history(sid)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[1]["offer"]["candidates"], first)
                self.assertEqual(rows[1]["offer"]["selected_id"], rows[1]["candidate_id"])
            self.assertEqual(self.store.decision_attempts(sessions[1])[0]["status"], "selected")

            export = history(self.store, sessions[1])
            audit = export["offers"][0]
            self.assertEqual(audit["pool_policy"], "tpe-pool-v1")
            self.assertEqual(audit["accepted_count"], width)
            self.assertEqual(audit["declined_untrained_count"], width - 1)
            self.assertEqual([c["parameters"]["pool_slot"] for c in audit["candidates"]],
                             list(range(width)))
            self.assertEqual(sum(c["audit_status"] == "declined_untrained"
                                 for c in audit["candidates"]), width - 1)
            self.assertEqual(audit["selected_trial"]["status"], "completed")
            self.assertEqual(len(audit["decision_attempt_ids"]), 1)

    def test_resume_replays_declined_suggestions_and_adapts(self):
        def run(paused):
            runner = self.runner(JevController(FakeJev(), max_calls=20), 3)
            sid = runner.start(12, 7)
            if paused:
                runner.run(sid, 3)
                self.reopen()
                runner = self.runner(JevController(FakeJev(), max_calls=20), 3)
            runner.run(sid)
            offers = self.offers(sid)
            keys = [row["config_key"] for row in self.store.proposal_history(sid)]
            self.assertEqual(len(keys), 12)
            self.assertEqual(len({ExperimentSpec(**c["spec"]).config_key
                                  for pool in offers for c in pool}), 33)
            self.assertEqual(offers[-1][0]["parameters"]["phase"], "adaptive")
            self.assertEqual(offers[-1][0]["parameters"]["completed_observations"], 11)
            return keys, offers

        uninterrupted = run(False)
        self.store.close()
        self.path = self.root / "resumed.sqlite"
        self.store = Store(self.path)
        resumed = run(True)
        self.assertEqual(uninterrupted, resumed)

    def test_saved_offer_is_reused_after_jev_failure(self):
        transport = FakeJev(fail_once=True)
        runner = self.runner(JevController(transport), 2)
        sid = runner.start(2, 7)
        runner.run(sid)
        self.assertEqual(len(self.store.offers(sid)), 1)
        self.assertEqual(len(self.store.trials(sid)), 1)
        with patch.object(runner.generator, "generate", side_effect=AssertionError("redrew offer")):
            runner.run(sid)
        self.assertEqual(len(self.store.offers(sid)), 1)
        self.assertEqual(len(self.store.trials(sid)), 2)
        self.assertEqual([a["status"] for a in self.store.decision_attempts(sid)],
                         ["unknown", "selected"])

    def test_replay_rejects_tampered_offer(self):
        runner = self.runner(width=2)
        sid = runner.start(3, 7)
        runner.run(sid, 2)
        offer = self.store.offers(sid)[0]
        candidates = json.loads(offer["candidates"])
        candidates[1]["parameters"]["pool_slot"] = 99
        with self.store.db:
            self.store.db.execute("UPDATE offers SET candidates=? WHERE id=?",
                                  (json.dumps(candidates), offer["id"]))
        with self.assertRaisesRegex(InvariantError, "replay differs"):
            runner.run(sid)

    def test_duplicate_exhaustion_stops_without_partial_offer(self):
        runner = self.runner(width=2)
        sid = runner.start(2, 7)
        baseline = runner.generator.domain.parameters(self.task.baseline())
        with patch.object(type(runner.generator.domain), "suggest", return_value=baseline):
            runner.run(sid)
        self.assertEqual(len(self.store.offers(sid)), 0)
        self.assertEqual(len(self.store.trials(sid)), 1)
        self.assertEqual(self.store.session(sid)["stop_reason"],
                         "global proposal resample limit reached")

    def test_failed_training_adds_no_objective_or_failed_declined_experiments(self):
        class FailingExecutor(CheapExecutor):
            def execute(self, task, spec, trial_id):
                if spec.config["lr"] != 0.01:
                    return ExperimentResult("failed", None, {}, 0.001, "Injected", "failed")
                return super().execute(task, spec, trial_id)

        runner = self.runner(width=3)
        runner.executor = FailingExecutor()
        sid = runner.start(3, 7)
        runner.run(sid)
        rows = self.store.proposal_history(sid)
        self.assertEqual([row["status"] for row in rows], ["completed", "failed", "failed"])
        self.assertIsNone(rows[1]["result"]["objective"])
        self.assertEqual(rows[2]["proposal"]["completed_observations"], 1)
        self.assertEqual(len(self.store.offers(sid)), 2)
        self.assertEqual(sum(len(offer) - 1 for offer in self.offers(sid)), 4)

    def test_crashes_before_offer_after_offer_and_after_validated_decision(self):
        for boundary in ("save_offer", "select", "validated_select"):
            with self.subTest(boundary=boundary):
                self.store.close()
                self.path = self.root / f"{boundary}.sqlite"
                self.store = Store(self.path)
                transport = FakeJev()
                controller = JevController(transport) if boundary == "validated_select" else RandomController()
                runner = self.runner(controller, 2)
                sid = runner.start(2, 7)
                runner.run(sid, 1)
                expected = runner.generator.generate(
                    self.task, self.store.state(sid), 7,
                    json.loads(self.store.session(sid)["source"])["digest"],
                    self.store.proposal_history(sid))
                method = "save_offer" if boundary == "save_offer" else "select"
                with patch.object(self.store, method, side_effect=RuntimeError("injected crash")):
                    with self.assertRaisesRegex(RuntimeError, "injected crash"):
                        runner.run(sid)
                calls_before = transport.calls
                self.reopen()
                runner = self.runner(controller, 2)
                runner.run(sid)
                self.assertEqual(self.offers(sid)[0], [asdict(c) for c in expected])
                self.assertEqual(len(self.store.trials(sid)), 2)
                if boundary == "validated_select":
                    self.assertEqual(transport.calls, calls_before)
                    self.assertEqual(self.store.decision_attempts(sid)[0]["status"], "selected")

    def test_crash_after_worker_result_does_not_retrain(self):
        runner = self.runner(width=2)
        sid = runner.start(3, 7)
        runner.run(sid, 1)
        original = self.store.finish

        def finish_then_crash(trial_id, result):
            original(trial_id, result)
            raise RuntimeError("injected crash after result")

        with patch.object(self.store, "finish", side_effect=finish_then_crash):
            with self.assertRaisesRegex(RuntimeError, "injected crash after result"):
                runner.run(sid)
        self.assertEqual([row["status"] for row in self.store.trials(sid)],
                         ["completed", "completed"])
        self.reopen()
        self.runner(width=2).run(sid)
        self.assertEqual(len(self.store.trials(sid)), 3)
        self.assertEqual(len(self.store.offers(sid)), 2)

    def test_crash_after_atomic_selection_runs_saved_pending_trial(self):
        runner = self.runner(width=2)
        sid = runner.start(2, 7)
        runner.run(sid, 1)
        original = self.store.select

        def select_then_crash(*args):
            original(*args)
            raise RuntimeError("injected crash after selection")

        with patch.object(self.store, "select", side_effect=select_then_crash):
            with self.assertRaisesRegex(RuntimeError, "injected crash after selection"):
                runner.run(sid)
        self.assertEqual(self.store.trials(sid)[-1]["status"], "pending")
        saved = self.offers(sid)[0]
        self.reopen()
        runner = self.runner(width=2)
        with patch.object(runner.generator, "generate", side_effect=AssertionError("redrew offer")):
            runner.run(sid)
        self.assertEqual(self.offers(sid)[0], saved)
        self.assertEqual([row["status"] for row in self.store.trials(sid)],
                         ["completed", "completed"])

    def test_crash_before_result_commit_records_interruption_once(self):
        runner = self.runner(width=2)
        sid = runner.start(3, 7)
        runner.run(sid, 1)
        with patch.object(self.store, "finish", side_effect=RuntimeError("injected commit crash")):
            with self.assertRaisesRegex(RuntimeError, "injected commit crash"):
                runner.run(sid)
        self.assertEqual(self.store.trials(sid)[-1]["status"], "running")
        self.reopen()
        self.runner(width=2).run(sid)
        rows = self.store.proposal_history(sid)
        self.assertEqual([row["status"] for row in rows],
                         ["completed", "interrupted", "completed"])
        self.assertEqual(rows[2]["proposal"]["completed_observations"], 1)
        self.assertEqual(len(self.store.offers(sid)), 2)

    def test_cli_rejects_invalid_pool_pairings(self):
        base = ["cifar-run", "--data-dir", str(self.root / "data"),
                "--run-dir", str(self.root / "cli"), "--budget", "2",
                "--proposal-strategy", "tpe-pool", "--proposal-domain", "cifar-mixed-v1"]
        for extra in (("--candidate-limit", "1"),
                      ("--proposal-domain", "cifar-sgd-numeric-v1"),
                      ("--candidate-limit", "9")):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(base + list(extra))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(base + ["--controller", "jev", "--proposal-strategy", "tpe"])

    def test_comparison_checks_pool_width_only_for_same_pool_strategy(self):
        base = {"settings": {"task": "cifar10_fixture", "protocol": "fixture-v1",
                             "proposal_strategy": "tpe-pool", "proposal_domain": "cifar-mixed-v1",
                             "candidate_limit": 2, "task_details": {}},
                "source": {"digest": "same"}}
        other = {"settings": {**base["settings"], "candidate_limit": 8},
                 "source": {"digest": "same"}}
        self.assertEqual(protocol_differences(base, other), ["candidate_limit"])
        other["settings"]["proposal_strategy"] = "tpe"
        self.assertEqual(protocol_differences(base, other), [])

    def test_fixture_study_reports_pool_and_single_tpe_arms(self):
        try:
            import torch
            import torchvision
        except ImportError:
            self.skipTest("vision dependencies are optional")
        spec_path = self.root / "study.json"
        spec_path.write_text(json.dumps({
            "version": 1, "task": "cifar10_fixture", "protocol": "fixture-v1",
            "data_dir": str(self.root / "data"), "output_root": str(self.root / "study"),
            "trial_budget": 2, "active_time_budget_seconds": 300,
            "candidate_limit": 2, "seeds": [7],
            "arms": [
                {"id": "jev_tpe_pool", "controller": "jev", "proposal_strategy": "tpe-pool",
                 "proposal_domain": "cifar-mixed-v1"},
                {"id": "random_tpe_pool", "controller": "random", "proposal_strategy": "tpe-pool",
                 "proposal_domain": "cifar-mixed-v1"},
                {"id": "single_tpe", "controller": "single", "proposal_strategy": "tpe",
                 "proposal_domain": "cifar-mixed-v1"},
            ]}))
        with patch("jevresearch.core.study_runner.live_controller",
                   side_effect=lambda *args: JevController(FakeJev())):
            study_id = StudyRunner(spec_path).run(max_new_trials=6)
        study_store = Store(self.root / "study" / "study.sqlite")
        try:
            report = study_report(study_store, study_id)
            self.assertTrue(report["compatible"])
            members = {m["arm_id"]: m for m in report["members"]}
            self.assertEqual({m["status"] for m in members.values()}, {"stopped"})
            self.assertEqual({m["training_attempts"] for m in members.values()}, {2})
            jev = members["jev_tpe_pool"]
            self.assertEqual(jev["pool_policy"], "tpe-pool-v1")
            self.assertEqual(jev["proposal_summary"]["declined_untrained"], 1)
            self.assertEqual(jev["logical_api_attempts"], 1)
            self.assertEqual(jev["offers"][0]["selected_trial"]["status"], "completed")
            self.assertEqual([{k: v for k, v in c.items() if k != "audit_status"}
                              for c in jev["offers"][0]["candidates"]],
                             [{k: v for k, v in c.items() if k != "audit_status"}
                              for c in members["random_tpe_pool"]["offers"][0]["candidates"]])
            self.assertEqual(members["single_tpe"]["proposal_summary"]["accepted"], 1)
        finally:
            study_store.close()


if __name__ == "__main__":
    unittest.main()
