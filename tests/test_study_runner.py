import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jevresearch.core.study_runner import StudyRunner
from jevresearch.core.study_report import study_report
from jevresearch.storage.history import Store
from jevresearch.storage.history import InvariantError


class StudyRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "study.json"
        self.raw = {"version": 1, "task": "cifar10_fixture", "protocol": "fixture-v1",
                    "data_dir": str(self.root / "data"),
                    "output_root": str(self.root / "study"),
                    "trial_budget": 1, "active_time_budget_seconds": 60,
                    "trial_timeout_seconds": 30, "batch_size": 16,
                    "arms": [{"id": "random_a", "controller": "random"},
                             {"id": "random_b", "controller": "random"}],
                    "seeds": [7, 11]}

    def runner(self):
        self.path.write_text(json.dumps(self.raw))
        return StudyRunner(self.path)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch") and
                         __import__("importlib").util.find_spec("torchvision"),
                         "vision dependencies are optional")
    def test_registry_idempotence_pause_resume_report_and_checkpoint(self):
        self.raw["checkpoint_policy"] = "best"
        self.raw["compute_hourly_usd"] = 1.0
        runner = self.runner()
        study_id = runner.run(max_new_trials=1)
        store = Store(runner.root / "study.sqlite")
        self.assertEqual(len(store.study_members(study_id)), 1)
        first_sid = store.study_members(study_id)[0]["session_id"]
        self.assertEqual(len(store.trials(first_sid)), 1)
        with store.db:
            store.db.execute("UPDATE sessions SET created_at=created_at-3600 WHERE id=?", (first_sid,))
        runner.run()
        runner.run()
        members = store.study_members(study_id)
        self.assertEqual(len(members), 4)
        self.assertEqual(members[0]["session_id"], first_sid)
        report = study_report(store, study_id)
        self.assertTrue(report["compatible"])
        self.assertEqual(len(report["aggregate_table"]), 2)
        self.assertTrue(all(m["trajectory"][0]["observed_active_seconds"] > 0 for m in report["members"]))
        first = next(m for m in report["members"] if m["session_id"] == first_sid)
        self.assertGreater(first["trajectory"][0]["calendar_elapsed_seconds"],
                           first["trajectory"][0]["observed_active_seconds"] + 3000)
        self.assertEqual(report["members"][0]["compute_cost"]["kind"], "estimate")
        self.assertEqual(report["members"][0]["jev_cost"]["value_usd"], None)
        checkpoint = store.checkpoints(first_sid)[0]
        self.assertEqual(checkpoint["status"], "retained")
        self.assertTrue((runner.root / checkpoint["path"]).is_file())
        from jevresearch.tasks.vision.cifar10.worker import validate_checkpoint
        input_path = (runner.root / checkpoint["path"]).with_name("input.json")
        repeated = validate_checkpoint(json.loads(input_path.read_text()),
                                       runner.root / checkpoint["path"])
        recorded = json.loads(store.trials(first_sid)[0]["result"])["objective"]
        self.assertAlmostEqual(repeated, recorded, places=7)
        store.close()
        self.raw["trial_budget"] = 2
        with self.assertRaises(InvariantError):
            self.runner().run()

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch") and
                         __import__("importlib").util.find_spec("torchvision"),
                         "vision dependencies are optional")
    def test_only_best_checkpoint_is_retained(self):
        self.raw["arms"] = [{"id": "random", "controller": "random"}]
        self.raw["seeds"] = [7]
        self.raw["trial_budget"] = 2
        self.raw["checkpoint_policy"] = "best"
        runner = self.runner()
        study_id = runner.run()
        store = Store(runner.root / "study.sqlite")
        sid = store.study_members(study_id)[0]["session_id"]
        from jevresearch.cli import history
        exported = history(store, sid)
        reported = study_report(store, study_id)["members"][0]["trajectory"]
        self.assertEqual([row["best_so_far"] for row in exported["trials"]],
                         [row["best_objective"] for row in reported])
        checkpoints = store.checkpoints(sid)
        self.assertEqual(sorted(row["status"] for row in checkpoints), ["pruned", "retained"])
        for row in checkpoints:
            self.assertEqual((runner.root / row["path"]).exists(), row["status"] == "retained")
        store.close()

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch") and
                         __import__("importlib").util.find_spec("torchvision"),
                         "vision dependencies are optional")
    def test_unknown_crash_interval_blocks_new_trial(self):
        self.raw["arms"] = [{"id": "random", "controller": "random"}]
        self.raw["seeds"] = [7]
        self.raw["trial_budget"] = 2
        self.raw["active_time_budget_seconds"] = 30
        runner = self.runner()
        study_id = runner.run(max_new_trials=1)
        store = Store(runner.root / "study.sqlite")
        sid = store.study_members(study_id)[0]["session_id"]
        trial = store.trials(sid)[0]
        with store.db:
            store.db.execute("UPDATE trials SET status='running',result=NULL WHERE id=?",
                             (trial["id"],))
        activity_id = store.start_activity(study_id, sid)
        with store.db:
            store.db.execute("UPDATE study_activity SET started_at=started_at-60 WHERE id=?",
                             (activity_id,))
        runner.run()
        self.assertEqual(len(store.trials(sid)), 1)
        self.assertEqual(store.trials(sid)[0]["status"], "completed")
        self.assertEqual(store.session(sid)["status"], "paused")
        self.assertIn("unknown interval", store.session(sid)["stop_reason"])
        self.assertEqual(study_report(store, study_id)["members"][0]["unknown_active_intervals"], 1)
        store.close()

    def test_missing_jev_key_is_blocked_without_fake_session(self):
        self.raw["arms"] = [{"id": "jev", "controller": "jev"}]
        runner = self.runner()
        with patch.dict("os.environ", {"TYPESAFE_API_KEY": ""}):
            study_id = runner.run()
        store = Store(runner.root / "study.sqlite")
        member = store.study_members(study_id)[0]
        self.assertIsNone(member["session_id"])
        self.assertIn("TYPESAFE_API_KEY", member["blocked_reason"])
        self.assertEqual(study_report(store, study_id)["members"][0]["status"], "blocked")
        store.close()

    def test_report_rejects_incompatible_protocol(self):
        # The report checks recorded campaign settings, including data/split,
        # rather than assuming all registry members are comparable.
        store = Store(self.root / "study.sqlite")
        try:
            study_id = store.register_study(self.runner().spec.normalized(), "fingerprint", "source", {})
            for arm_id, split in (("random_a", "one"), ("random_b", "two")):
                settings = {"task": "cifar10_fixture", "protocol": "fixture-v1",
                            "data_split": split, "eval_budget": 1, "direction": "max",
                            "seed_schedule": "sha256(seed:index)", "budget": 1,
                            "candidate_limit": 8, "task_details": {"split_sha256": split}}
                from jevresearch.core.experiment import ExperimentSpec
                spec = ExperimentSpec("cifar10_fixture", "fixture-v1", {"x": 1}, split, 1, 1, "source")
                sid = store.create(settings, {"digest": "source"}, "rng", spec,
                                   (study_id, arm_id, 7))
                if arm_id == "random_a":
                    from jevresearch.core.result import ExperimentResult
                    trial_id = store.trials(sid)[0]["id"]
                    store.running(trial_id)
                    store.finish(trial_id, ExperimentResult("failed", None, {}, 0.1,
                                                            "FixtureFailure", "deliberate"))
            report = study_report(store, study_id)
            self.assertFalse(report["compatible"])
            self.assertIsNone(report["aggregate_table"])
            self.assertIn("data_split", str(report["compatibility_differences"]))
            failed = next(m for m in report["members"] if m["arm_id"] == "random_a")
            self.assertEqual(failed["trajectory"][0]["status"], "failed")
        finally:
            store.close()
