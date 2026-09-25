import json
import tempfile
import unittest
from pathlib import Path

from jevresearch.core.study import StudySpec
from jevresearch.storage.history import InvariantError, Store


class StudySpecTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "study.json"
        self.raw = {"version": 1, "task": "cifar10_fixture", "protocol": "fixture-v1",
                    "data_dir": self.tmp.name,
                    "output_root": str(Path(self.tmp.name) / "output"),
                    "trial_budget": 2, "active_time_budget_seconds": 30,
                    "arms": [{"id": "random_a", "controller": "random"},
                             {"id": "jev", "controller": "jev"}], "seeds": [7, 11]}

    def load(self, raw=None):
        self.path.write_text(json.dumps(self.raw if raw is None else raw))
        return StudySpec.load(self.path)

    def test_strict_fields_defaults_overrides_and_fingerprint(self):
        spec, overrides = self.load()
        self.assertEqual(spec.candidate_limit, 8)
        self.assertEqual(spec.arms[1].model, "jev-1.13.0")
        self.assertEqual(spec.fingerprint("source"), self.load()[0].fingerprint("source"))
        self.assertNotEqual(spec.fingerprint("source"), spec.fingerprint("changed"))
        overridden, overrides = StudySpec.load(self.path, data_dir=self.tmp.name + "/other")
        self.assertIn("data_dir", overrides)
        self.assertNotEqual(spec.fingerprint("source"), overridden.fingerprint("source"))
        for change in ({"extra": 1}, {"version": 2}, {"seeds": [7, 7]},
                       {"arms": [{"id": "same", "controller": "random"},
                                  {"id": "same", "controller": "random"}]},
                       {"device": "invalid"}, {"protocol": "wrong"}, {"trial_budget": 0},
                       {"arms": [{"id": "r", "controller": "random", "model": "x"}]},
                       {"arms": [{"id": 3, "controller": "random"}]},
                       {"arms": [{"id": "j", "controller": "jev", "model": "jev-latest"}]}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.load({**self.raw, **change})

    def test_registry_refuses_changed_spec_and_migration_keeps_history(self):
        spec, overrides = self.load()
        store = Store(Path(self.tmp.name) / "history.sqlite")
        try:
            study_id = store.register_study(spec.normalized(), spec.fingerprint("source"), "source", overrides)
            self.assertEqual(study_id, store.register_study(
                spec.normalized(), spec.fingerprint("source"), "source", overrides))
            with self.assertRaises(InvariantError):
                store.register_study(spec.normalized(), spec.fingerprint("different"), "different", overrides)
            self.assertEqual(store.db.execute("PRAGMA user_version").fetchone()[0], 5)
        finally:
            store.close()

    def test_global_arm_pairings_and_categorical_cma_rejected(self):
        valid = [
            {"id": "global_random", "controller": "single", "proposal_strategy": "global-random",
             "proposal_domain": "cifar-mixed-v1"},
            {"id": "tpe", "controller": "single", "proposal_strategy": "tpe",
             "proposal_domain": "cifar-mixed-v1"},
            {"id": "pool", "controller": "random", "proposal_strategy": "global-pool",
             "proposal_domain": "cifar-mixed-v1"},
            {"id": "jev_pool", "controller": "jev", "proposal_strategy": "global-pool",
             "proposal_domain": "cifar-mixed-v1", "model": "jev-1.13.0", "api_timeout": 4.0,
             "sdk_retries": 0, "max_api_calls": 2},
            {"id": "tpe_random_pool", "controller": "random", "proposal_strategy": "tpe-pool",
             "proposal_domain": "cifar-mixed-v1"},
            {"id": "tpe_jev_pool", "controller": "jev", "proposal_strategy": "tpe-pool",
             "proposal_domain": "cifar-mixed-v1"},
            {"id": "cma", "controller": "single", "proposal_strategy": "cmaes",
             "proposal_domain": "cifar-sgd-numeric-v1"},
        ]
        parsed = self.load({**self.raw, "arms": valid})[0].arms
        self.assertEqual(len(parsed), 7)
        self.assertEqual((parsed[3].model, parsed[3].api_timeout, parsed[3].sdk_retries,
                          parsed[3].max_api_calls, parsed[3].proposal_strategy, parsed[3].proposal_domain),
                         ("jev-1.13.0", 4.0, 0, 2, "global-pool", "cifar-mixed-v1"))
        for arm in ({**valid[-1], "proposal_domain": "cifar-mixed-v1"},
                    {**valid[1], "controller": "jev"},
                    {**valid[2], "controller": "single"},
                    {"id": "local_single", "controller": "single"},
                    {**valid[0], "controller": "random"},
                    {**valid[4], "controller": "single"},
                    {**valid[4], "proposal_domain": "cifar-sgd-numeric-v1"}):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                self.load({**self.raw, "arms": [arm]})
        with self.assertRaises(ValueError):
            self.load({**self.raw, "candidate_limit": 1, "arms": [valid[4]]})
