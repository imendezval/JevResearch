import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from jevresearch.controller import RandomController
from jevresearch.core import ExperimentResult, ExperimentSpec, better
from jevresearch.runner import Runner, generate, trial_seed
from jevresearch.source import identity
from jevresearch.storage import InvariantError, Store
from jevresearch.synthetic import SyntheticTask


class TinyTask:
    name = "tiny"
    protocol = "v1"
    data_split = "fixed"
    eval_budget = 1
    objective_direction = "min"

    def baseline(self):
        return {"n": 2}

    def validate(self, config):
        if set(config) != {"n"} or type(config["n"]) is not int or not 0 <= config["n"] <= 2:
            raise ValueError("invalid")

    def proposals(self, state):
        n = state.incumbent_config["n"]
        return [("decrement", {}, {"n": n - 1}),
                ("duplicate", {}, {"n": n - 1}),
                ("unchanged", {}, {"n": n})]

    def evaluate(self, spec):
        return ExperimentResult("completed", float(spec.config["n"]), {}, 0.0)


class Phase1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "test.sqlite")
        self.addCleanup(self.store.close)

    def runner(self, task=None):
        return Runner(self.store, task or SyntheticTask(), RandomController())

    def test_spec_identity_and_seed_schedule(self):
        a = ExperimentSpec("t", "v1", {"b": 2, "a": 1}, "split", 10, 1, "source")
        b = replace(a, config={"a": 1, "b": 2})
        c = replace(a, seed=2)
        self.assertEqual(a.fingerprint, b.fingerprint)
        self.assertNotEqual(a.fingerprint, c.fingerprint)
        self.assertEqual(a.config_key, c.config_key)
        self.assertNotEqual(trial_seed(4, 0), trial_seed(4, 1))
        with self.assertRaises(TypeError):
            a.config["a"] = 3
        with self.assertRaises(TypeError):
            a.config |= {"c": 4}
        nested = replace(a, config={"layers": [1, {"width": 2}]})
        with self.assertRaises(TypeError):
            nested.config["layers"][1]["width"] = 4

    def test_candidate_mapping_dedup_and_second_task(self):
        runner = self.runner(TinyTask())
        sid = runner.start(5, 3)
        runner.run(sid, 1)
        state = self.store.state(sid)
        offered = generate(runner.task, state, 3, identity(runner.task)["digest"])
        self.assertEqual(len(offered), 1)
        self.assertEqual(offered[0].spec.config, {"n": 1})
        self.assertEqual(offered[0].config, offered[0].spec.config)
        self.assertEqual(offered[0].spec.parent_id, state.best_trial_id)
        self.assertEqual(offered[0].spec.seed, trial_seed(3, 1))
        runner.run(sid)
        self.assertEqual(self.store.session(sid)["stop_reason"], "no novel candidates")
        self.assertEqual(self.store.state(sid).attempted, 3)
        self.assertEqual(self.store.state(sid).best_objective, 0)

    def test_objective_direction_tie_and_invalid(self):
        self.assertTrue(better(1, 2, "min"))
        self.assertFalse(better(2, 2, "min"))
        self.assertTrue(better(3, 2, "max"))
        class Tied(TinyTask):
            def evaluate(self, spec):
                return ExperimentResult("completed", 1.0, {}, 0)
        sid = self.runner(Tied()).start(3, 2)
        self.runner(Tied()).run(sid)
        self.assertEqual(self.store.state(sid).best_trial_id, 0)
        class Nonfinite(TinyTask):
            def evaluate(self, spec):
                return ExperimentResult("completed", math.nan, {}, 0)
        sid2 = self.runner(Nonfinite()).start(2, 2)
        self.runner(Nonfinite()).run(sid2)
        self.assertIsNone(self.store.state(sid2).best_trial_id)
        self.assertEqual(self.store.session(sid2)["stop_reason"], "baseline failed")
        class BadSecondary(TinyTask):
            def evaluate(self, spec):
                return ExperimentResult("completed", 1.0, {"bad": math.inf}, 0)
        sid3 = self.runner(BadSecondary()).start(2, 2)
        self.runner(BadSecondary()).run(sid3)
        self.assertEqual(self.store.trials(sid3)[0]["status"], "failed")

    def test_failed_attempt_and_baseline_failure(self):
        seed = 8
        sid = self.runner(SyntheticTask(fail_seed=trial_seed(seed, 1))).start(3, seed)
        self.runner(SyntheticTask(fail_seed=trial_seed(seed, 1))).run(sid)
        rows = self.store.trials(sid)
        self.assertEqual([r["status"] for r in rows], ["completed", "failed", "completed"])
        self.assertEqual(json.loads(rows[1]["result"])["error_type"], "RuntimeError")
        self.assertEqual([r["number"] for r in rows], [0, 1, 2])
        sid2 = self.runner(SyntheticTask(fail_seed=trial_seed(seed, 0))).start(3, seed)
        self.runner(SyntheticTask(fail_seed=trial_seed(seed, 0))).run(sid2)
        self.assertEqual(self.store.state(sid2).attempted, 1)
        self.assertIsNone(self.store.state(sid2).best_trial_id)

    def test_resume_offer_pending_running(self):
        runner = self.runner()
        sid = runner.start(4, 5)
        runner.run(sid, 1)
        state = self.store.state(sid)
        offer = generate(runner.task, state, 5, identity()["digest"])
        oid = self.store.save_offer(sid, state, offer, self.store.session(sid)["rng_state"])
        saved = self.store.outstanding(sid)
        self.assertEqual(saved["id"], oid)
        runner.run(sid, 1)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM offers WHERE session_id=?", (sid,)).fetchone()[0], 1)
        self.assertEqual(self.store.trials(sid)[1]["candidate_id"],
                         self.store.db.execute("SELECT selected_id FROM offers WHERE id=?", (oid,)).fetchone()[0])

        sid2 = runner.start(3, 6)
        runner.run(sid2, 1)
        state2 = self.store.state(sid2)
        offer2 = generate(runner.task, state2, 6, identity()["digest"])
        oid2 = self.store.save_offer(sid2, state2, offer2, self.store.session(sid2)["rng_state"])
        self.store.select(sid2, oid2, offer2[0], self.store.session(sid2)["rng_state"])
        runner.run(sid2, 1)
        self.assertEqual(self.store.state(sid2).attempted, 2)
        self.assertEqual(self.store.trials(sid2)[1]["status"], "completed")
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM offers WHERE session_id=?", (sid2,)).fetchone()[0], 1)

        sid3 = runner.start(3, 7)
        runner.run(sid3, 2)
        trial = self.store.trials(sid3)[1]
        self.store.db.execute("UPDATE trials SET status='running',result=NULL WHERE id=?", (trial["id"],))
        self.store.db.commit()
        runner.run(sid3)
        self.assertEqual(self.store.state(sid3).attempted, 3)
        self.assertEqual(self.store.trials(sid3)[1]["status"], "interrupted")
        self.assertEqual(len({r["fingerprint"] for r in self.store.trials(sid3)}), 3)

    def test_pause_resume_matches_fresh(self):
        runner = self.runner()
        full = runner.start(6, 7)
        runner.run(full)
        paused = runner.start(6, 7)
        runner.run(paused, 2)
        self.assertEqual(self.store.state(paused).attempted, 2)
        reopened = Store(Path(self.tmp.name) / "test.sqlite")
        try:
            Runner(reopened, SyntheticTask(), RandomController()).run(paused)
        finally:
            reopened.close()
        def signature(sid):
            return [(r["candidate_id"], r["fingerprint"], r["status"],
                     json.loads(r["result"])["objective"]) for r in self.store.trials(sid)]
        self.assertEqual(signature(full), signature(paused))
        self.assertEqual(self.store.session(paused)["stop_reason"], "budget exhausted")

    def test_session_compatibility_and_uniqueness(self):
        runner = self.runner()
        sid = runner.start(3, 1)
        with patch("jevresearch.runner.identity", return_value={"digest": "changed"}):
            with self.assertRaises(InvariantError):
                runner.run(sid)
        with self.assertRaises(InvariantError):
            self.runner(TinyTask()).run(sid)
        runner.run(sid, 1)
        state = self.store.state(sid)
        offer = generate(runner.task, state, 1, identity()["digest"])
        oid = self.store.save_offer(sid, state, offer, self.store.session(sid)["rng_state"])
        self.store.select(sid, oid, offer[0], self.store.session(sid)["rng_state"])
        with self.assertRaises(InvariantError):
            self.store.select(sid, oid, offer[0], self.store.session(sid)["rng_state"])
        self.assertEqual(self.store.state(sid).attempted, 2)

    def test_duplicate_config_with_new_seed_is_invariant_error(self):
        runner = self.runner()
        sid = runner.start(3, 2)
        runner.run(sid, 1)
        state = self.store.state(sid)
        offered = generate(runner.task, state, 2, identity()["digest"])
        oid = self.store.save_offer(sid, state, offered, self.store.session(sid)["rng_state"])
        baseline = ExperimentSpec(**json.loads(self.store.trials(sid)[0]["spec"]))
        repeated = replace(offered[0], spec=replace(baseline, seed=baseline.seed + 1))
        # Even when a corrupt offer presents a repeat under a different seed,
        # the unique config key prevents a second attempted trial.
        with self.store.db:
            import dataclasses
            self.store.db.execute("UPDATE offers SET candidates=? WHERE id=?",
                                  (json.dumps([dataclasses.asdict(repeated)]), oid))
        with self.assertRaises(InvariantError):
            self.store.select(sid, oid, repeated, self.store.session(sid)["rng_state"])
        self.assertEqual(self.store.state(sid).attempted, 1)
        self.assertIsNone(self.store.outstanding(sid)["selected_id"])

    def test_interruption_costs_one_attempt_against_fresh(self):
        runner = self.runner()
        fresh = runner.start(4, 11)
        runner.run(fresh)
        sid = runner.start(4, 11)
        runner.run(sid, 1)
        state = self.store.state(sid)
        candidates = generate(runner.task, state, 11, identity()["digest"])
        oid = self.store.save_offer(sid, state, candidates, self.store.session(sid)["rng_state"])
        selected, next_rng = runner.controller.select(state, candidates, self.store.session(sid)["rng_state"])
        self.store.select(sid, oid, next(c for c in candidates if c.id == selected), next_rng)
        self.store.running(self.store.trials(sid)[1]["id"])
        runner.run(sid)
        self.assertEqual(self.store.state(sid).attempted, 4)
        self.assertEqual(len([r for r in self.store.trials(sid) if r["status"] == "completed"]), 3)
        self.assertEqual(len([r for r in self.store.trials(fresh) if r["status"] == "completed"]), 4)
        self.assertEqual(len({r["fingerprint"] for r in self.store.trials(sid)}), 4)


if __name__ == "__main__":
    unittest.main()
