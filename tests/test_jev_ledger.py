import json
import tempfile
import unittest
from pathlib import Path

from jevresearch.controllers.jev import JevController, TransportFailure
from jevresearch.controllers.random import RandomController
from jevresearch.core.candidate_generator import trial_seed
from jevresearch.core.runner import Runner
from jevresearch.storage.history import Store
from jevresearch.tasks.synthetic import SyntheticTask


class FakeJevTransport:
    kind = "fake"
    sdk_version = None
    timeout = 1
    max_retries = 0

    def __init__(self):
        self.calls = []
        self.mode = "success"

    def invoke(self, body):
        self.calls.append(body)
        if self.mode == "timeout":
            raise TransportFailure("timeout", True)
        options = list(body["questions"]["next_trial"]["criteria"])
        choice = "invalid-id" if self.mode == "invalid" else options[0]
        probabilities = {id_: float(id_ == options[0]) for id_ in options}
        if self.mode == "bad_distribution":
            probabilities[options[0]] = 0.5
        return {"model": "jev-1.13.0", "answers": {"next_trial": {
            "type": "choice", "choice": choice, "confidence": 1.0,
            "probabilities": probabilities}},
            "usage": {"input_tokens": 40, "output_tokens": 2}}


class JevLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "history.sqlite")
        self.addCleanup(self.store.close)
        self.transport = FakeJevTransport()
        self.controller = JevController(self.transport, max_calls=1)
        self.runner = Runner(self.store, SyntheticTask(), self.controller)

    def test_success_records_request_response_and_selected_trial(self):
        sid = self.runner.start(2, 7)
        self.runner.run(sid)
        trials = self.store.trials(sid)
        attempts = self.store.decision_attempts(sid)
        self.assertEqual([t["status"] for t in trials], ["completed", "completed"])
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["status"], "selected")
        self.assertEqual(attempts[0]["selected_id"], trials[1]["candidate_id"])
        self.assertEqual(json.loads(attempts[0]["request"])["body"], self.transport.calls[0])
        self.assertEqual(json.loads(attempts[0]["response"])["usage"]["input_tokens"], 40)
        self.assertEqual(self.store.session(sid)["stop_reason"], "budget exhausted")

    def test_timeout_pauses_without_trial_then_retries_same_offer(self):
        sid = self.runner.start(2, 7)
        self.transport.mode = "timeout"
        self.runner.run(sid)
        first = self.store.decision_attempts(sid)[0]
        self.assertEqual(self.store.session(sid)["status"], "paused")
        self.assertEqual(first["status"], "unknown")
        self.assertEqual(len(self.store.trials(sid)), 1)
        self.assertIsNotNone(self.store.outstanding(sid))
        self.transport.mode = "success"
        self.runner.run(sid)
        attempts = self.store.decision_attempts(sid)
        self.assertEqual([a["status"] for a in attempts], ["unknown", "selected"])
        self.assertEqual(attempts[0]["request"], attempts[1]["request"])
        self.assertEqual(len(self.store.trials(sid)), 2)

    def test_bad_choice_or_distribution_never_selects(self):
        for mode in ("invalid", "bad_distribution"):
            with self.subTest(mode=mode):
                sid = self.runner.start(2, 8)
                self.transport.mode = mode
                self.runner.run(sid)
                self.assertEqual(len(self.store.trials(sid)), 1)
                self.assertEqual(self.store.session(sid)["status"], "paused")
                self.assertEqual(self.store.decision_attempts(sid)[-1]["error_code"], "invalid_response")

    def test_validated_choice_recovers_without_second_call(self):
        sid = self.runner.start(2, 9)
        self.runner.run(sid, 1)
        state = self.store.state(sid)
        candidates = self.runner.generator.generate(self.runner.task, state, 9,
                                                     json.loads(self.store.session(sid)["source"])["digest"])
        oid = self.store.save_offer(sid, state, candidates, self.store.session(sid)["rng_state"])
        offer = self.store.offer(sid, oid)
        prepared = self.controller.prepare(state, candidates, "max")
        aid = self.store.begin_decision(sid, oid, prepared, self.controller.details())
        raw = self.transport.invoke(prepared["body"])
        validated = self.controller.validate(prepared, raw)
        self.store.validate_decision(aid, validated.response, validated.selected_id, 0.1)
        self.assertEqual(len(self.transport.calls), 1)
        self.runner.run(sid)
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(self.store.decision_attempts(sid)[0]["status"], "selected")
        self.assertEqual(len(self.store.trials(sid)), 2)

    def test_committed_pending_trial_does_not_call_again(self):
        sid = self.runner.start(2, 12)
        self.runner.run(sid, 1)
        state = self.store.state(sid)
        source = json.loads(self.store.session(sid)["source"])["digest"]
        candidates = self.runner.generator.generate(self.runner.task, state, 12, source)
        oid = self.store.save_offer(sid, state, candidates, self.store.session(sid)["rng_state"])
        offer = self.store.offer(sid, oid)
        self.runner._audited_select(sid, offer, candidates, state, "max", 0)
        self.assertEqual(self.store.trials(sid)[1]["status"], "pending")
        self.runner.run(sid)
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(self.store.trials(sid)[1]["status"], "completed")

    def test_call_cap_pauses_with_an_outstanding_offer(self):
        sid = self.runner.start(3, 11)
        self.runner.run(sid)
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(len(self.store.trials(sid)), 2)
        self.assertEqual(self.store.session(sid)["status"], "paused")
        self.assertIsNotNone(self.store.outstanding(sid))

    def test_random_and_jev_receive_same_first_offer(self):
        seed = 17
        random_runner = Runner(self.store, SyntheticTask(), RandomController())
        jev_sid = self.runner.start(2, seed)
        random_sid = random_runner.start(2, seed)
        self.runner.run(jev_sid, 1)
        random_runner.run(random_sid, 1)
        jev_state = self.store.state(jev_sid)
        random_state = self.store.state(random_sid)
        source = json.loads(self.store.session(jev_sid)["source"])["digest"]
        jev_offer = self.runner.generator.generate(self.runner.task, jev_state, seed, source)
        random_offer = random_runner.generator.generate(random_runner.task, random_state, seed, source)
        self.assertEqual([c.id for c in jev_offer], [c.id for c in random_offer])

    def test_started_call_becomes_unknown_on_resume(self):
        sid = self.runner.start(2, 3)
        self.runner.run(sid, 1)
        state = self.store.state(sid)
        candidates = self.runner.generator.generate(self.runner.task, state, 3,
                                                     json.loads(self.store.session(sid)["source"])["digest"])
        oid = self.store.save_offer(sid, state, candidates, self.store.session(sid)["rng_state"])
        prepared = self.controller.prepare(state, candidates, "max")
        self.store.begin_decision(sid, oid, prepared, self.controller.details())
        self.runner.run(sid)
        self.assertEqual([a["status"] for a in self.store.decision_attempts(sid)],
                         ["unknown", "selected"])
        self.assertEqual(len(self.store.trials(sid)), 2)

    def test_training_failure_is_separate_from_controller_failure(self):
        seed = 4
        runner = Runner(self.store, SyntheticTask(fail_seed=trial_seed(seed, 1)), self.controller)
        sid = runner.start(2, seed)
        runner.run(sid)
        self.assertEqual([t["status"] for t in self.store.trials(sid)], ["completed", "failed"])
        self.assertEqual(self.store.decision_attempts(sid)[0]["status"], "selected")


if __name__ == "__main__":
    unittest.main()
