import importlib.util
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

from jevresearch.controllers.jev import (JevController, InvalidJevResponse,
                                         MAX_REQUEST_BYTES, TypeSafeSDKTransport)
from jevresearch.core import Candidate, ExperimentSpec, SearchState, canonical
from jevresearch.core.candidate_generator import CandidateGenerator
from jevresearch.tasks.synthetic import SyntheticTask


class FakeTransport:
    kind = "fake"
    sdk_version = None
    timeout = 1
    max_retries = 0

    def __init__(self, response=None):
        self.response = response
        self.requests = []

    def invoke(self, body):
        self.requests.append(body)
        return self.response


def offer():
    task = SyntheticTask()
    baseline_key = ExperimentSpec(task.name, task.protocol, task.baseline(),
                                  task.data_split, task.eval_budget, 0, "source").config_key
    state = SearchState(1, 1, 3, 0, -5.0, task.baseline(),
                        ({"trial_id": 0, "status": "completed", "objective": -5.0,
                          "config_key": baseline_key},))
    return state, CandidateGenerator().generate(task, state, 7, "source")


def answer(prepared, choice=None):
    ids = list(prepared["body"]["questions"]["next_trial"]["criteria"])
    choice = choice or ids[0]
    return {"model": "jev-1.13.0", "answers": {"next_trial": {"type": "choice",
            "choice": choice, "confidence": 1.0,
            "probabilities": {id_: float(id_ == choice) for id_ in ids}}},
            "usage": {"input_tokens": 42, "output_tokens": 3}}


class JevAdapterTests(unittest.TestCase):
    def test_exact_option_mapping_and_bounded_state(self):
        state, candidates = offer()
        controller = JevController(FakeTransport())
        prepared = controller.prepare(state, candidates, "max")
        body = prepared["body"]
        self.assertEqual(body["model"], "jev-1.13.0")
        self.assertEqual(body["questions"]["next_trial"]["type"], "choice")
        self.assertEqual(list(body["questions"]["next_trial"]["criteria"]), [c.id for c in candidates])
        self.assertEqual(body["state"]["best_completed"]["objective"], -5.0)
        self.assertEqual(body["state"]["remaining_trial_slots"], 2)
        self.assertLessEqual(len(canonical(body).encode()), MAX_REQUEST_BYTES)
        for candidate in candidates:
            description = body["questions"]["next_trial"]["criteria"][candidate.id]
            self.assertIn(candidate.operator, description)
            self.assertIn("full_config=", description)
        self.assertEqual(prepared, controller.prepare(state, candidates, "max"))

    def test_sensitive_fields_and_paths_are_not_sent(self):
        state, candidates = offer()
        original = candidates[0]
        config = {"api_key": "secret-value", "secret_count": 123456,
                  "dataset_path": "/private/data",
                  "message": "x" * 300}
        state = replace(state, incumbent_config={"secret_count": 1})
        candidate = replace(original, config=config, spec=replace(original.spec, config=config))
        prepared = JevController(FakeTransport()).prepare(state, (candidate,), "max")
        wire = canonical(prepared["body"])
        self.assertNotIn("secret-value", wire)
        self.assertNotIn("/private/data", wire)
        self.assertNotIn("x" * 300, wire)
        self.assertNotIn("123455", wire)
        self.assertIn("[redacted]", wire)
        self.assertGreater(prepared["truncation"]["redacted_fields"], 0)
        bad_config = {"/private/key": "value"}
        bad_candidate = replace(original, config=bad_config,
                                spec=replace(original.spec, config=bad_config))
        with self.assertRaisesRegex(ValueError, "absolute path"):
            JevController(FakeTransport()).prepare(state, (bad_candidate,), "max")

    def test_validate_choice_model_usage_and_distribution(self):
        state, candidates = offer()
        controller = JevController(FakeTransport())
        prepared = controller.prepare(state, candidates, "max")
        good = answer(prepared)
        validated = controller.validate(prepared, good)
        self.assertEqual(validated.selected_id, candidates[0].id)
        self.assertEqual(validated.response["usage"]["input_tokens"], 42)
        partial = answer(prepared)
        partial["usage"] = {"input_tokens": 42}
        self.assertEqual(controller.validate(prepared, partial).response["usage"],
                         {"input_tokens": 42})
        for change in (
            lambda r: r["answers"]["next_trial"].update(choice="unoffered"),
            lambda r: r["answers"]["next_trial"]["probabilities"].update({candidates[0].id: 0.6}),
            lambda r: r["answers"]["next_trial"].update(confidence=float("nan")),
            lambda r: r.update(model="jev-1.14.0"),
            lambda r: r["usage"].update(input_tokens=-1),
        ):
            bad = json.loads(json.dumps(good))
            change(bad)
            with self.assertRaises(InvalidJevResponse):
                controller.validate(prepared, bad)

    @unittest.skipUnless(importlib.util.find_spec("typesafe_sdk"), "optional TypeSafe SDK not installed")
    def test_sdk_uses_typed_choice_and_bounded_retry_policy(self):
        from typesafe_sdk import Choice
        state, candidates = offer()
        prepared = JevController(FakeTransport()).prepare(state, candidates, "max")
        captured = {}

        class Client:
            def __init__(self, **kwargs):
                captured["client"] = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def system_one(self, **kwargs):
                captured["call"] = kwargs
                return SimpleNamespace(model_dump=lambda mode: answer(prepared))

        transport = TypeSafeSDKTransport(timeout=3, max_retries=1, client_factory=Client)
        result = transport.invoke(prepared["body"])
        self.assertEqual(result["model"], "jev-1.13.0")
        self.assertEqual(transport.sdk_version, "0.7.1")
        self.assertIsInstance(captured["call"]["questions"]["next_trial"], Choice)
        self.assertEqual(captured["call"]["questions"]["next_trial"].criteria,
                         prepared["body"]["questions"]["next_trial"]["criteria"])
        self.assertEqual(captured["client"]["retry"].max_retries, 1)
        self.assertEqual(captured["client"]["timeout"], 3)


if __name__ == "__main__":
    unittest.main()
