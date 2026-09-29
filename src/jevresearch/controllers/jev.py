"""One audited TypeSafe Choice over an already valid experiment offer."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Literal

from ..core import Candidate, SearchState, canonical

LEGACY_QUESTION_VERSION = "next-trial-v1"
DEFAULT_QUESTION_VERSION = "next-trial-v2"
QUESTION_VERSIONS = (LEGACY_QUESTION_VERSION, DEFAULT_QUESTION_VERSION)
INSTRUCTIONS = (
    "Choose one offered experiment ID to try next for the stated objective. "
    "Use the completed results and the stated configuration changes as evidence. "
    "Return exactly one supplied ID; do not invent a trial. Numeric comparisons are provided in state."
)
V2_INSTRUCTIONS = (
    "Choose exactly one offered experiment ID to train next. Maximize the best final "
    "objective achieved within the remaining trial slots. Consider both plausible "
    "objective improvement and what the result could teach later proposals. "
    "Use only the supplied observations; do not invent a trial."
)
MAX_CANDIDATES = 8
MAX_REQUEST_BYTES = 8192
_PRIVATE_KEY = re.compile(r"(secret|token|password|credential|api.?key|authorization)", re.I)
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class InvalidJevResponse(ValueError):
    """The provider answer cannot safely select a persisted candidate."""


class TransportFailure(RuntimeError):
    def __init__(self, category: str, outcome_unknown: bool = False):
        super().__init__(category)
        self.category = category
        self.outcome_unknown = outcome_unknown


def _safe(value: Any, stats: dict[str, int], key: str = "") -> Any:
    if _PRIVATE_KEY.search(key):
        stats["redacted_fields"] += 1
        return "[redacted]"
    if isinstance(value, str):
        if value.startswith(("/", "~", "\\\\")) or _WINDOWS_PATH.match(value):
            stats["redacted_fields"] += 1
            return "[path]"
        if len(value) > 160:
            stats["truncated_fields"] += 1
            return value[:160] + "[truncated]"
        return value
    if isinstance(value, dict):
        if any(str(k).startswith(("/", "~", "\\\\")) or _WINDOWS_PATH.match(str(k))
               for k in value):
            raise ValueError("absolute path in candidate config key")
        return {str(k): _safe(v, stats, str(k)) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_safe(v, stats) for v in value]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    raise ValueError("candidate config contains an unsupported value")


def _recent(state: SearchState, direction: str):
    recent = [dict(row) for row in state.history[-5:]]
    completed = [row for row in recent if row["status"] == "completed"
                 and isinstance(row["objective"], (int, float))
                 and math.isfinite(row["objective"])]
    ranked = sorted(completed, key=lambda row: row["objective"], reverse=direction == "max")
    ranks = {row["trial_id"]: i + 1 for i, row in enumerate(ranked)}
    return [{"trial_id": row["trial_id"], "status": row["status"],
             "objective": row["objective"],
             "rank_among_recent": ranks.get(row["trial_id"]),
             "delta_from_best": round(row["objective"] - state.best_objective, 8)
             if row["status"] == "completed" and row["objective"] is not None
             and state.best_objective is not None else None} for row in recent]


def _description(candidate: Candidate, incumbent: dict | None, stats: dict[str, int]) -> str:
    config = _safe(candidate.config, stats)
    old = incumbent or {}
    changes = {}
    for key, new in candidate.config.items():
        previous = old.get(key)
        if previous != new:
            change = {"from": _safe(previous, stats, key), "to": config[key]}
            if (not _PRIVATE_KEY.search(key)
                    and isinstance(previous, (int, float)) and not isinstance(previous, bool)
                    and isinstance(new, (int, float)) and not isinstance(new, bool)):
                change["delta"] = round(new - previous, 10)
            changes[key] = change
    operator = _safe(candidate.operator, stats)
    return f"operator={operator}; changes={canonical(changes)}; full_config={canonical(config)}"


def _search_config(config: dict, fields: dict, stats: dict[str, int]) -> dict:
    return {name: _safe(config.get(name), stats, name) for name in fields}


def _distance(left: dict, right: dict, fields: dict) -> float:
    """Simple, declared search-space distance; never an objective prediction."""
    total = 0.0
    for name, scale in fields.items():
        a, b = left.get(name), right.get(name)
        if a is None or b is None:
            total += 1.0
            continue
        if scale == "categorical":
            total += float(a != b)
        elif scale == "linear":
            total += abs(a - b)
        elif scale == "log10":
            total += abs(math.log10(a / b))
        elif scale == "zero_or_log10":
            total += (0.0 if a == b == 0 else 1.0 if a == 0 or b == 0
                      else abs(math.log10(a / b)))
        else:
            raise ValueError(f"unsupported search field scale: {scale}")
    return round(total, 6)


def _scale_changes(config: dict, incumbent: dict | None, fields: dict) -> dict:
    old = incumbent or {}
    changes = {}
    for name, scale in fields.items():
        before, after = old.get(name), config.get(name)
        if before is None or before == after:
            continue
        if scale == "categorical":
            changes[name] = {"from": before, "to": after}
        elif scale == "log10":
            changes[name] = {"log10_ratio_to_best": round(math.log10(after / before), 4)}
        elif scale == "zero_or_log10":
            changes[name] = {"from_branch": "zero" if before == 0 else "positive",
                             "to_branch": "zero" if after == 0 else "positive"}
            if before > 0 and after > 0:
                changes[name]["log10_ratio_to_best"] = round(math.log10(after / before), 4)
        elif scale != "linear":
            raise ValueError(f"unsupported search field scale: {scale}")
    return changes


def _completed(history: tuple[dict, ...]) -> list[dict]:
    return [row for row in history if row["status"] == "completed"
            and row["result"] is not None
            and isinstance(row["result"].get("objective"), (int, float))
            and math.isfinite(row["result"]["objective"])]


def _history_evidence(history: tuple[dict, ...], state: SearchState, direction: str,
                      fields: dict, stats: dict[str, int], nearest: tuple[dict, ...]) -> list[dict]:
    completed = _completed(history)
    ordered = sorted(completed, key=lambda row: row["result"]["objective"],
                     reverse=direction == "max")
    selected = {}
    for label, rows in (("nearest", nearest),
                        ("strong", ordered[:4]),
                        ("recent", list(reversed(history[-5:]))),
                        ("weak", list(reversed(ordered[-2:])))):
        for row in rows:
            number = row["number"]
            if number == state.best_trial_id or number in selected:
                continue
            selected[number] = {"trial_id": number, "reason": label,
                                "status": row["status"],
                                "objective": row["result"]["objective"] if row["result"] else None,
                                "config": _search_config(row["spec"]["config"], fields, stats)}
    return list(selected.values())


@dataclass(frozen=True)
class ValidatedChoice:
    selected_id: str
    response: dict[str, Any]


class TypeSafeSDKTransport:
    """Only this adapter knows the optional SDK and its retry/timeout behavior."""

    kind = "typesafe-sdk"

    def __init__(self, timeout: float = 10.0, max_retries: int = 1, client_factory=None):
        if not math.isfinite(timeout) or timeout <= 0 or max_retries not in (0, 1):
            raise ValueError("Jev timeout must be positive and SDK retries must be 0 or 1")
        try:
            import typesafe_sdk
        except ImportError as exc:
            raise RuntimeError("install the Jev extra: pip install '.[jev]'") from exc
        self.sdk = typesafe_sdk
        self.sdk_version = metadata.version("typesafe-sdk")
        self.timeout = timeout
        self.max_retries = max_retries
        self.client_factory = client_factory or typesafe_sdk.TypeSafeClient

    def invoke(self, body: dict) -> dict:
        sdk = self.sdk
        question = body["questions"]["next_trial"]
        retry = sdk.RetryPolicy(max_retries=self.max_retries,
                                timeout=self.timeout * (self.max_retries + 1) + 2)
        try:
            with self.client_factory(timeout=self.timeout, retry=retry) as client:
                response = client.system_one(
                    state=body["state"], model=body["model"],
                    questions={"next_trial": sdk.Choice(instructions=question["instructions"],
                                                        criteria=question["criteria"])},
                    timeout=self.timeout,
                )
        except sdk.TypeSafeAuthenticationError as exc:
            raise TransportFailure("authentication") from None
        except sdk.TypeSafeRateLimitError as exc:
            raise TransportFailure("rate_limit_exhausted", True) from None
        except sdk.TypeSafeAPITimeoutError as exc:
            raise TransportFailure("timeout", True) from None
        except sdk.TypeSafeAPIConnectionError as exc:
            raise TransportFailure("connection", True) from None
        except sdk.TypeSafeAPIResponseValidationError as exc:
            raise TransportFailure("provider_response_invalid", True) from None
        except sdk.TypeSafeAPIError as exc:
            raise TransportFailure(f"http_{exc.status}", exc.status >= 500) from None
        except sdk.TypeSafeError as exc:
            raise TransportFailure("sdk_error") from None
        return response.model_dump(mode="json")


class JevController:
    kind = "jev"
    selection_mode: Literal["audited"] = "audited"

    def __init__(self, transport, model: str = "jev-1.13.0", max_calls: int = 1,
                 question_version: str = DEFAULT_QUESTION_VERSION):
        if not model or max_calls < 1 or question_version not in QUESTION_VERSIONS:
            raise ValueError("Jev model, positive call limit, and supported question version are required")
        self.transport = transport
        self.model = model
        self.max_calls = max_calls
        self.question_version = question_version

    def details(self):
        return {"model": self.model, "question_version": self.question_version,
                "transport": self.transport.kind,
                "sdk_version": getattr(self.transport, "sdk_version", None),
                "request_timeout": getattr(self.transport, "timeout", None),
                "sdk_max_retries": getattr(self.transport, "max_retries", None),
                "max_calls_per_run": self.max_calls}

    def prepare(self, state: SearchState, candidates: tuple[Candidate, ...], direction: str,
                history: tuple[dict, ...] = (), context: dict | None = None) -> dict:
        if not 1 <= len(candidates) <= MAX_CANDIDATES:
            raise ValueError("Jev requires one to eight saved candidates")
        if len({c.id for c in candidates}) != len(candidates):
            raise ValueError("duplicate candidate IDs")
        stats = {"redacted_fields": 0, "truncated_fields": 0}
        first = candidates[0]
        if self.question_version == LEGACY_QUESTION_VERSION:
            criteria = {c.id: _description(c, state.incumbent_config, stats) for c in candidates}
            evidence = {"task": _safe(first.spec.task, stats),
                        "protocol": _safe(first.spec.protocol, stats),
                        "objective_direction": direction,
                        "remaining_trial_slots": state.budget - state.attempted,
                        "best_completed": {"trial_id": state.best_trial_id,
                                           "objective": state.best_objective},
                        "recent_outcomes": _recent(state, direction)}
            instructions, trim_key, protected = INSTRUCTIONS, "recent_outcomes", 0
        else:
            if context is None or not isinstance(context.get("search_fields"), dict):
                raise ValueError("v2 requires task decision context")
            fields = context["search_fields"]
            completed = _completed(history)
            criteria = {}
            nearest_rows = []
            for candidate in candidates:
                nearest = (min(completed,
                               key=lambda row: (_distance(candidate.config,
                                                          row["spec"]["config"], fields), row["number"]))
                           if completed else None)
                if nearest is not None:
                    nearest_rows.append(nearest)
                extra = {"scale_aware_changes": _scale_changes(candidate.config,
                                                               state.incumbent_config, fields),
                         "nearest_completed": {"trial_id": nearest["number"],
                                               "search_distance": _distance(candidate.config,
                                                                            nearest["spec"]["config"], fields)}
                         if nearest else None}
                criteria[candidate.id] = (_description(candidate, state.incumbent_config, stats)
                                          + "; evidence=" + canonical(_safe(extra, stats)))
            safe_context = _safe(context, stats)
            evidence = {"task": _safe(first.spec.task, stats),
                        "protocol": _safe(first.spec.protocol, stats),
                        "objective": {"metric": safe_context["metric"], "direction": direction},
                        "evaluation": safe_context["evaluation"],
                        "search_fields": safe_context["search_fields"],
                        "search_distance": "sum of categorical mismatches, linear gaps, and absolute log10 ratios; zero-to-positive transitions cost one",
                        "remaining_trial_slots": state.budget - state.attempted,
                        "incumbent": {"trial_id": state.best_trial_id,
                                      "objective": state.best_objective,
                                      "config": _search_config(state.incumbent_config or {}, fields, stats)},
                        "history_evidence": _history_evidence(history, state, direction,
                                                              fields, stats, tuple(nearest_rows))}
            instructions, trim_key = V2_INSTRUCTIONS, "history_evidence"
            protected = len({row["number"] for row in nearest_rows
                             if row["number"] != state.best_trial_id})
        body = {"state": evidence, "model": self.model,
                "questions": {"next_trial": {"type": "choice", "instructions": instructions,
                                             "criteria": criteria}}}
        dropped = 0
        while (len(canonical(body).encode()) > MAX_REQUEST_BYTES
               and len(body["state"][trim_key]) > protected):
            if self.question_version == LEGACY_QUESTION_VERSION:
                body["state"][trim_key].pop(0)
            else:
                body["state"][trim_key].pop()
            dropped += 1
        if len(canonical(body).encode()) > MAX_REQUEST_BYTES:
            raise ValueError("Jev request exceeds fixed byte limit")
        return {"body": body, "truncation": {**stats, "history_entries_dropped": dropped}}

    def invoke(self, prepared: dict) -> dict:
        return self.transport.invoke(prepared["body"])

    def validate(self, prepared: dict, raw: Any) -> ValidatedChoice:
        try:
            answer = raw["answers"]["next_trial"]
            options = prepared["body"]["questions"]["next_trial"]["criteria"]
            choice = answer["choice"]
            probabilities = answer["probabilities"]
            confidence = answer["confidence"]
            model = raw["model"]
            usage = raw.get("usage")
            if answer["type"] != "choice" or type(choice) is not str or choice not in options:
                raise ValueError("choice is not an offered ID")
            if type(model) is not str or not model:
                raise ValueError("response model missing")
            if re.match(r"^jev-\d", self.model) and model != self.model:
                raise ValueError("response model differs from pinned model")
            if type(probabilities) is not dict or set(probabilities) != set(options):
                raise ValueError("probabilities do not match offer")
            if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probabilities.values()) or abs(sum(probabilities.values()) - 1) > 0.02:
                raise ValueError("malformed probabilities")
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("malformed confidence")
            if probabilities[choice] < max(probabilities.values()) - 0.02:
                raise ValueError("selected choice conflicts with distribution")
            if usage is not None:
                if type(usage) is not dict or any(
                    usage.get(k) is not None and (type(usage[k]) is not int or usage[k] < 0)
                    for k in ("input_tokens", "output_tokens")
                ):
                    raise ValueError("malformed usage")
            response = {"model": model,
                        "answers": {"next_trial": {"type": "choice", "choice": choice,
                                                   "probabilities": probabilities,
                                                   "confidence": confidence}},
                        "usage": {k: usage[k] for k in ("input_tokens", "output_tokens")
                                  if usage.get(k) is not None}
                        if usage is not None else None}
            return ValidatedChoice(choice, response)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidJevResponse(str(exc)) from None


def live_controller(model: str, timeout: float, retries: int, max_calls: int,
                    question_version: str = DEFAULT_QUESTION_VERSION) -> JevController:
    """Build the credentialed SDK controller used by CLI and studies."""
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("TYPESAFE_API_KEY is required for a live Jev session")
    return JevController(TypeSafeSDKTransport(timeout=timeout, max_retries=retries),
                         model=model, max_calls=max_calls, question_version=question_version)
