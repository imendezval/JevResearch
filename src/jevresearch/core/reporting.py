"""Shared projections of durable session history for reports and exports."""

from __future__ import annotations

import json

from ..storage.history import Store


def protocol_differences(a: dict, b: dict, *, same_seed: bool = False) -> list[str]:
    fields = ("task", "protocol", "data_split", "eval_budget", "direction",
              "seed_schedule", "budget")
    if same_seed:
        fields += ("seed",)
    differences = [key for key in fields if a["settings"].get(key) != b["settings"].get(key)]
    a_strategy = a["settings"].get("proposal_strategy", "local-move")
    b_strategy = b["settings"].get("proposal_strategy", "local-move")
    if (a_strategy == b_strategy and a_strategy in ("local-move", "global-pool")
            and a["settings"].get("candidate_limit") != b["settings"].get("candidate_limit")):
        differences.append("candidate_limit")
    for field in ("proposal_domain", "domain_fingerprint"):
        if a["settings"].get(field) != b["settings"].get(field):
            differences.append(field)
    for key in ("dataset_sha256", "split_sha256", "model", "preprocessing", "metric",
                "epochs", "batch_size", "device", "scheduler"):
        if (a["settings"].get("task_details", {}).get(key)
                != b["settings"].get("task_details", {}).get(key)):
            differences.append(f"task_details.{key}")
    if a["source"]["digest"] != b["source"]["digest"]:
        differences.append("source_digest")
    return differences


def history(store: Store, sid: int):
    session = store.session(sid)
    settings = json.loads(session["settings"])
    source = json.loads(session["source"])
    offers = {row["id"]: row for row in store.offers(sid)}
    attempts = store.decision_attempts(sid)
    trials = []
    trajectory = []
    best = None
    for row in store.trials(sid):
        offer = offers.get(row["offer_id"])
        choice = None
        if offer:
            choice = next(c for c in json.loads(offer["candidates"])
                          if c["id"] == row["candidate_id"])
        result = json.loads(row["result"]) if row["result"] else None
        spec = json.loads(row["spec"])
        if result and result["status"] == "completed":
            value = result["objective"]
            if best is None or (value > best if settings["direction"] == "max" else value < best):
                best = value
        trials.append({"trial_id": row["number"], "parent_id": spec["parent_id"],
                       "candidate_id": row["candidate_id"],
                       "operator": choice["operator"] if choice else "baseline",
                       "parameters": choice["parameters"] if choice else {},
                       "spec": spec, "status": row["status"], "result": result,
                       "best_so_far": best, "created_at": row["created_at"],
                       "started_at": row["started_at"], "finished_at": row["finished_at"],
                       "offer_id": row["offer_id"]})
        trajectory.append({"attempted_trials": row["number"] + 1,
                           "elapsed_wall_seconds": max(0.0, (row["finished_at"] or row["started_at"]
                                                         or row["created_at"]) - session["created_at"]),
                           "best_objective": best, "trial_status": row["status"]})
    decisions = [{"attempt_id": a["id"], "offer_id": a["offer_id"],
                  "status": a["status"], "transport": a["transport_kind"],
                  "requested_model": a["requested_model"], "sdk_version": a["sdk_version"],
                  "request": json.loads(a["request"]),
                  "response": json.loads(a["response"]) if a["response"] else None,
                  "selected_id": a["selected_id"], "error_code": a["error_code"],
                  "latency_seconds": a["latency"], "created_at": a["created_at"],
                  "finished_at": a["finished_at"]} for a in attempts]
    observed_usage = [d["response"]["usage"] for d in decisions
                      if d["response"] and d["response"].get("usage")]
    controller_summary = {"logical_calls": len(decisions),
                          "live_api_calls": sum(d["transport"] == "typesafe-sdk" for d in decisions),
                          "lower_level_retries": None,
                          "input_tokens_observed": sum(u.get("input_tokens", 0) for u in observed_usage),
                          "output_tokens_observed": sum(u.get("output_tokens", 0) for u in observed_usage),
                          "usage_complete": len(observed_usage) == len(decisions)
                          and all("input_tokens" in u and "output_tokens" in u for u in observed_usage),
                          "cost_usd": None, "cost_note": "unavailable: no pricing snapshot stored"}
    return {"session_id": sid, "task": settings["task"],
            "proposal_strategy": settings.get("proposal_strategy", "local-move"),
            "proposal_domain": settings.get("proposal_domain", "cifar-local-v1"
                                            if settings["task"].startswith("cifar10") else "synthetic-local-v1"),
            "fixture": settings["task"].endswith("_fixture"),
            "status": session["status"], "stop_reason": session["stop_reason"],
            "settings": settings, "source": source, "trials": trials,
            "trajectory": trajectory, "decision_attempts": decisions,
            "controller_summary": controller_summary,
            "controller_is_live": settings.get("controller_details", {}).get("transport") == "typesafe-sdk"}
