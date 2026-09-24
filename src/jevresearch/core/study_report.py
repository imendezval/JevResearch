"""Comparable per-seed study trajectories and compact aggregate table."""

from __future__ import annotations

import json
import statistics
import time


def _compatibility(a: dict, b: dict):
    fields = ("task", "protocol", "data_split", "eval_budget", "direction", "seed_schedule",
              "budget", "candidate_limit")
    differences = [key for key in fields if a["settings"].get(key) != b["settings"].get(key)]
    for key in ("dataset_sha256", "split_sha256", "model", "preprocessing", "metric",
                "epochs", "batch_size", "device", "scheduler"):
        if a["settings"]["task_details"].get(key) != b["settings"]["task_details"].get(key):
            differences.append(f"task_details.{key}")
    if a["source"]["digest"] != b["source"]["digest"]:
        differences.append("source_digest")
    return differences


def study_report(store, study_id: int):
    study = store.db.execute("SELECT * FROM studies WHERE id=?", (study_id,)).fetchone()
    if study is None:
        raise KeyError("study missing")
    spec = json.loads(study["spec"])
    intervals = store.activity(study_id)
    members = []
    reference = None
    differences = []
    for member in store.study_members(study_id):
        item = {"arm_id": member["arm_id"], "seed": member["seed"],
                "session_id": member["session_id"], "blocked_reason": member["blocked_reason"]}
        if member["session_id"] is None:
            item.update({"status": "blocked", "trajectory": [], "controller_failures": []})
            members.append(item)
            continue
        sid = member["session_id"]
        session = store.session(sid)
        settings = json.loads(session["settings"])
        source = json.loads(session["source"])
        entry = {"settings": settings, "source": source}
        if reference is None:
            reference = entry
        else:
            differences.extend(f"{member['arm_id']}:{member['seed']}:{field}"
                               for field in _compatibility(reference, entry))
        own_intervals = [row for row in intervals if row["session_id"] == sid]
        observed = sum(row["finished_at"] - row["started_at"] for row in own_intervals
                       if row["status"] == "completed")
        unknown = sum(row["status"] == "unknown" for row in own_intervals)
        unknown_upper = sum(max(0.0, time.time() - row["started_at"]) for row in own_intervals
                            if row["status"] == "unknown")
        best = None
        trajectory = []
        process_wall = 0.0
        training = 0.0
        missing_training = 0
        for trial in store.trials(sid):
            result = json.loads(trial["result"]) if trial["result"] else None
            if result:
                process_wall += result["duration"]
                measured = result["metrics"].get("training_seconds")
                if measured is None:
                    missing_training += 1
                else:
                    training += measured
                value = result["objective"]
                if result["status"] == "completed" and value is not None and (
                        best is None or value > best):
                    best = value
            endpoint = trial["finished_at"] or trial["started_at"] or trial["created_at"]
            active = sum(row["finished_at"] - row["started_at"] for row in own_intervals
                         if row["status"] == "completed" and
                         ((row["trial_number"] is not None and row["trial_number"] <= trial["number"])
                          or (row["trial_number"] is None and row["finished_at"] <= endpoint)))
            trajectory.append({"attempted_trials": trial["number"] + 1,
                               "trial_id": trial["number"],
                               "candidate_id": trial["candidate_id"],
                               "status": trial["status"],
                               "objective": result["objective"] if result else None,
                               "error_type": result["error_type"] if result else None,
                               "best_objective": best,
                               "calendar_elapsed_seconds": endpoint - session["created_at"],
                               "observed_active_seconds": active})
        decisions = store.decision_attempts(sid)
        controller_latency = sum(d["latency"] for d in decisions if d["latency"] is not None)
        controller_latency_missing = sum(d["latency"] is None for d in decisions)
        usage = [json.loads(d["response"]).get("usage") for d in decisions if d["response"]]
        usage = [u for u in usage if u]
        rate = spec["compute_hourly_usd"]
        item.update({"status": session["status"], "stop_reason": session["stop_reason"],
                     "protocol_identity": {"task": settings["task"],
                                           "protocol": settings["protocol"],
                                           "dataset_sha256": settings["task_details"].get("dataset_sha256"),
                                           "split_sha256": settings["task_details"].get("split_sha256"),
                                           "source_digest": source["digest"]},
                     "trajectory": trajectory, "observed_active_seconds": observed,
                     "unknown_active_intervals": unknown,
                     "unknown_active_upper_bound_seconds": unknown_upper,
                     "process_wall_seconds": process_wall,
                     "training_seconds_observed": training,
                     "training_measurements_missing": missing_training,
                     "controller_latency_seconds_observed": controller_latency,
                     "controller_latency_measurements_missing": controller_latency_missing,
                     "input_tokens_observed": sum(u.get("input_tokens", 0) for u in usage),
                     "output_tokens_observed": sum(u.get("output_tokens", 0) for u in usage),
                     "gpu_active_seconds": None,
                     "compute_cost": {"value_usd": process_wall * rate / 3600 if rate is not None else None,
                                      "kind": "estimate" if rate is not None else "unavailable"},
                     "jev_cost": {"value_usd": None, "kind": "unavailable: no archived pricing snapshot"},
                     "logical_api_attempts": len(decisions), "lower_level_retries": None,
                     "controller_failures": [{"attempt_id": d["id"], "status": d["status"],
                                              "error_code": d["error_code"]} for d in decisions
                                             if d["status"] in ("failed", "unknown")]})
        members.append(item)
    aggregate = None
    if not differences:
        aggregate = []
        for arm in spec["arms"]:
            grouped = [m for m in members if m["arm_id"] == arm["id"] and m["trajectory"]]
            for attempted in range(1, spec["trial_budget"] + 1):
                points = [m["trajectory"][attempted - 1] for m in grouped
                          if len(m["trajectory"]) >= attempted]
                values = [p["best_objective"] for p in points if p["best_objective"] is not None]
                aggregate.append({"arm_id": arm["id"], "attempted_trials": attempted,
                                  "seeds_observed": len(points),
                                  "median_best_objective": statistics.median(values) if values else None,
                                  "median_observed_active_seconds": statistics.median(
                                      p["observed_active_seconds"] for p in points) if points else None})
    return {"study_id": study_id, "fingerprint": study["fingerprint"],
            "source_digest": study["source_digest"], "spec": spec,
            "overrides": json.loads(study["overrides"]),
            "compatible": not differences, "compatibility_differences": differences,
            "members": members, "aggregate_table": aggregate,
            "active_time_note": "observed orchestration intervals include worker and controller time; unknown intervals excluded",
            "cost_note": "compute cost is an occupancy estimate, not measured energy or Jev spend"}
