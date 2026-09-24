"""Synthetic and CIFAR-10 experiment commands."""

import argparse
import json
import os
from pathlib import Path

from .controllers.jev import JevController, TypeSafeSDKTransport
from .controllers.random import RandomController
from .core.runner import Runner
from .execution.subprocess import SubprocessExecutor
from .storage.history import Store
from .tasks.synthetic import SyntheticTask
from .tasks.vision.cifar10 import CifarTask


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
            "fixture": settings["task"].endswith("_fixture"),
            "status": session["status"], "stop_reason": session["stop_reason"],
            "settings": settings, "source": source, "trials": trials,
            "trajectory": trajectory, "decision_attempts": decisions,
            "controller_summary": controller_summary,
            "controller_is_live": settings.get("controller_details", {}).get("transport") == "typesafe-sdk"}


def compare(random_history: dict, jev_history: dict):
    fields = ("task", "protocol", "data_split", "eval_budget", "direction", "seed", "budget",
              "candidate_limit")
    differences = [field for field in fields
                   if random_history["settings"].get(field) != jev_history["settings"].get(field)]
    protocol_fields = ("dataset_sha256", "split_sha256", "model", "preprocessing", "metric",
                       "epochs", "batch_size", "device", "scheduler")
    for field in protocol_fields:
        if (random_history["settings"].get("task_details", {}).get(field)
                != jev_history["settings"].get("task_details", {}).get(field)):
            differences.append(f"task_details.{field}")
    if random_history["source"]["digest"] != jev_history["source"]["digest"]:
        differences.append("source_digest")
    if random_history["settings"]["controller"] != "random":
        differences.append("first controller is not random")
    if jev_history["settings"]["controller"] != "jev":
        differences.append("second controller is not jev")
    return {"comparable_protocol": not differences, "differences": differences,
            "random": {"session_id": random_history["session_id"],
                       "trajectory": random_history["trajectory"],
                       "controller_summary": random_history["controller_summary"]},
            "jev": {"session_id": jev_history["session_id"],
                    "live": jev_history["controller_is_live"],
                    "trajectory": jev_history["trajectory"],
                    "controller_failures": [d for d in jev_history["decision_attempts"]
                                            if d["status"] in ("failed", "unknown")],
                    "controller_summary": jev_history["controller_summary"]}}


def _resolve_device(requested):
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("install the vision extra: pip install '.[vision]'") from exc
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return "cuda" if torch.cuda.is_available() else "cpu"


def _live_controller(model="jev-1.13.0", timeout=10.0, retries=1, max_calls=1):
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("TYPESAFE_API_KEY is required for a live Jev session")
    return JevController(TypeSafeSDKTransport(timeout=timeout, max_retries=retries),
                         model=model, max_calls=max_calls)


def _controller_from_settings(settings):
    if settings["controller"] == "random":
        return RandomController()
    if settings["controller"] != "jev":
        raise ValueError("unknown controller in session")
    details = settings["controller_details"]
    if details["transport"] != "typesafe-sdk":
        raise RuntimeError("fake Jev sessions are test-only and cannot be resumed through the live CLI")
    return _live_controller(details["model"], details["request_timeout"],
                            details["sdk_max_retries"], details["max_calls_per_run"])


def _cifar_runner(store, details, timeout, controller):
    task = CifarTask(details["data_dir"], details["run_dir"],
                     split_seed=details["split_seed"], epochs=details["epochs"],
                     batch_size=details["batch_size"], device=details["device"],
                     download=False, fixture=details["dataset"] == "cifar10_fixture")
    return Runner(store, task, controller,
                  SubprocessExecutor(details["run_dir"], timeout))


def show(store: Store, sid: int):
    session = store.session(sid)
    state = store.state(sid)
    trials = store.trials(sid)
    offers = store.db.execute("SELECT * FROM offers WHERE session_id=? ORDER BY id", (sid,)).fetchall()
    print(json.dumps({
        "session_id": sid, "status": session["status"],
        "task": json.loads(session["settings"])["task"],
        "fixture": json.loads(session["settings"])["task"].endswith("_fixture"),
        "controller": json.loads(session["settings"])["controller"],
        "controller_is_live": json.loads(session["settings"]).get("controller_details", {}).get("transport") == "typesafe-sdk",
        "decision_attempts": len(store.decision_attempts(sid)),
        "stop_reason": session["stop_reason"] or "paused or in progress",
        "attempted": state.attempted, "budget": state.budget,
        "best_trial_id": state.best_trial_id, "best_objective": state.best_objective,
        "trials": [{"number": t["number"], "candidate_id": t["candidate_id"],
                    "offer_id": t["offer_id"], "status": t["status"],
                    "config_key": t["config_key"], "fingerprint": t["fingerprint"],
                    "result": json.loads(t["result"]) if t["result"] else None} for t in trials],
        "offers": [{"id": o["id"], "selected_id": o["selected_id"],
                    "candidate_count": len(json.loads(o["candidates"]))} for o in offers],
    }, indent=2))


def _controller_arguments(parser):
    parser.add_argument("--controller", choices=("random", "jev"), default="random")
    parser.add_argument("--model", default="jev-1.13.0")
    parser.add_argument("--api-timeout", type=float, default=10.0)
    parser.add_argument("--sdk-retries", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-api-calls", type=int, default=1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jevresearch")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--db", required=True)
    run.add_argument("--budget", type=int, required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-new-trials", type=int)
    _controller_arguments(run)
    resume = sub.add_parser("resume")
    resume.add_argument("--db", required=True)
    resume.add_argument("--session", type=int, required=True)
    resume.add_argument("--max-new-trials", type=int)
    inspect = sub.add_parser("show")
    inspect.add_argument("--db", required=True)
    inspect.add_argument("--session", type=int, required=True)
    cifar = sub.add_parser("cifar-run")
    cifar.add_argument("--data-dir", required=True)
    cifar.add_argument("--run-dir", required=True)
    cifar.add_argument("--db")
    cifar.add_argument("--download", action="store_true")
    cifar.add_argument("--fixture", action="store_true")
    cifar.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    cifar.add_argument("--epochs", type=int, default=1)
    cifar.add_argument("--batch-size", type=int, default=256)
    cifar.add_argument("--budget", type=int, required=True)
    cifar.add_argument("--seed", type=int, default=0)
    cifar.add_argument("--split-seed", type=int, default=1729)
    cifar.add_argument("--timeout", type=float, default=900)
    cifar.add_argument("--max-new-trials", type=int)
    _controller_arguments(cifar)
    cifar_resume = sub.add_parser("cifar-resume")
    cifar_resume.add_argument("--db", required=True)
    cifar_resume.add_argument("--session", type=int, required=True)
    cifar_resume.add_argument("--max-new-trials", type=int)
    export = sub.add_parser("export")
    export.add_argument("--db", required=True)
    export.add_argument("--session", type=int, required=True)
    export.add_argument("--output")
    comparison = sub.add_parser("compare")
    comparison.add_argument("--random-db", required=True)
    comparison.add_argument("--random-session", type=int, required=True)
    comparison.add_argument("--jev-db", required=True)
    comparison.add_argument("--jev-session", type=int, required=True)
    comparison.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "compare":
        random_store = Store(args.random_db)
        jev_store = Store(args.jev_db)
        try:
            output = json.dumps(compare(history(random_store, args.random_session),
                                        history(jev_store, args.jev_session)), indent=2)
        finally:
            random_store.close()
            jev_store.close()
        if args.output:
            Path(args.output).write_text(output + "\n")
        else:
            print(output)
        return
    controller = None
    if args.command in ("run", "cifar-run"):
        controller = (_live_controller(args.model, args.api_timeout, args.sdk_retries,
                                       args.max_api_calls) if args.controller == "jev"
                      else RandomController())
    if args.command == "cifar-run":
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        db_path = args.db or str(run_dir / "history.sqlite")
        device = _resolve_device(args.device)
        task = CifarTask(args.data_dir, run_dir, split_seed=args.split_seed,
                         epochs=args.epochs, batch_size=args.batch_size,
                         device=device, download=args.download, fixture=args.fixture)
        store = Store(db_path)
    else:
        store = Store(args.db)
    try:
        if args.command == "run":
            runner = Runner(store, SyntheticTask(), controller)
            sid = runner.start(args.budget, args.seed)
            runner.run(sid, args.max_new_trials)
        elif args.command == "resume":
            sid = args.session
            settings = json.loads(store.session(sid)["settings"])
            Runner(store, SyntheticTask(), _controller_from_settings(settings)).run(sid, args.max_new_trials)
        elif args.command == "cifar-run":
            runner = Runner(store, task, controller,
                            SubprocessExecutor(run_dir, args.timeout))
            sid = runner.start(args.budget, args.seed)
            runner.run(sid, args.max_new_trials)
        elif args.command == "cifar-resume":
            sid = args.session
            settings = json.loads(store.session(sid)["settings"])
            runner = _cifar_runner(store, settings["task_details"],
                                   settings["execution_details"]["timeout"],
                                   _controller_from_settings(settings))
            runner.run(sid, args.max_new_trials)
        else:
            sid = args.session
        if args.command == "export":
            output = json.dumps(history(store, sid), indent=2)
            if args.output:
                Path(args.output).write_text(output + "\n")
            else:
                print(output)
        else:
            show(store, sid)
    finally:
        store.close()


if __name__ == "__main__":
    main()
