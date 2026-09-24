"""Synthetic and CIFAR-10 experiment commands."""

import argparse
import json
from pathlib import Path

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
    offers = {row["id"]: row for row in store.db.execute(
        "SELECT * FROM offers WHERE session_id=?", (sid,)).fetchall()}
    trials = []
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
    return {"session_id": sid, "task": settings["task"],
            "fixture": settings["task"].endswith("_fixture"),
            "status": session["status"], "stop_reason": session["stop_reason"],
            "settings": settings, "source": source, "trials": trials}


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


def _cifar_runner(store, details, timeout):
    task = CifarTask(details["data_dir"], details["run_dir"],
                     split_seed=details["split_seed"], epochs=details["epochs"],
                     batch_size=details["batch_size"], device=details["device"],
                     download=False, fixture=details["dataset"] == "cifar10_fixture")
    return Runner(store, task, RandomController(),
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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jevresearch")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--db", required=True)
    run.add_argument("--budget", type=int, required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-new-trials", type=int)
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
    cifar_resume = sub.add_parser("cifar-resume")
    cifar_resume.add_argument("--db", required=True)
    cifar_resume.add_argument("--session", type=int, required=True)
    cifar_resume.add_argument("--max-new-trials", type=int)
    export = sub.add_parser("export")
    export.add_argument("--db", required=True)
    export.add_argument("--session", type=int, required=True)
    export.add_argument("--output")
    args = parser.parse_args(argv)
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
            runner = Runner(store, SyntheticTask(), RandomController())
            sid = runner.start(args.budget, args.seed)
            runner.run(sid, args.max_new_trials)
        elif args.command == "resume":
            sid = args.session
            Runner(store, SyntheticTask(), RandomController()).run(sid, args.max_new_trials)
        elif args.command == "cifar-run":
            runner = Runner(store, task, RandomController(),
                            SubprocessExecutor(run_dir, args.timeout))
            sid = runner.start(args.budget, args.seed)
            runner.run(sid, args.max_new_trials)
        elif args.command == "cifar-resume":
            sid = args.session
            settings = json.loads(store.session(sid)["settings"])
            runner = _cifar_runner(store, settings["task_details"],
                                   settings["execution_details"]["timeout"])
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
