"""Synthetic and CIFAR-10 experiment commands."""

import argparse
import json
from pathlib import Path

from .controllers.jev import live_controller
from .controllers.random import RandomController
from .controllers.single import SingleCandidateController
from .core.runner import Runner
from .core.reporting import history, protocol_differences
from .core.study_runner import StudyRunner
from .core.study_report import study_report
from .execution.subprocess import SubprocessExecutor
from .storage.history import Store
from .tasks.synthetic import SyntheticTask
from .tasks.vision.cifar10 import CifarTask
from .tasks.vision.cifar10.global_search import cifar_generator


def compare(random_history: dict, jev_history: dict):
    differences = protocol_differences(random_history, jev_history, same_seed=True)
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


def _controller_from_settings(settings):
    if settings["controller"] == "random":
        return RandomController()
    if settings["controller"] == "single":
        return SingleCandidateController()
    if settings["controller"] != "jev":
        raise ValueError("unknown controller in session")
    details = settings["controller_details"]
    if details["transport"] != "typesafe-sdk":
        raise RuntimeError("fake Jev sessions are test-only and cannot be resumed through the live CLI")
    return live_controller(details["model"], details["request_timeout"],
                            details["sdk_max_retries"], details["max_calls_per_run"])


def _cifar_runner(store, settings, controller):
    details = settings["task_details"]
    task = CifarTask(details["data_dir"], details["run_dir"],
                     split_seed=details["split_seed"], epochs=details["epochs"],
                     batch_size=details["batch_size"], device=details["device"],
                     download=False, fixture=details["dataset"] == "cifar10_fixture")
    generator = cifar_generator(settings.get("proposal_strategy", "local-move"),
                                settings.get("proposal_domain"), settings.get("candidate_limit", 8))
    return Runner(store, task, controller,
                  SubprocessExecutor(details["run_dir"], settings["execution_details"]["timeout"]), generator)


def show(store: Store, sid: int):
    session = store.session(sid)
    settings = json.loads(session["settings"])
    state = store.state(sid)
    trials = store.trials(sid)
    offers = store.db.execute("SELECT * FROM offers WHERE session_id=? ORDER BY id", (sid,)).fetchall()
    print(json.dumps({
        "session_id": sid, "status": session["status"],
        "task": settings["task"],
        "fixture": settings["task"].endswith("_fixture"),
        "controller": settings["controller"],
        "proposal_strategy": settings.get("proposal_strategy", "local-move"),
        "proposal_domain": settings.get("proposal_domain", "cifar-local-v1"
                                        if settings["task"].startswith("cifar10") else "synthetic-local-v1"),
        "controller_is_live": settings.get("controller_details", {}).get("transport") == "typesafe-sdk",
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
    cifar.add_argument("--proposal-strategy", choices=("local-move", "global-random", "global-pool",
                                                      "tpe", "tpe-pool", "cmaes"), default="local-move")
    cifar.add_argument("--proposal-domain", choices=("cifar-mixed-v1", "cifar-sgd-numeric-v1"))
    cifar.add_argument("--candidate-limit", type=int, default=8)
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
    study = sub.add_parser("study")
    study_commands = study.add_subparsers(dest="study_command", required=True)
    for name in ("run", "resume", "report"):
        command = study_commands.add_parser(name)
        command.add_argument("--spec", required=True)
        command.add_argument("--data-dir")
        command.add_argument("--output-root")
        if name != "report":
            command.add_argument("--max-new-trials", type=int)
        else:
            command.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "study":
        scheduled = StudyRunner(args.spec, data_dir=args.data_dir,
                                output_root=args.output_root)
        if args.study_command == "report":
            study_store = Store(scheduled.root / "study.sqlite")
            try:
                row = study_store.study()
                if row is None:
                    raise RuntimeError("study has not been started")
                output = json.dumps(study_report(study_store, row["id"]), indent=2)
            finally:
                study_store.close()
            path = Path(args.output) if args.output else scheduled.root / "report.json"
            path.write_text(output + "\n")
            print(path)
        else:
            study_id = scheduled.run(args.max_new_trials)
            print(json.dumps({"study_id": study_id,
                              "registry": str(scheduled.root / "study.sqlite")}, indent=2))
        return
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
    if args.command == "cifar-run":
        if args.proposal_strategy == "local-move" and args.proposal_domain is not None:
            parser.error("local-move does not take a global proposal domain")
        if args.proposal_strategy != "local-move" and args.proposal_domain is None:
            parser.error("global search requires --proposal-domain")
        if args.proposal_strategy == "cmaes" and args.proposal_domain != "cifar-sgd-numeric-v1":
            parser.error("CMA-ES requires the numeric domain")
        if not 1 <= args.candidate_limit <= 8:
            parser.error("candidate limit must be in [1,8]")
        if args.proposal_strategy == "tpe-pool" and (
                args.proposal_domain != "cifar-mixed-v1" or args.candidate_limit < 2):
            parser.error("TPE pool requires the mixed domain and candidate limit in [2,8]")
    controller = None
    if args.command in ("run", "cifar-run"):
        if args.command == "cifar-run" and args.proposal_strategy in ("global-random", "tpe", "cmaes"):
            if args.controller != "random":
                parser.error("single-proposal strategies cannot use Jev selection")
            controller = SingleCandidateController()
        else:
            controller = (live_controller(args.model, args.api_timeout, args.sdk_retries,
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
        generator = cifar_generator(args.proposal_strategy, args.proposal_domain, args.candidate_limit)
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
                            SubprocessExecutor(run_dir, args.timeout), generator)
            sid = runner.start(args.budget, args.seed)
            runner.run(sid, args.max_new_trials)
        elif args.command == "cifar-resume":
            sid = args.session
            settings = json.loads(store.session(sid)["settings"])
            runner = _cifar_runner(store, settings, _controller_from_settings(settings))
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
