"""CPU-only Phase 1 CLI."""

import argparse
import json

from .controller import RandomController
from .runner import Runner
from .storage import Store
from .synthetic import SyntheticTask


def show(store: Store, sid: int):
    session = store.session(sid)
    state = store.state(sid)
    trials = store.trials(sid)
    offers = store.db.execute("SELECT * FROM offers WHERE session_id=? ORDER BY id", (sid,)).fetchall()
    print(json.dumps({
        "session_id": sid, "status": session["status"],
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
    args = parser.parse_args(argv)
    store = Store(args.db)
    try:
        if args.command == "run":
            runner = Runner(store, SyntheticTask(), RandomController())
            sid = runner.start(args.budget, args.seed)
            runner.run(sid, args.max_new_trials)
        elif args.command == "resume":
            sid = args.session
            Runner(store, SyntheticTask(), RandomController()).run(sid, args.max_new_trials)
        else:
            sid = args.session
        show(store, sid)
    finally:
        store.close()


if __name__ == "__main__":
    main()
