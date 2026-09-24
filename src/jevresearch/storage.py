"""SQLite audit trail and atomic state transitions."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from .core import Candidate, ExperimentResult, ExperimentSpec, SearchState, canonical


class InvariantError(RuntimeError):
    pass


class Store:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
          id INTEGER PRIMARY KEY, settings TEXT NOT NULL, source TEXT NOT NULL,
          rng_state TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active',
          stop_reason TEXT, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS offers (
          id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES sessions(id),
          state TEXT NOT NULL, candidates TEXT NOT NULL, rng_before TEXT NOT NULL,
          selected_id TEXT, rng_after TEXT, created_at REAL NOT NULL,
          UNIQUE(session_id, id));
        CREATE TABLE IF NOT EXISTS trials (
          id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES sessions(id),
          number INTEGER NOT NULL, offer_id INTEGER REFERENCES offers(id),
          candidate_id TEXT NOT NULL, decision TEXT NOT NULL, spec TEXT NOT NULL,
          fingerprint TEXT NOT NULL, config_key TEXT NOT NULL,
          status TEXT NOT NULL, result TEXT, created_at REAL NOT NULL,
          started_at REAL, finished_at REAL,
          UNIQUE(session_id, number), UNIQUE(session_id, fingerprint),
          UNIQUE(session_id, config_key), UNIQUE(offer_id));
        """)

    def close(self):
        self.db.close()

    def session(self, session_id: int):
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"session {session_id} does not exist")
        return row

    def create(self, settings: dict, source: dict, rng_state: str, spec: ExperimentSpec) -> int:
        with self.db:
            cur = self.db.execute("INSERT INTO sessions(settings,source,rng_state,created_at) VALUES(?,?,?,?)",
                                  (canonical(settings), canonical(source), rng_state, time.time()))
            sid = cur.lastrowid
            self.db.execute("""INSERT INTO trials(session_id,number,candidate_id,decision,spec,
                fingerprint,config_key,status,created_at) VALUES(?,0,'baseline',?,?,?,?,'pending',?)""",
                (sid, canonical({"kind": "baseline", "seed": spec.seed}), canonical(asdict(spec)),
                 spec.fingerprint, spec.config_key, time.time()))
        return sid

    def trials(self, sid: int):
        return self.db.execute("SELECT * FROM trials WHERE session_id=? ORDER BY number", (sid,)).fetchall()

    def outstanding(self, sid: int):
        return self.db.execute("SELECT * FROM offers WHERE session_id=? AND selected_id IS NULL ORDER BY id DESC LIMIT 1",
                               (sid,)).fetchone()

    def save_offer(self, sid: int, state: SearchState, candidates: tuple[Candidate, ...], rng: str) -> int:
        with self.db:
            cur = self.db.execute("INSERT INTO offers(session_id,state,candidates,rng_before,created_at) VALUES(?,?,?,?,?)",
                                  (sid, canonical(asdict(state)), canonical([asdict(c) for c in candidates]), rng, time.time()))
        return cur.lastrowid

    def select(self, sid: int, offer_id: int, candidate: Candidate, rng_after: str):
        with self.db:
            offer = self.db.execute("SELECT * FROM offers WHERE id=? AND session_id=?", (offer_id, sid)).fetchone()
            if offer is None or offer["selected_id"] is not None:
                raise InvariantError("offer missing or already selected")
            choices = json.loads(offer["candidates"])
            if sum(c["id"] == candidate.id and c == asdict(candidate) for c in choices) != 1:
                raise InvariantError("selected candidate is not exactly in saved offer")
            number = self.db.execute("SELECT COUNT(*) FROM trials WHERE session_id=?", (sid,)).fetchone()[0]
            settings = json.loads(self.session(sid)["settings"])
            if number >= settings["budget"]:
                raise InvariantError("budget exhausted")
            try:
                self.db.execute("""INSERT INTO trials(session_id,number,offer_id,candidate_id,decision,
                    spec,fingerprint,config_key,status,created_at)
                    VALUES(?,?,?,?,?,?,?,?,'pending',?)""",
                    (sid, number, offer_id, candidate.id,
                     canonical({"kind": "controller", "selected_id": candidate.id}),
                     canonical(asdict(candidate.spec)), candidate.spec.fingerprint,
                     candidate.spec.config_key, time.time()))
            except sqlite3.IntegrityError as exc:
                raise InvariantError(f"duplicate trial/config or selection: {exc}") from exc
            self.db.execute("UPDATE offers SET selected_id=?,rng_after=? WHERE id=?",
                            (candidate.id, rng_after, offer_id))
            self.db.execute("UPDATE sessions SET rng_state=? WHERE id=?", (rng_after, sid))

    def running(self, trial_id: int):
        with self.db:
            cur = self.db.execute("UPDATE trials SET status='running',started_at=? WHERE id=? AND status='pending'",
                                  (time.time(), trial_id))
            if cur.rowcount != 1:
                raise InvariantError("trial is not pending")

    def finish(self, trial_id: int, result: ExperimentResult):
        with self.db:
            cur = self.db.execute("UPDATE trials SET status=?,result=?,finished_at=? WHERE id=? AND status='running'",
                                  (result.status, canonical(asdict(result)), time.time(), trial_id))
            if cur.rowcount != 1:
                raise InvariantError("trial is not running")

    def interrupt(self, trial_id: int):
        result = ExperimentResult("interrupted", None, {}, 0.0, "Interrupted",
                                  "process ended while trial was running")
        self.finish(trial_id, result)

    def stop(self, sid: int, reason: str):
        with self.db:
            self.db.execute("UPDATE sessions SET status='stopped',stop_reason=? WHERE id=?", (reason, sid))

    def state(self, sid: int) -> SearchState:
        rows = self.trials(sid)
        settings = json.loads(self.session(sid)["settings"])
        best = None
        history = []
        from .core import better, valid_objective
        for row in rows:
            result = json.loads(row["result"]) if row["result"] else None
            history.append({"trial_id": row["number"], "number": row["number"],
                            "candidate_id": row["candidate_id"], "status": row["status"],
                            "config_key": row["config_key"],
                            "objective": result["objective"] if result else None})
            if result and valid_objective(ExperimentResult(**result)):
                value = result["objective"]
                if better(value, json.loads(best["result"])["objective"] if best else None,
                          settings["direction"]):
                    best = row
        return SearchState(sid, len(rows), settings["budget"], best["number"] if best else None,
                           json.loads(best["result"])["objective"] if best else None,
                           json.loads(best["spec"])["config"] if best else None, tuple(history))
