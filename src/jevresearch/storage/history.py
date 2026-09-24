"""SQLite audit trail and atomic state transitions."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from ..core import Candidate, ExperimentResult, ExperimentSpec, SearchState, canonical


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
        CREATE TABLE IF NOT EXISTS decision_attempts (
          id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES sessions(id),
          offer_id INTEGER NOT NULL REFERENCES offers(id),
          request TEXT NOT NULL, requested_model TEXT NOT NULL,
          sdk_version TEXT, transport_kind TEXT NOT NULL,
          status TEXT NOT NULL, response TEXT, selected_id TEXT,
          error_code TEXT, latency REAL, created_at REAL NOT NULL,
          finished_at REAL);
        CREATE INDEX IF NOT EXISTS decision_attempts_offer
          ON decision_attempts(offer_id, id);
        CREATE TABLE IF NOT EXISTS studies (
          id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL UNIQUE,
          spec TEXT NOT NULL, source_digest TEXT NOT NULL, overrides TEXT NOT NULL,
          created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS study_members (
          study_id INTEGER NOT NULL REFERENCES studies(id), arm_id TEXT NOT NULL,
          seed INTEGER NOT NULL, session_id INTEGER UNIQUE REFERENCES sessions(id),
          blocked_reason TEXT, PRIMARY KEY(study_id,arm_id,seed));
        CREATE TABLE IF NOT EXISTS study_activity (
          id INTEGER PRIMARY KEY, study_id INTEGER NOT NULL REFERENCES studies(id),
          started_at REAL NOT NULL, finished_at REAL, status TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS workers (
          trial_id INTEGER PRIMARY KEY REFERENCES trials(id), pid INTEGER NOT NULL,
          start_ticks INTEGER NOT NULL, input_sha256 TEXT NOT NULL,
          recorded_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS checkpoints (
          trial_id INTEGER PRIMARY KEY REFERENCES trials(id), path TEXT NOT NULL,
          sha256 TEXT NOT NULL, status TEXT NOT NULL);
        """)
        # Version 1 was the unversioned Phase 1 database. Preserve its rows.
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > 4:
            raise InvariantError(f"unsupported database version {version}")
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(trials)")}
        with self.db:
            for name, kind in (("exit_code", "INTEGER"), ("stdout_artifact", "TEXT"),
                               ("stderr_artifact", "TEXT")):
                if name not in columns:
                    self.db.execute(f"ALTER TABLE trials ADD COLUMN {name} {kind}")
            if version < 4:
                self.db.execute("PRAGMA user_version=4")

    def close(self):
        self.db.close()

    def session(self, session_id: int):
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"session {session_id} does not exist")
        return row

    def create(self, settings: dict, source: dict, rng_state: str, spec: ExperimentSpec,
               study_member: tuple[int, str, int] | None = None) -> int:
        with self.db:
            cur = self.db.execute("INSERT INTO sessions(settings,source,rng_state,created_at) VALUES(?,?,?,?)",
                                  (canonical(settings), canonical(source), rng_state, time.time()))
            sid = cur.lastrowid
            self.db.execute("""INSERT INTO trials(session_id,number,candidate_id,decision,spec,
                fingerprint,config_key,status,created_at) VALUES(?,0,'baseline',?,?,?,?,'pending',?)""",
                (sid, canonical({"kind": "baseline", "seed": spec.seed}), canonical(asdict(spec)),
                 spec.fingerprint, spec.config_key, time.time()))
            if study_member is not None:
                self.db.execute("""INSERT INTO study_members(study_id,arm_id,seed,session_id)
                    VALUES(?,?,?,?)""", (*study_member, sid))
        return sid

    def register_study(self, spec: dict, fingerprint: str, source_digest: str,
                       overrides: dict) -> int:
        with self.db:
            row = self.db.execute("SELECT * FROM studies").fetchone()
            if row is not None:
                if row["fingerprint"] != fingerprint or row["spec"] != canonical(spec):
                    raise InvariantError("study specification or source changed; use a new output root")
                return row["id"]
            cur = self.db.execute("""INSERT INTO studies(fingerprint,spec,source_digest,overrides,created_at)
                VALUES(?,?,?,?,?)""", (fingerprint, canonical(spec), source_digest,
                                       canonical(overrides), time.time()))
            return cur.lastrowid

    def study(self):
        return self.db.execute("SELECT * FROM studies ORDER BY id LIMIT 1").fetchone()

    def study_members(self, study_id: int):
        return self.db.execute("SELECT * FROM study_members WHERE study_id=? ORDER BY arm_id,seed",
                               (study_id,)).fetchall()

    def study_member(self, study_id: int, arm_id: str, seed: int):
        return self.db.execute("""SELECT * FROM study_members
            WHERE study_id=? AND arm_id=? AND seed=?""", (study_id, arm_id, seed)).fetchone()

    def block_member(self, study_id: int, arm_id: str, seed: int, reason: str):
        with self.db:
            self.db.execute("""INSERT INTO study_members(study_id,arm_id,seed,blocked_reason)
                VALUES(?,?,?,?) ON CONFLICT(study_id,arm_id,seed)
                DO UPDATE SET blocked_reason=excluded.blocked_reason""",
                (study_id, arm_id, seed, reason))

    def clear_block(self, study_id: int, arm_id: str, seed: int):
        with self.db:
            self.db.execute("""DELETE FROM study_members WHERE study_id=? AND arm_id=? AND seed=?
                AND session_id IS NULL""", (study_id, arm_id, seed))
            self.db.execute("""UPDATE study_members SET blocked_reason=NULL
                WHERE study_id=? AND arm_id=? AND seed=?""", (study_id, arm_id, seed))

    def start_activity(self, study_id: int) -> int:
        with self.db:
            cur = self.db.execute("""INSERT INTO study_activity(study_id,started_at,status)
                VALUES(?,?,'running')""", (study_id, time.time()))
            return cur.lastrowid

    def finish_activity(self, activity_id: int):
        with self.db:
            self.db.execute("""UPDATE study_activity SET finished_at=?,status='completed'
                WHERE id=? AND status='running'""", (time.time(), activity_id))

    def activity(self, study_id: int):
        return self.db.execute("SELECT * FROM study_activity WHERE study_id=? ORDER BY id",
                               (study_id,)).fetchall()

    def mark_unknown_activity(self, study_id: int):
        with self.db:
            self.db.execute("""UPDATE study_activity SET status='unknown'
                WHERE study_id=? AND status='running'""", (study_id,))

    def record_worker(self, trial_id: int, pid: int, start_ticks: int, input_sha256: str):
        with self.db:
            self.db.execute("""INSERT OR REPLACE INTO workers
                (trial_id,pid,start_ticks,input_sha256,recorded_at) VALUES(?,?,?,?,?)""",
                (trial_id, pid, start_ticks, input_sha256, time.time()))

    def worker(self, trial_id: int):
        return self.db.execute("SELECT * FROM workers WHERE trial_id=?", (trial_id,)).fetchone()

    def record_checkpoint(self, trial_id: int, path: str, sha256: str):
        with self.db:
            self.db.execute("""INSERT OR REPLACE INTO checkpoints(trial_id,path,sha256,status)
                VALUES(?,?,?,'retained')""", (trial_id, path, sha256))

    def checkpoints(self, sid: int):
        return self.db.execute("""SELECT c.*,t.number,t.session_id FROM checkpoints c
            JOIN trials t ON t.id=c.trial_id WHERE t.session_id=? ORDER BY t.number""", (sid,)).fetchall()

    def prune_checkpoint(self, trial_id: int):
        with self.db:
            self.db.execute("UPDATE checkpoints SET status='pruned' WHERE trial_id=?", (trial_id,))

    def trials(self, sid: int):
        return self.db.execute("SELECT * FROM trials WHERE session_id=? ORDER BY number", (sid,)).fetchall()

    def outstanding(self, sid: int):
        return self.db.execute("SELECT * FROM offers WHERE session_id=? AND selected_id IS NULL ORDER BY id DESC LIMIT 1",
                               (sid,)).fetchone()

    def offers(self, sid: int):
        return self.db.execute("SELECT * FROM offers WHERE session_id=? ORDER BY id", (sid,)).fetchall()

    def decision_attempts(self, sid: int):
        return self.db.execute("SELECT * FROM decision_attempts WHERE session_id=? ORDER BY id", (sid,)).fetchall()

    def first_decision_request(self, offer_id: int):
        row = self.db.execute("SELECT request FROM decision_attempts WHERE offer_id=? ORDER BY id LIMIT 1",
                              (offer_id,)).fetchone()
        return json.loads(row["request"]) if row else None

    def validated_decision(self, offer_id: int):
        return self.db.execute("""SELECT * FROM decision_attempts
            WHERE offer_id=? AND status='validated' ORDER BY id DESC LIMIT 1""",
                               (offer_id,)).fetchone()

    def mark_unknown_started(self, sid: int):
        with self.db:
            self.db.execute("""UPDATE decision_attempts SET status='unknown',
                error_code='process_interrupted',finished_at=?
                WHERE session_id=? AND status='started'""", (time.time(), sid))

    def begin_decision(self, sid: int, offer_id: int, prepared: dict, details: dict) -> int:
        with self.db:
            offer = self.offer(sid, offer_id)
            if offer["selected_id"] is not None:
                raise InvariantError("cannot call controller on selected offer")
            cur = self.db.execute("""INSERT INTO decision_attempts
                (session_id,offer_id,request,requested_model,sdk_version,transport_kind,status,created_at)
                VALUES(?,?,?,?,?,?,'started',?)""",
                (sid, offer_id, canonical(prepared), details["model"], details.get("sdk_version"),
                 details["transport"], time.time()))
        return cur.lastrowid

    def fail_decision(self, attempt_id: int, status: str, error_code: str, latency: float):
        if status not in ("failed", "unknown"):
            raise ValueError("invalid decision failure status")
        with self.db:
            cur = self.db.execute("""UPDATE decision_attempts SET status=?,error_code=?,latency=?,finished_at=?
                WHERE id=? AND status='started'""",
                (status, error_code, latency, time.time(), attempt_id))
            if cur.rowcount != 1:
                raise InvariantError("decision attempt is not started")

    def validate_decision(self, attempt_id: int, response: dict, selected_id: str, latency: float):
        with self.db:
            cur = self.db.execute("""UPDATE decision_attempts
                SET status='validated',response=?,selected_id=?,latency=?,finished_at=?
                WHERE id=? AND status='started'""",
                (canonical(response), selected_id, latency, time.time(), attempt_id))
            if cur.rowcount != 1:
                raise InvariantError("decision attempt is not started")

    def offer(self, sid: int, offer_id: int):
        row = self.db.execute("SELECT * FROM offers WHERE session_id=? AND id=?",
                              (sid, offer_id)).fetchone()
        if row is None:
            raise KeyError(f"offer {offer_id} does not exist in session {sid}")
        return row

    def save_offer(self, sid: int, state: SearchState, candidates: tuple[Candidate, ...], rng: str) -> int:
        with self.db:
            cur = self.db.execute("INSERT INTO offers(session_id,state,candidates,rng_before,created_at) VALUES(?,?,?,?,?)",
                                  (sid, canonical(asdict(state)), canonical([asdict(c) for c in candidates]), rng, time.time()))
        return cur.lastrowid

    def select(self, sid: int, offer_id: int, candidate: Candidate, rng_after: str,
               decision_attempt_id: int | None = None):
        with self.db:
            offer = self.db.execute("SELECT * FROM offers WHERE id=? AND session_id=?", (offer_id, sid)).fetchone()
            if offer is None or offer["selected_id"] is not None:
                raise InvariantError("offer missing or already selected")
            choices = json.loads(offer["candidates"])
            if sum(c["id"] == candidate.id and c == asdict(candidate) for c in choices) != 1:
                raise InvariantError("selected candidate is not exactly in saved offer")
            if decision_attempt_id is not None:
                attempt = self.db.execute("""SELECT * FROM decision_attempts
                    WHERE id=? AND session_id=? AND offer_id=?""",
                    (decision_attempt_id, sid, offer_id)).fetchone()
                if (attempt is None or attempt["status"] != "validated"
                        or attempt["selected_id"] != candidate.id):
                    raise InvariantError("decision attempt does not validate this candidate")
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
            if decision_attempt_id is not None:
                self.db.execute("UPDATE decision_attempts SET status='selected' WHERE id=?",
                                (decision_attempt_id,))

    def running(self, trial_id: int):
        with self.db:
            cur = self.db.execute("UPDATE trials SET status='running',started_at=? WHERE id=? AND status='pending'",
                                  (time.time(), trial_id))
            if cur.rowcount != 1:
                raise InvariantError("trial is not pending")

    def finish(self, trial_id: int, result: ExperimentResult):
        def artifact(suffix):
            return next((p for p in result.artifacts if p.endswith(suffix)), None)
        with self.db:
            cur = self.db.execute("""UPDATE trials SET status=?,result=?,finished_at=?,
                exit_code=?,stdout_artifact=?,stderr_artifact=?
                WHERE id=? AND status='running'""",
                (result.status, canonical(asdict(result)), time.time(),
                 result.metrics.get("exit_code"), artifact("stdout.log"),
                 artifact("stderr.log"), trial_id))
            if cur.rowcount != 1:
                raise InvariantError("trial is not running")

    def interrupt(self, trial_id: int):
        result = ExperimentResult("interrupted", None, {}, 0.0, "Interrupted",
                                  "process ended while trial was running")
        self.finish(trial_id, result)

    def stop(self, sid: int, reason: str):
        with self.db:
            self.db.execute("UPDATE sessions SET status='stopped',stop_reason=? WHERE id=?", (reason, sid))

    def pause(self, sid: int, reason: str):
        with self.db:
            self.db.execute("UPDATE sessions SET status='paused',stop_reason=? WHERE id=? AND status!='stopped'",
                            (reason, sid))

    def resume(self, sid: int):
        with self.db:
            self.db.execute("UPDATE sessions SET status='active',stop_reason=NULL WHERE id=? AND status='paused'",
                            (sid,))

    def state(self, sid: int) -> SearchState:
        rows = self.trials(sid)
        settings = json.loads(self.session(sid)["settings"])
        best = None
        history = []
        from ..core import better, valid_objective
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
