"""Supervise a single trial process and retain its logs and result."""

from __future__ import annotations

import json
import hashlib
import os
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from ..core import ExperimentResult, canonical
from ..tasks.base import ProcessTask
from .ownership import parent_death_guard, process_identity, same_worker_alive


def _stop_group(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


class SubprocessExecutor:
    kind = "subprocess-v1"

    def __init__(self, run_dir, timeout: float, command=None, *, safe_store=None,
                 checkpoint_policy="none"):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.run_dir = Path(run_dir).resolve()
        self.timeout = timeout
        self.command = command
        self.safe_store = safe_store
        self.checkpoint_policy = checkpoint_policy

    def details(self):
        details = {"timeout": self.timeout, "run_dir": str(self.run_dir)}
        if self.safe_store is not None:
            details.update({"worker_ownership": "pdeathsig-v1",
                            "checkpoint_policy": self.checkpoint_policy})
        return details

    def execute(self, task: ProcessTask, spec, trial_id):
        folder = self.run_dir / f"trial-{trial_id}"
        folder.mkdir(parents=True, exist_ok=True)
        input_path, result_path = folder / "input.json", folder / "result.json"
        out_path, err_path = folder / "stdout.log", folder / "stderr.log"
        if result_path.exists():
            raise RuntimeError("trial result already exists; refusing to overwrite")
        payload = task.worker_payload(spec)
        if self.safe_store is not None:
            payload = {**payload, "trial_id": trial_id,
                       "spec_fingerprint": spec.fingerprint,
                       "checkpoint_policy": self.checkpoint_policy}
        input_path.write_text(canonical(payload) + "\n")
        input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
        artifacts = tuple(str(p.relative_to(self.run_dir)) for p in
                          (input_path, out_path, err_path, result_path))
        start = time.monotonic()
        with out_path.open("wb") as out, err_path.open("wb") as err:
            command = (self.command(input_path, result_path) if self.command else
                       task.worker_command(input_path, result_path))
            parent_pid = os.getpid()
            proc = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True,
                                    preexec_fn=(lambda: parent_death_guard(parent_pid))
                                    if self.safe_store is not None else None)
            if self.safe_store is not None:
                identity = process_identity(proc.pid)
                if identity is None:
                    _stop_group(proc)
                    raise RuntimeError("cannot record worker process identity")
                self.safe_store.record_worker(trial_id, proc.pid, identity[1], input_sha)
            try:
                code = proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                _stop_group(proc)
                return ExperimentResult("failed", None, {"exit_code": proc.returncode},
                                        time.monotonic() - start, "Timeout",
                                        f"trial exceeded {self.timeout} seconds", artifacts)
            except BaseException:
                _stop_group(proc)
                raise
        duration = time.monotonic() - start
        if code != 0:
            return ExperimentResult("failed", None, {"exit_code": code}, duration,
                                    "ChildExit", f"worker exited with status {code}", artifacts)
        try:
            if self.safe_store is not None:
                result = self._verified_result(spec, trial_id)
                if result is None:
                    raise ValueError("worker result manifest is absent or invalid")
            else:
                result = ExperimentResult(**json.loads(result_path.read_text()))
        except (OSError, ValueError, TypeError) as exc:
            return ExperimentResult("failed", None, {"exit_code": code}, duration,
                                    "WorkerResult", str(exc), artifacts)
        return ExperimentResult(result.status, result.objective,
                                {**result.metrics, "exit_code": code}, duration,
                                result.error_type, result.error_message,
                                artifacts + tuple(result.artifacts))

    def _verified_result(self, spec, trial_id):
        folder = self.run_dir / f"trial-{trial_id}"
        input_path, result_path, manifest_path = (folder / name for name in
                                                  ("input.json", "result.json", "result-manifest.json"))
        worker = self.safe_store.worker(trial_id)
        if worker is None:
            return None
        try:
            payload = json.loads(input_path.read_text())
            manifest = json.loads(manifest_path.read_text())
            raw_result = result_path.read_bytes()
            if (payload["trial_id"] != trial_id or payload["spec_fingerprint"] != spec.fingerprint
                    or payload["spec"] != json.loads(canonical(asdict(spec)))
                    or hashlib.sha256(input_path.read_bytes()).hexdigest() != worker["input_sha256"]
                    or manifest["input_sha256"] != worker["input_sha256"]
                    or manifest["result_sha256"] != hashlib.sha256(raw_result).hexdigest()
                    or manifest["spec_fingerprint"] != spec.fingerprint
                    or manifest["trial_id"] != trial_id):
                return None
            parsed = ExperimentResult(**json.loads(raw_result))
            if self.checkpoint_policy == "best" and parsed.status == "completed":
                checkpoint = folder / "model.pt"
                checkpoint_manifest = folder / "checkpoint-manifest.json"
                metadata = json.loads(checkpoint_manifest.read_text())
                if (metadata["spec_fingerprint"] != spec.fingerprint
                        or metadata["sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
                    return None
            return ExperimentResult(parsed.status, parsed.objective, parsed.metrics,
                                    manifest["worker_wall_seconds"], parsed.error_type,
                                    parsed.error_message, tuple(parsed.artifacts))
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def recover(self, spec, trial_id):
        if self.safe_store is None:
            return None
        worker = self.safe_store.worker(trial_id)
        if worker is not None and same_worker_alive(worker["pid"], worker["start_ticks"]):
            raise RuntimeError("recorded worker is still alive; stop it safely before resuming")
        result = self._verified_result(spec, trial_id)
        if result is None:
            return None
        folder = self.run_dir / f"trial-{trial_id}"
        artifacts = tuple(str((folder / name).relative_to(self.run_dir)) for name in
                          ("input.json", "stdout.log", "stderr.log", "result.json", "result-manifest.json")
                          if (folder / name).exists())
        return ExperimentResult(result.status, result.objective, result.metrics, result.duration,
                                result.error_type, result.error_message, artifacts + tuple(result.artifacts))
