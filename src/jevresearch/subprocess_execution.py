"""Supervise a single trial process and retain its logs and result."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .core import ExperimentResult, canonical


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

    def __init__(self, run_dir, timeout: float, command=None):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.run_dir = Path(run_dir).resolve()
        self.timeout = timeout
        self.command = command

    def details(self):
        return {"timeout": self.timeout, "run_dir": str(self.run_dir)}

    def execute(self, task, spec, trial_id):
        folder = self.run_dir / f"trial-{trial_id}"
        folder.mkdir(parents=True, exist_ok=True)
        input_path, result_path = folder / "input.json", folder / "result.json"
        out_path, err_path = folder / "stdout.log", folder / "stderr.log"
        if result_path.exists():
            raise RuntimeError("trial result already exists; refusing to overwrite")
        payload = task.worker_payload(spec)
        input_path.write_text(canonical(payload) + "\n")
        artifacts = tuple(str(p.relative_to(self.run_dir)) for p in
                          (input_path, out_path, err_path, result_path))
        start = time.monotonic()
        with out_path.open("wb") as out, err_path.open("wb") as err:
            command = (self.command(input_path, result_path) if self.command else
                       task.worker_command(input_path, result_path))
            proc = subprocess.Popen(command,
                                    stdout=out, stderr=err, start_new_session=True)
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
            data = json.loads(result_path.read_text())
            result = ExperimentResult(**data)
        except (OSError, ValueError, TypeError) as exc:
            return ExperimentResult("failed", None, {"exit_code": code}, duration,
                                    "WorkerResult", str(exc), artifacts)
        return ExperimentResult(result.status, result.objective,
                                {**result.metrics, "exit_code": code}, duration,
                                result.error_type, result.error_message, artifacts)
