"""Sequential, resumable CIFAR study scheduling."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from ..controllers.jev import JevController, TypeSafeSDKTransport
from ..controllers.random import RandomController
from ..execution.ownership import study_lock
from ..execution.subprocess import SubprocessExecutor
from ..source import identity
from ..storage.history import InvariantError, Store
from ..tasks.vision.cifar10 import CifarTask
from .candidate_generator import CandidateGenerator
from .study import StudyArm, StudySpec
from .runner import Runner
from .experiment import ExperimentSpec, canonical


class StudyRunner:
    def __init__(self, spec_path, *, data_dir=None, output_root=None):
        self.spec, self.overrides = StudySpec.load(spec_path, data_dir=data_dir,
                                                   output_root=output_root)
        self.root = Path(self.spec.output_root)
        self.source_digest = identity()["digest"]

    def _controller(self, arm: StudyArm):
        if arm.controller == "random":
            return RandomController()
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("TYPESAFE_API_KEY unavailable")
        return JevController(TypeSafeSDKTransport(arm.api_timeout, arm.sdk_retries),
                             model=arm.model, max_calls=arm.max_api_calls)

    def _runner(self, store: Store, arm: StudyArm, seed: int):
        controller = self._controller(arm)
        import torch
        import torchvision

        if self.spec.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        run_dir = self.root / arm.id / f"seed-{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        task = CifarTask(self.spec.data_dir, run_dir, split_seed=self.spec.split_seed,
                         epochs=self.spec.epochs, batch_size=self.spec.batch_size,
                         device=self.spec.device, fixture=self.spec.task == "cifar10_fixture")
        if task.protocol != self.spec.protocol:
            raise InvariantError("task protocol differs from study specification")
        executor = SubprocessExecutor(run_dir, self.spec.trial_timeout_seconds,
                                      safe_store=store,
                                      checkpoint_policy=self.spec.checkpoint_policy)
        return Runner(store, task, controller, executor,
                      CandidateGenerator(self.spec.candidate_limit))

    def _active(self, store: Store, study_id: int):
        rows = store.activity(study_id)
        observed = sum(row["finished_at"] - row["started_at"] for row in rows
                       if row["status"] == "completed")
        unknown_upper = sum(max(0, time.time() - row["started_at"])
                            for row in rows if row["status"] == "unknown")
        return observed, unknown_upper, sum(row["status"] == "unknown" for row in rows)

    def _checkpoints(self, store: Store, sid: int, run_dir: Path):
        if self.spec.checkpoint_policy != "best":
            return
        state = store.state(sid)
        for row in store.trials(sid):
            trial_id = row["id"]
            folder = run_dir / f"trial-{trial_id}"
            if row["status"] == "running":
                continue
            for temp in folder.glob("*.tmp") if folder.exists() else ():
                temp.unlink()
            checkpoint = folder / "model.pt"
            manifest_path = folder / "checkpoint-manifest.json"
            if row["status"] != "completed":
                for path in (checkpoint, manifest_path):
                    if path.exists():
                        path.unlink()
                continue
            if not checkpoint.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            spec = ExperimentSpec(**json.loads(row["spec"]))
            if sha != manifest["sha256"] or manifest["spec_fingerprint"] != spec.fingerprint:
                raise InvariantError("checkpoint manifest integrity failed")
            if not store.db.execute("SELECT 1 FROM checkpoints WHERE trial_id=?", (trial_id,)).fetchone():
                store.record_checkpoint(trial_id, str(checkpoint.relative_to(self.root)), sha)
            if row["number"] != state.best_trial_id:
                checkpoint.unlink()
                manifest_path.unlink()
                store.prune_checkpoint(trial_id)

    def run(self, max_new_trials: int | None = None):
        if max_new_trials is not None and max_new_trials < 0:
            raise ValueError("max_new_trials must be nonnegative")
        with study_lock(self.root):
            store = Store(self.root / "study.sqlite")
            try:
                study_id = store.register_study(self.spec.normalized(),
                                                self.spec.fingerprint(self.source_digest),
                                                self.source_digest, self.overrides)
                manifest_path = self.root / "resolved-study.json"
                if not manifest_path.exists():
                    manifest_path.write_text(canonical({"spec": self.spec.normalized(),
                        "source_digest": self.source_digest, "overrides": self.overrides}) + "\n")
                store.mark_unknown_activity(study_id)
                new = 0
                for seed in self.spec.seeds:
                    for arm in self.spec.arms:
                        if max_new_trials is not None and new >= max_new_trials:
                            return study_id
                        member = store.study_member(study_id, arm.id, seed)
                        try:
                            runner = self._runner(store, arm, seed)
                        except (ImportError, FileNotFoundError, RuntimeError) as exc:
                            reason = type(exc).__name__ + ": " + str(exc)
                            store.block_member(study_id, arm.id, seed, reason)
                            continue
                        store.clear_block(study_id, arm.id, seed)
                        sid = member["session_id"] if member and member["session_id"] else runner.start(
                            self.spec.trial_budget, seed, (study_id, arm.id, seed))
                        while store.session(sid)["status"] != "stopped":
                            observed, unknown, count = self._active(store, study_id)
                            if observed >= self.spec.active_time_budget_seconds or (
                                    count and observed + unknown >= self.spec.active_time_budget_seconds):
                                store.pause(sid, "active-time cap exhausted or unknown interval may exhaust cap")
                                break
                            if max_new_trials is not None and new >= max_new_trials:
                                return study_id
                            before = sum(row["status"] in ("completed", "failed", "interrupted")
                                         for row in store.trials(sid))
                            activity_id = store.start_activity(study_id, sid)
                            try:
                                runner.run(sid, 1)
                            finally:
                                completed_rows = [row for row in store.trials(sid)
                                                  if row["status"] in ("completed", "failed", "interrupted")]
                                after = len(completed_rows)
                                store.finish_activity(activity_id, completed_rows[-1]["number"]
                                                      if after > before else None)
                            self._checkpoints(store, sid, Path(runner.task.run_dir))
                            if after == before:
                                break
                            new += 1
                return study_id
            finally:
                store.close()
