import json
import multiprocessing
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path

from jevresearch.controllers.random import RandomController
from jevresearch.core.runner import Runner
from jevresearch.execution.ownership import same_worker_alive, study_lock
from jevresearch.execution.subprocess import SubprocessExecutor
from jevresearch.storage.history import Store
from jevresearch.tasks.synthetic import SyntheticTask
from jevresearch.tasks.vision.cifar10 import CifarTask


def hold_lock(root, ready):
    with study_lock(Path(root)):
        ready.set()
        time.sleep(10)


class SlowTask(SyntheticTask):
    def worker_payload(self, spec):
        import dataclasses
        return {"spec": dataclasses.asdict(spec)}

    def worker_command(self, input_path, result_path):
        return [sys.executable, "-c", "import time; time.sleep(20)"]


def run_slow(db, root, sid):
    store = Store(db)
    runner = Runner(store, SlowTask(), RandomController(),
                    SubprocessExecutor(root, 30, safe_store=store))
    runner.run(sid)


class StudyOwnershipTests(unittest.TestCase):
    def test_second_process_cannot_take_live_study_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = multiprocessing.Event()
            child = multiprocessing.Process(target=hold_lock, args=(tmp, ready))
            child.start()
            try:
                self.assertTrue(ready.wait(3))
                with self.assertRaisesRegex(RuntimeError, "already owned"):
                    with study_lock(Path(tmp)):
                        pass
            finally:
                child.terminate()
                child.join(3)
            with study_lock(Path(tmp)):
                pass

    def test_parent_death_kills_worker_and_resume_counts_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "history.sqlite"
            store = Store(db)
            task = SlowTask()
            runner = Runner(store, task, RandomController(),
                            SubprocessExecutor(root, 30, safe_store=store))
            sid = runner.start(1, 7)
            child = multiprocessing.Process(target=run_slow, args=(str(db), str(root), sid))
            child.start()
            worker = None
            deadline = time.time() + 5
            while time.time() < deadline:
                worker = store.worker(store.trials(sid)[0]["id"])
                if worker:
                    break
                time.sleep(0.05)
            try:
                self.assertIsNotNone(worker)
                self.assertTrue(same_worker_alive(worker["pid"], worker["start_ticks"]))
                os.kill(child.pid, signal.SIGKILL)
                child.join(3)
                deadline = time.time() + 3
                while same_worker_alive(worker["pid"], worker["start_ticks"]) and time.time() < deadline:
                    time.sleep(0.05)
                self.assertFalse(same_worker_alive(worker["pid"], worker["start_ticks"]))
                runner.run(sid)
                self.assertEqual([row["status"] for row in store.trials(sid)], ["interrupted"])
            finally:
                if child.is_alive():
                    child.kill()
                    child.join(3)
                store.close()

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch") and
                         __import__("importlib").util.find_spec("torchvision"),
                         "vision dependencies are optional")
    def test_completed_worker_manifest_recovers_without_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root / "history.sqlite")
            task = CifarTask(root / "data", root / "run", fixture=True, batch_size=16)
            runner = Runner(store, task, RandomController(),
                            SubprocessExecutor(task.run_dir, 30, safe_store=store))
            sid = runner.start(1, 7)
            runner.run(sid)
            trial = store.trials(sid)[0]
            objective = json.loads(trial["result"])["objective"]
            with store.db:
                store.db.execute("UPDATE trials SET status='running',result=NULL WHERE id=?", (trial["id"],))
            runner.run(sid)
            recovered = store.trials(sid)[0]
            self.assertEqual(recovered["status"], "completed")
            self.assertEqual(json.loads(recovered["result"])["objective"], objective)
            self.assertEqual(len(store.trials(sid)), 1)
            store.close()
