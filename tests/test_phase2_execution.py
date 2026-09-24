import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from jevresearch.controller import RandomController
from jevresearch.runner import Runner
from jevresearch.storage import Store
from jevresearch.subprocess_execution import SubprocessExecutor
from jevresearch.tasks.vision.cifar10 import CifarTask


class Phase2ExecutionTests(unittest.TestCase):
    def test_old_unversioned_db_migrates_without_losing_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.db"
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, settings TEXT NOT NULL, source TEXT NOT NULL, rng_state TEXT NOT NULL, status TEXT NOT NULL, stop_reason TEXT, created_at REAL NOT NULL)")
            db.execute("INSERT INTO sessions VALUES (1, '{}', '{}', '[]', 'stopped', 'old', 1.0)")
            db.commit()
            db.close()
            store = Store(path)
            self.assertEqual(store.db.execute("PRAGMA user_version").fetchone()[0], 2)
            self.assertEqual(store.session(1)["stop_reason"], "old")
            self.assertIn("exit_code", {r[1] for r in store.db.execute("PRAGMA table_info(trials)")})
            store.close()

    def test_child_exit_and_timeout_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = CifarTask(root / "data", root / "run", fixture=True)
            store = Store(root / "history.sqlite")
            exit_executor = SubprocessExecutor(root / "run", 5,
                command=lambda _i, _o: [sys.executable, "-c", "import sys; sys.exit(17)"])
            runner = Runner(store, task, RandomController(), exit_executor)
            sid = runner.start(2, 4)
            runner.run(sid)
            result = json.loads(store.trials(sid)[0]["result"])
            self.assertEqual(result["error_type"], "ChildExit")
            self.assertEqual(result["metrics"]["exit_code"], 17)
            self.assertEqual(store.trials(sid)[0]["exit_code"], 17)
            self.assertEqual(store.session(sid)["stop_reason"], "baseline failed")
            self.assertTrue((root / "run" / result["artifacts"][2]).exists())

            pidfile = root / "pid.txt"
            code = f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(60)"
            timeout_executor = SubprocessExecutor(root / "run", 0.5,
                command=lambda _i, _o: [sys.executable, "-c", code])
            runner = Runner(store, task, RandomController(), timeout_executor)
            sid = runner.start(2, 5)
            runner.run(sid)
            result = json.loads(store.trials(sid)[0]["result"])
            self.assertEqual(result["error_type"], "Timeout")
            self.assertTrue(pidfile.exists())
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pidfile.read_text()), 0)
            store.close()

    def test_pending_and_running_resume_counts_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = CifarTask(root / "data", root / "run", fixture=True)
            store = Store(root / "history.sqlite")
            executor = SubprocessExecutor(root / "run", 5,
                command=lambda _i, _o: [sys.executable, "-c", "import sys; sys.exit(2)"])
            runner = Runner(store, task, RandomController(), executor)
            sid = runner.start(3, 8)
            self.assertEqual(store.trials(sid)[0]["status"], "pending")
            runner.run(sid)
            self.assertEqual(len(store.trials(sid)), 1)
            sid2 = runner.start(3, 9)
            trial_id = store.trials(sid2)[0]["id"]
            store.running(trial_id)
            runner.run(sid2)
            self.assertEqual(len(store.trials(sid2)), 1)
            self.assertEqual(store.trials(sid2)[0]["status"], "interrupted")
            store.close()

    def test_official_split_counts_without_download(self):
        from jevresearch.tasks.vision.cifar10.task import stratified_indices
        labels = [label for label in range(10) for _ in range(5000)]
        train, val = stratified_indices(labels, 23, 4500)
        self.assertEqual((len(train), len(val)), (45000, 5000))
        self.assertFalse(set(train) & set(val))
        self.assertEqual({labels[i] for i in val}, set(range(10)))


if __name__ == "__main__":
    unittest.main()
