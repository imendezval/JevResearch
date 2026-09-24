"""Single-host study ownership and Linux worker identity checks."""

import ctypes
import fcntl
import os
import signal
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def study_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".study.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("study is already owned by another local process") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def parent_death_guard(parent_pid: int):
    # Called in the single-threaded forked child before exec. Closing the
    # spawn-to-record gap is why this runs before the worker can train.
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0:
        os._exit(127)
    if os.getppid() != parent_pid:
        os._exit(127)


def process_identity(pid: int):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[0], int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def same_worker_alive(pid: int, start_ticks: int) -> bool:
    identity = process_identity(pid)
    return identity is not None and identity[0] != "Z" and identity[1] == start_ticks
