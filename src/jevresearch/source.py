"""Executable source identity: Git commit for provenance, content digest for compatibility."""

import hashlib
import inspect
import platform
import subprocess
from pathlib import Path


def identity(task=None) -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = sorted(root.glob("src/jevresearch/**/*.py")) + sorted(root.glob("pyproject.toml"))
    if task is not None:
        task_file = inspect.getsourcefile(type(task))
        if task_file is None:
            raise ValueError("task source file is unavailable for source identity")
        task_path = Path(task_file).resolve()
        if task_path not in paths:
            paths.append(task_path)
    hash_ = hashlib.sha256()
    for path in paths:
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path)
        hash_.update(label.encode() + b"\0")
        hash_.update(path.read_bytes())
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"digest": hash_.hexdigest(), "git_commit": commit,
            "python": platform.python_version(), "package_version": "0.1.0"}
