# JevResearch Phase 1

Phase 1 provides a CPU-only, deterministic synthetic search task and a durable SQLite experiment loop. It uses Python's standard library at runtime. The task's optimum is `x=2, y=-1`, with objective `0` (larger is better). Each experiment changes one coordinate by one step. Trial 0 is the baseline.

From the repository root, run:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m jevresearch run --db /tmp/jevresearch-full.sqlite --budget 6 --seed 7
PYTHONPATH=src python -m jevresearch run --db /tmp/jevresearch-pause.sqlite --budget 6 --seed 7 --max-new-trials 2
PYTHONPATH=src python -m jevresearch resume --db /tmp/jevresearch-pause.sqlite --session 1
PYTHONPATH=src python -m jevresearch show --db /tmp/jevresearch-pause.sqlite --session 1
```

`--budget` is the session's total attempted trials, including the baseline and failures. `--max-new-trials` pauses after that many newly executed trials without changing the total budget. `show` reports status, stop reason, attempted count, best objective, trials, results, and offer selections. Trial IDs in specs and search state are session-local trial numbers, starting at zero. The SQLite database also retains full specs, ordered offers, controller input snapshots, RNG positions, timestamps, and source identity.

On resume, a saved unselected offer is reused, a selected pending trial is executed, and a trial left running is recorded as interrupted and counted once. A failed baseline ends the session. A content digest of package source, `pyproject.toml`, and any task source outside the package prevents resuming after executable changes; the Git commit is stored for provenance. Results are deterministic for this task, but the framework does not promise bitwise reproducibility across hardware or future training tasks.

Phase 2 can add the CIFAR-10 task through the task contract while retaining this loop. It is outside this phase.
