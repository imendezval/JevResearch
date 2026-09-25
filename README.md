# JevResearch

Phase 1 provides a CPU-only, deterministic synthetic search task and a durable SQLite experiment loop. It uses Python's standard library at runtime. The task's optimum is `x=2, y=-1`, with objective `0` (larger is better). Each experiment changes one coordinate by one step. Trial 0 is the baseline.

The implementation follows the package boundaries in [the architecture](docs/architecture.md): `core` owns specs, state, candidate generation, and the runner; `controllers` selects; `operators` defines explicit config moves; `tasks` owns data and training protocols; `execution` runs trials; and `storage` persists history. The CIFAR worker belongs to its task, and a process task supplies its worker command to the generic executor.

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

Phase 2 adds a fixed small CNN trained from scratch on official CIFAR-10. The search uses only the 50,000-image official training portion, split reproducibly into 45,000 training and 5,000 validation images; it never loads the official test portion. Its objective is final validation top-1 accuracy. The baseline uses SGD (learning rate 0.01, weight decay 0.0001, momentum 0.9). Operators change learning rate by a factor of two within [0.0001, 0.1], set weight decay to 0 or 0.001, or swap between SGD and AdamW. Model, preprocessing, scheduler (`none`), epochs, batch size, and device are frozen per session. Both train and validation use `ToTensor` and fixed CIFAR normalization; there is no augmentation in this protocol. Accuracy after one epoch is a wiring check, not a claim of strong CIFAR-10 performance.

Install the optional, compatible vision packages with `pip install '.[vision]'`. For a cached official torchvision dataset:

```bash
PYTHONPATH=src python -m jevresearch cifar-run --data-dir /path/to/cache --run-dir runs/cifar-example --device cpu --epochs 1 --batch-size 256 --budget 2 --seed 7 --timeout 900
PYTHONPATH=src python -m jevresearch export --db runs/cifar-example/history.sqlite --session 1 --output runs/cifar-example/history.json
```

Add `--download` to `cifar-run` only when you explicitly want torchvision to download official CIFAR-10 into `--data-dir`. The run directory holds the SQLite database, exact stratified split indices, trial inputs/results, and stdout/stderr logs. `--run-dir` should be unique per split seed and protocol. The split and dataset content hashes are checked again on resume and inside the worker. Each trial runs in a supervised child process; a timeout or failure consumes its attempt. Trial seeds depend only on the search seed and trial index, regardless of the controller's random choices.

For an offline CPU wiring check with generated CIFAR-shaped images, use `--fixture`; its task and protocol are explicitly labeled as a fixture:

```bash
PYTHONPATH=src python -m jevresearch cifar-run --data-dir /tmp/jev-fixture-data --run-dir runs/fixture-example --fixture --device cpu --epochs 1 --batch-size 16 --budget 2 --seed 7 --timeout 120
```

Resume with `PYTHONPATH=src python -m jevresearch cifar-resume --db runs/cifar-example/history.sqlite --session 1`. `show` gives a compact status view; `export` emits all trial specs, operators, results, failures, timing, device information, parent links, source identity, and best-so-far values as JSON. Old Phase 1 databases migrate in place and remain readable. Source changes prevent resuming a campaign with a different executable. CUDA determinism is requested, but exact reproducibility across devices and PyTorch versions is not guaranteed. GPU-active time is reported as unavailable unless measured; wall time is never presented as GPU-active time.

Phase 3 adds a Jev controller that selects one ID from the same bounded candidate generator as random. Install the optional SDK with `pip install '.[jev]'` and set `TYPESAFE_API_KEY` in your environment. The integration uses the versioned `jev-1.13.0` model by default, one typed `Choice`, an explicit 10-second request timeout, at most one SDK retry, and at most one logical Jev call per command by default. A live synthetic wiring run is:

```bash
PYTHONPATH=src python -m jevresearch run --db runs/jev-synthetic.sqlite --budget 2 --seed 7 --controller jev --model jev-1.13.0 --api-timeout 10 --sdk-retries 1 --max-api-calls 1
```

For a paired CIFAR-10 wiring comparison, use separate run directories and the **same** official data cache, split seed, search seed, budget, epoch count, batch size, device, fixed model, and source revision. Change only `--controller` and Jev API settings:

```bash
PYTHONPATH=src python -m jevresearch cifar-run --data-dir /path/to/cache --run-dir runs/random-pair --device cpu --epochs 1 --batch-size 256 --budget 2 --seed 7 --split-seed 1729 --timeout 900 --controller random
PYTHONPATH=src python -m jevresearch cifar-run --data-dir /path/to/cache --run-dir runs/jev-pair --device cpu --epochs 1 --batch-size 256 --budget 2 --seed 7 --split-seed 1729 --timeout 900 --controller jev --model jev-1.13.0 --api-timeout 10 --sdk-retries 1 --max-api-calls 1
PYTHONPATH=src python -m jevresearch compare --random-db runs/random-pair/history.sqlite --random-session 1 --jev-db runs/jev-pair/history.sqlite --jev-session 1 --output runs/pair-comparison.json
```

`export` includes the saved, sanitized request, requested and response model IDs, SDK version, validated option probabilities and confidence, usage if supplied, call latency, failures, and best objective versus attempted trials and wall time. These Choice probabilities describe the provider's selection distribution; they are not calibrated probabilities of validation improvement. Lower-level SDK retry count and money are unavailable unless separately observed and priced. A failed call pauses the session with its saved offer; `resume` or `cifar-resume` may make another logged attempt. No random fallback occurs. Offline fake transports are test doubles and are labeled non-live in exports. The official test set stays untouched. A two-trial result only verifies integration; claims about relative search quality require multiple paired seeds with the full variability reported, and any question wording changes must be versioned and disclosed.

Phase 4 adds a local study scheduler. Save a versioned JSON specification like this outside the repository's tracked source:

```json
{
  "version": 1,
  "task": "cifar10_fixture",
  "protocol": "fixture-v1",
  "data_dir": "/tmp/jev-fixture-data",
  "output_root": "/tmp/jev-fixture-study",
  "split_seed": 1729,
  "epochs": 1,
  "batch_size": 16,
  "device": "cpu",
  "trial_timeout_seconds": 120,
  "active_time_budget_seconds": 120,
  "trial_budget": 2,
  "candidate_limit": 8,
  "checkpoint_policy": "none",
  "compute_hourly_usd": null,
  "arms": [{"id": "random_a", "controller": "random"},
           {"id": "random_b", "controller": "random"}],
  "seeds": [7, 11]
}
```

Run `PYTHONPATH=src python -m jevresearch study run --spec /tmp/fixture-study.json`, repeat that command to verify idempotence, or use `study resume` after a bounded `--max-new-trials 1` run. `PYTHONPATH=src python -m jevresearch study report --spec /tmp/fixture-study.json` writes `report.json` beside the study registry. `--data-dir` and `--output-root` override local paths and are recorded in the resolved manifest; changing them after a study starts requires a new output root. Use `task: "cifar10"`, `protocol: "official-train-v1"`, the official data cache, and a versioned Jev arm for a real paired study. A missing key, dataset, SDK, or device is reported as a blocked arm; the fixture is never substituted for official data.

Study execution is single-host and Linux-specific. It holds an OS lock across scheduling, records worker process identity, and gives child workers a parent-death signal. A resumed study verifies saved worker results before accepting them; interrupted trials still count once. The active-time cap applies between trials and never cuts a training run short. Calendar elapsed includes pauses; observed active time does not. Unknown crash intervals are reported separately and can conservatively block more work. `checkpoint_policy: "best"` retains only the current best completed model and its provenance manifest, with intentional pruning recorded in SQLite. The compute hourly rate, if supplied, produces an occupancy **estimate**; GPU-active time, lower-level SDK retries, and Jev monetary cost remain unavailable without separate validated measurements or an archived pricing snapshot.

Phase 5 adds versioned global CIFAR search domains. Install `pip install '.[vision,classical]'` for Optuna 5.0.0 and CMA-ES 0.12.0. `cifar-mixed-v1` has SGD/AdamW, log learning rate, and an explicit zero-or-positive weight-decay branch. `cifar-sgd-numeric-v1` fixes SGD and uses two positive log parameters; standard CMA-ES is accepted only on this numeric domain. Every global proposal is a full configuration trained from scratch and has no parent trial. The earlier local-move random/Jev comparison remains separate.

`cifar-run` accepts `--proposal-strategy global-random|global-pool|tpe|tpe-pool|cmaes` and requires `--proposal-domain`. `global-random`, `tpe`, and `cmaes` use the single-candidate selector; `global-pool` lets `--controller random|jev` select from random proposals. `tpe-pool` lets those controllers select from 2–8 TPE proposals on `cifar-mixed-v1`; set the fixed width with `--candidate-limit` (default 8). Example bounded wiring run:

```bash
PYTHONPATH=src python -m jevresearch cifar-run --data-dir /path/to/cache --run-dir /tmp/jev-tpe-wiring --device cpu --epochs 1 --batch-size 256 --budget 2 --seed 7 --proposal-strategy tpe --proposal-domain cifar-mixed-v1
PYTHONPATH=src python -m jevresearch cifar-run --data-dir /path/to/cache --run-dir /tmp/jev-cma-wiring --device cpu --epochs 1 --batch-size 256 --budget 2 --seed 7 --proposal-strategy cmaes --proposal-domain cifar-sgd-numeric-v1
PYTHONPATH=src python -m jevresearch cifar-run --data-dir /path/to/cache --run-dir /tmp/jev-tpe-pool-wiring --device cpu --epochs 1 --batch-size 256 --budget 2 --seed 7 --proposal-strategy tpe-pool --proposal-domain cifar-mixed-v1 --candidate-limit 4 --controller jev
```

Predeclared multiseed study recipes are [mixed](docs/examples/phase5-mixed.json), [numeric](docs/examples/phase5-numeric.json), and [TPE selector bridge](docs/examples/phase5b-tpe-pool.json). Pass the real cache with `--data-dir /path/to/cache`; use a fresh `--output-root` for each source revision or study specification. The bridge recipe pairs Jev and random on the same TPE pool width and includes single-TPE as a comparator. Jev versus random on `tpe-pool` is the selector ablation; single-TPE also changes proposal count and selector cost. These multiseed runs require an explicit compute budget and are not launched as part of installation. Reports include saved offer audits, trajectories, timing, and estimated resource cost when a rate is supplied.

Optuna ask/tell is replayed deterministically from JevResearch's saved baseline, offers, trials, and outcomes; no second optimizer database is used. The pinned Optuna version and saved proposal identity are checked on resume. Failed trials are told as `FAIL` without a fabricated objective. TPE uses ten completed observations before adapting; CMA-ES uses one startup observation and a population of four. A budget-two run checks only wiring, not optimizer quality. Fixture data is only an integration check, and this tiny one-epoch CIFAR domain is not evidence of general AutoML performance. The official test set is never used during search.

For `tpe-pool-v1`, Optuna 5.0.0 runs with `constant_liar=True`; accepted members stay running until the entire pool is drawn. Rejected duplicates/invalid draws and declined untrained members are internal sampler `FAIL` records without objective values. Only the selected candidate becomes a training trial. Exports and study reports retain each offer's candidate order, slot, Optuna trial number, rejection records, selected ID, decision audit, and measured outcome. Proposal duration is currently unmeasured. A CMA-ES/Jev hybrid needs a separate population-evaluation and budget rule and is deferred.
