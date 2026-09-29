# JevResearch

**Which experiment should we train next?**

Training every plausible configuration is expensive. JevResearch tests a cheaper decision layer: a task or classical optimizer proposes valid experiments, a fast Jev controller can choose one, and the measured result becomes the next piece of evidence. Every proposal, decision, and outcome is saved so the comparison can be replayed and audited.

The question is concrete: **given the same training budget, can Jev choose more useful experiments than random selection or a classical optimizer acting alone?** The longer-term question is when a fast controller should ask a slower, generative model to invent a new direction. That escalation is a research goal, not a feature of the current system.

The constraint is part of the experiment. TypeSafe [describes Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) as a System One model that returns typed decisions in parallel instead of generating a step-by-step text response. Here, it receives a compact experiment history and a fixed set of choices, with no external search or explicit multi-step reasoning loop. **Does it have enough AutoML knowledge to rank those choices, or does useful selection need richer domain information or slower reasoning?** A weak result could reflect that mismatch; it would not, by itself, settle the broader question of fast decision models.

## The loop

```text
history + remaining budget
           │
           ▼
task-defined search space → candidate generator (operators, random, or TPE)
           │
           ▼
      persisted offer
           │
           ▼
    random or Jev selects one
           │
           ▼
immutable experiment → train from scratch → validation score
           │
           └──────── durable history ────────↺
```

The task defines what is valid; the candidate generator builds complete experiment specs; the controller only chooses among them. The selected trial is recorded *before* execution. A supervised worker returns a result or failure, and the next search state is rebuilt from the ledger. This keeps proposal quality, decision quality, and training outcomes separable.

Jev's default `next-trial-v2` request gives it the named objective and fixed training protocol, a bounded set of past configurations with their measured results, and scale-aware comparisons for each offered candidate. The earlier `next-trial-v1` request remains available through `--question-version`; the version is saved with each campaign.

## What you can study today

The real-data task is CIFAR-10 with one fixed small ConvNet. Each hyperparameter configuration starts from scratch and trains for the same number of epochs. Search maximizes accuracy on a fixed 45,000/5,000 train/validation split; the official test set is never used during search. The current mixed domain varies learning rate, weight decay, and SGD versus AdamW. Model and data protocol stay fixed so the selector is the variable under study.

The available comparisons are:

| Proposal policy | Who chooses the training trial? | Why it matters |
| --- | --- | --- |
| Local operators or global random pool | Random or Jev | Is selection useful without an adaptive proposer? |
| TPE pool | Random or Jev | Does Jev add value *on top of* TPE proposals? |
| Single TPE proposal | Automatic | How does the hybrid compare with ordinary TPE? |
| CMA-ES, numeric domain | Automatic | A separate classical baseline on a compatible space. |

The TPE-pool comparison is the main selector test: both controllers use the same proposal policy and pool width. Single TPE is a useful practical comparator, but it changes the number of proposals as well as the selector. Studies record best-so-far accuracy against both trial count and observed active time, including controller overhead.

**Early evidence is modest.** In an exploratory 18-seed CIFAR-10 study with 32 trials per arm and 10 epochs per trial, final mean best validation accuracy was 69.31% for Jev + TPE pool, 69.10% for random + TPE pool, and 69.21% for single TPE. The Jev–random difference was about +0.21 percentage points, with uncertainty spanning no benefit. The current result establishes a working comparison, not a reliable accuracy advantage. The small fixed model and narrow search space are deliberate controls; whether the selector helps on richer decisions remains open.

## Run it

Python 3.10+ is required. Try the durable search loop without ML dependencies:

```bash
python -m pip install -e .
python -m jevresearch run --db runs/synthetic.sqlite --budget 6 --seed 7
python -m jevresearch show --db runs/synthetic.sqlite --session 1
```

For a bounded **real CIFAR-10** run, install the vision and classical-search extras. The example downloads CIFAR-10 if it is not already cached; omit `--download` when using an existing cache.

```bash
python -m pip install -e '.[vision,classical]'
python -m jevresearch cifar-run \
  --data-dir data/cifar10 --download --run-dir runs/cifar-smoke \
  --device cpu --epochs 1 --batch-size 256 --budget 2 --seed 7 \
  --proposal-strategy tpe-pool --proposal-domain cifar-mixed-v1 \
  --candidate-limit 4 --controller random
```

This two-trial command checks the pipeline; TPE needs more observations before it adapts. For a paired Jev/random/single-TPE study, copy the [versioned study recipe](docs/examples/phase5b-tpe-pool.json) and set the epoch, trial, and active-time budgets. Set the Jev arm's `max_api_calls` to at least `trial_budget - 1` so it can make one decision for each non-baseline trial. Install the `jev` extra and provide `TYPESAFE_API_KEY` through your environment for live Jev calls. On CUDA, set `CUBLAS_WORKSPACE_CONFIG=:16:8` before launching Python to satisfy PyTorch's deterministic-mode requirement.

```bash
python -m pip install -e '.[jev]'
python -m jevresearch study run \
  --spec path/to/your-study.json \
  --data-dir data/cifar10 --output-root runs/tpe-pool-study
python -m jevresearch study report \
  --spec path/to/your-study.json \
  --data-dir data/cifar10 --output-root runs/tpe-pool-study
```

Campaigns can be resumed. The SQLite ledger retains immutable specs, ordered offers, controller inputs and decisions, results and failures, seeds, timings, and source identity. The runner checks that the executable source, dataset, and split still match before resuming; Optuna state is reconstructed from this ledger rather than kept in a second optimizer database.

## Where this goes

JevResearch is built around a modality-independent runner: `core` owns the search loop, `operators` and `tasks` define valid experiments, `controllers` select, `execution` supervises workers, and `storage` records history. The [architecture](docs/architecture.md) describes the contracts and the planned path toward new tasks and optional generative-model escalation. The [phase briefs](docs/) document the implemented steps and their acceptance gates.

The near-term research work is to test selection on decisions with more meaningful variation than a small ConvNet hyperparameter sweep, while keeping the same auditable, paired comparison. The point of the project is to find **when fast decisions are enough, and when deeper reasoning earns its cost**.
