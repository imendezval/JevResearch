# JevResearch Architecture

## 1. Project Overview

**JevResearch** is a modular AutoML / autonomous experimentation framework built around a simple separation of responsibilities:

- **Jev** = fast decision/control layer. Jev refers to TypeSafe AI’s System One model for typed, probabilistic decision-making. JevController integrates that model to select among candidate experiments and, later, decide when to request autoregressive LM assistance.
- **Classical optimizers** = optional numerical search layer
- **Autoregressive LLMs** = optional invention/code-generation layer
- **Training/evaluation metrics** = ground truth

The central idea is to avoid using an expensive autoregressive LM for every experiment-selection step.

Instead, Jev receives structured experiment state and chooses among available actions. An autoregressive LM is only introduced when the system needs to expand the search space, invent a new operator, or modify code in an open-ended way.

The intended long-term architecture is:

```text
experiment history
        ↓
candidate generation
        ↓
      Jev
   ↙       ↘
search     escalate
  ↓           ↓
known       autoregressive LM
operator      ↓
  │       new idea / operator / code
  └───────────┘
        ↓
    experiment
        ↓
     metric
        ↓
     history
```

The repository must be designed so that:

1. Jev-only search works independently.
2. Classical optimization can be added without restructuring the system.
3. Autoregressive LM escalation can be added later as an optional component.
4. New ML modalities/tasks can be added through adapters instead of modifying the search core.

---

## 2. Core Design Principle

The controller must know as little as possible about the underlying ML task.

Every domain should expose the same generic concepts:

```text
State
Candidates
Result
Objective
Constraints
```

The task owns domain-specific knowledge.

The candidate generator instantiates concrete experiments from the task's valid operators/search space.

The controller owns search decisions.

The experiment runner owns execution.

The storage layer owns history.

This separation is the most important architectural requirement.

---

## 3. Main System Variants

### Option A — Jev-only AutoML

A predefined operator library generates valid candidate experiments.

Example operators:

- learning-rate changes
- weight-decay changes
- optimizer swaps
- scheduler changes
- augmentation changes
- dropout
- model depth
- model width
- normalization changes

Loop:

```text
state + history
      ↓
candidate operators
      ↓
     Jev
      ↓
selected experiment
      ↓
train/evaluate
      ↓
objective metric
      ↓
history
```

Advantages:

- simple
- cheap
- fast
- easy to benchmark
- deterministic search space

Limitation:

- cannot invent ideas outside the predefined operator space

This should be the **first functional JevResearch system**.

---

### Option B — Jev + Classical Optimizer

Classical optimization handles numerical parameters while Jev handles higher-level decisions.

Possible optimizers:

- TPE
- CMA-ES
- SMAC
- Bayesian optimization

Example division:

```text
Jev:
- architecture family
- optimizer family
- search mode
- restart / exploit / explore
- operator selection

CMA-ES / TPE:
- learning rate
- weight decay
- dropout
- widths
- continuous/discrete numeric parameters
```

This is useful both as a practical system and as a strong research baseline.

---

### Option C — Jev + Autoregressive LM Escalation

Jev remains the default controller.

At each step it may either:

```text
1. select an existing candidate
2. request System-2 escalation
```

If escalation is requested, an autoregressive LM receives the experiment state/history and may:

- propose new experiments
- create new operators
- modify existing operators
- propose architectural changes
- eventually edit code directly

The new action becomes part of the candidate pool.

Conceptually:

```text
Jev = search/control
LLM = invention/code generation
metric = truth
```

The LM should **not** be required for normal search.

The interesting long-term question is:

> How little expensive System-2 reasoning is required for effective autonomous ML research?

---

# 4. Repository Structure

Current implementation structure (only active components are present):

```text
src/jevresearch/
├── core/
│   ├── state.py
│   ├── experiment.py
│   ├── candidate.py
│   ├── candidate_generator.py
│   ├── result.py
│   ├── objective.py
│   ├── runner.py
│   ├── study.py
│   ├── study_runner.py
│   └── study_report.py
│
├── controllers/
│   ├── base.py
│   ├── random.py
│   └── jev.py
│
├── operators/
│   ├── base.py
│   ├── hyperparams.py
│   └── optimizer.py
│
├── tasks/
│   ├── base.py
│   ├── synthetic.py
│   ├── vision/
│   │   └── cifar10/{task.py, model.py, worker.py}
│
├── execution/
│   ├── base.py
│   ├── local.py
│   ├── subprocess.py
│   └── ownership.py
│
├── storage/
│   └── history.py
├── cli.py
└── source.py
```

New controllers, operators, tasks, executors, storage adapters, agents, and analysis modules belong in these packages when implemented; do not add empty future modules.

---

# 5. Core Interfaces

## Task

A task represents the underlying ML problem.

The implemented task contract is:

```python
class Task:
    def baseline(self) -> dict: ...
    def proposals(self, state) -> list: ...
    def validate(self, config) -> None: ...
```

An inline task also defines `evaluate(spec)`. A process task supplies `worker_payload(spec)` and `worker_command(input_path, result_path)`; its worker trains and evaluates, returning a structured result. The generic executor supervises that command without importing the task.

A task should encapsulate:

- dataset
- model
- training loop
- evaluation
- valid mutation/operator space
- task-specific constraints

The controller should not need to know whether the task is CV, tabular, language modeling, segmentation, etc.

---

## CandidateGenerator

```python
class CandidateGenerator:
    def generate(self, task, state, seed, source_digest):
        ...
```

The task defines valid operators/search space; the generator instantiates concrete candidates; the controller only selects among them. Random and Jev should use the same generation rules, and persisted offers permit comparisons. Identical search seeds need not yield identical pools after their incumbents diverge.

Later, classical optimizers can propose parameter values through this generation interface rather than being forced into selection-only logic.

---

## Controller

```python
class LocalController:
    selection_mode = "local"
    def select(self, state, candidates, rng_state):
        ...
```

An audited controller declares `selection_mode = "audited"`, prepares a bounded request
from the saved offer, invokes its transport, then validates the returned candidate ID
before the runner commits a trial. The runner dispatches by this explicit mode.

Examples:

```text
RandomController
JevController
TPEController
CMAESController
HybridController
```

The controller receives generic experiment information and returns a decision.

---

## Candidate

A candidate is an executable proposed modification.

Suggested fields:

```text
id
operator_type
parameters
description
estimated_cost
metadata
```

Candidates should be serializable.

They should ideally be deterministic and reproducible.

---

## Result

A result represents one completed experiment.

Suggested fields:

```text
candidate_id
metrics
objective_value
runtime
gpu_time
memory
status
seed
artifacts
error
```

---

## ExperimentSpec

An experiment is an immutable `ExperimentSpec` containing:

```text
id
full_config
seed
dataset_split
code_version
parent_experiment_id
```

The executor receives this specification, separately from campaign-level `SearchState`. The initial experiment has no parent. For the MVP, operators modify the current incumbent/best configuration, and every experiment trains from scratch with no checkpoint inheritance.

---

## SearchState

This should contain enough information for any controller to reason about the search.

Example:

```text
current_best
current_config
experiment_history
recent_results
budget_remaining
constraints
available_candidates
search_metadata
```

Keep this generic and serializable.

---

# 6. Operators

Operators define how the search space changes.

Start with safe, explicit operators instead of arbitrary code edits.

Examples:

```text
MultiplyLearningRate
SetOptimizer
ChangeScheduler
IncreaseDepth
DecreaseDepth
IncreaseWidth
ChangeDropout
ChangeWeightDecay
ChangeAugmentation
ChangeNormalization
```

Important:

Operators should expose:

```text
name
parameter schema
validity check
apply()
description
```

This allows:

- Jev to choose between them
- classical optimizers to parameterize them
- autoregressive LMs to create new ones later
- easy experiment logging and ablations

The operator system is one of the main extensibility points.

---

# 7. Execution Model

The runner should follow a generic loop:

```text
initialize task
initialize candidate generator
initialize controller
load/create SearchState
persist baseline ExperimentSpec as a pending trial when creating a session

while session has work:
    if there is no pending trial:
        candidates = generator.generate(task, state)
        persist candidate pool and controller input
        if controller calls an external service:
            persist each sanitized logical decision attempt before the call
            persist response or error; keep failed offers available for resume
            decision = validated choice from the saved offer
        else:
            decision = controller.select(state, candidates)
        spec = selected candidate's immutable ExperimentSpec
        persist decision and selected pending trial atomically
    mark trial running
    result = executor.execute(task, spec, trial_id)
    persist result or failure atomically
    reload SearchState from history
```

The core runner should not contain task-specific logic.

The task owns its fixed per-trial training budget; the campaign budget counts attempted trials, including failures. The worker records training duration and the executor records trial wall duration. A Phase 4 study schedules campaigns sequentially under a local OS lock and a total observed active-time cap. Its intervals include controller, worker, and orchestration time once; calendar elapsed remains separate. A study worker has a parent-death guard, recorded process identity, and an atomic result manifest. Resume accepts a completed result only after identity and artifact checks, and never retries an interrupted trial.

---

# 8. First Target Domain

## Start with Computer Vision

Recommended MVP:

```text
CIFAR-10
one fixed small ConvNet
fixed training steps per experiment
fixed attempted-trial budget per campaign initially
```

Why CV first:

- experiments are cheap
- metrics are easy to interpret
- architecture and training choices are rich enough to make search meaningful
- hundreds of experiments are feasible
- easier to debug than RL
- richer than pure tabular HPO

Initial operator space:

```text
learning rate
optimizer
weight decay
```

Maximize validation accuracy on a fixed held-out validation split. Never use the test set during search; evaluate final selected configurations on the test set only after the campaign.

Architecture search comes after the basic Jev-vs-random comparison works cleanly. The broader operator examples above are future extensions, not MVP scope.

Aim for experiments on the order of roughly 1–3 minutes where possible.

---

# 9. Additional Modalities

Once the architecture is stable:

## Tabular
First Tabular.
Add later as a strong classical AutoML benchmark.

Useful because:

- TPE
- CMA-ES
- SMAC
- Bayesian optimization

are strong here.

If Jev can compete, complement, or efficiently route between them, the result is meaningful.

---

## Language Modeling

Integrate a Karpathy-style autoresearch task.

Autoresearch is especially valuable because it gives a direct comparison against:

```text
autoregressive LM every iteration
vs
Jev every iteration
vs
Jev + adaptive LM escalation
```

Objective can remain the task's real validation metric.

---

## Later

Possible future tasks:

```text
segmentation
object detection
time series
RL
robotics
multimodal systems
```

These should require new `Task` implementations, not search-core rewrites.

---

# 10. Implementation Phases

## Phase 1 — Core Abstractions and Synthetic Task

Implement:

```text
Task
CandidateGenerator
Controller
Candidate
Result
ExperimentSpec
SearchState
Runner
```

No Jev yet.

Goal:

> Prove that the architecture is genuinely domain-independent.

---

## Phase 2 — Minimal CV Benchmark

Implement:

```text
CIFAR-10
one fixed small ConvNet
fixed training budget and attempted-trial campaign budget
learning rate, weight decay, and optimizer operators only
RandomController
persistent logging and campaign resume support
```

Full loop:

```text
Task
→ candidate generator
→ candidates
→ random selection
→ train
→ metric
→ history
```

Goal:

> End-to-end infrastructure works reliably.

Do not add complexity before this works.

Reproducible seeds, complete experiment specifications, decision/result logging, failure recording, timings, and campaign state persistence are required from this phase (see Rule 6).

---

## Phase 3 — Jev-only System

Implement `JevController`.

Input should include:

```text
objective
constraints
current best
recent experiments
available candidates
remaining budget
```

Possible Jev outputs:

```text
selected_candidate
search_mode
expected_improvement
risk
confidence
```

Only `selected_candidate` is essential initially.

Goal:

> First real JevResearch system.

This is **Option A**.

---

## Phase 4 — Experiment Infrastructure

Before adding more intelligence, harden the framework.

Extend the Phase 1 persistence and reproducibility foundation with:

```text
reproducible seeds
config files
persistent experiment history
checkpointing
runtime tracking
GPU-time tracking
cost tracking
failure handling
OOM handling
resume support
basic plots
search trajectory visualization
```

This phase is extremely important.

Research conclusions are worthless if the experiment layer is unreliable.

---

## Phase 5 — Classical AutoML Baselines

Implement:

```text
Random
TPE
CMA-ES
possibly SMAC
```

Compare them under identical budgets.

Then explore combinations such as:

```text
Jev high-level routing
+
TPE/CMA-ES low-level parameter search
```

This enables **Option B**.

---

## Phase 6 — Autoregressive LM Escalation

Add the LM as a separate module.

Do not embed LLM logic into `JevController`.

Introduce an interface such as:

```python
class ResearchAgent:
    def propose(self, state):
        ...
```

Jev may produce:

```text
selected_candidate
OR
escalate
```

On escalation:

```text
state/history
    ↓
LLM agent
    ↓
new candidate / operator / code change
    ↓
candidate pool
```

Initially, restrict LLM output to generating structured operators/candidates.

Direct arbitrary code editing should come later.

This is **Option C**.

---

## Phase 7 — Multi-Domain Expansion

Add:

```text
tabular
language/autoresearch
```

Potentially later:

```text
segmentation
RL
other CV tasks
```

The goal is to validate that the core controller is not specific to CIFAR.

---

## Phase 8 — Research Hardening

For serious publication-quality evaluation:

```text
multiple datasets
multiple tasks
multiple seeds
strict compute budgets
strict inference-cost accounting
ablations
controller calibration
search trajectory analysis
failure analysis
```

---

# 11. Metrics

Never let Jev replace the true task objective.

The training/evaluation metric remains ground truth.

Possible system-level metrics:

```text
best objective
objective vs experiments
objective vs GPU-hours
objective vs wall-clock
objective vs inference cost

experiments/hour
number of Jev calls
number of LLM calls

failed/OOM experiment rate
redundant experiment rate
search-space coverage
```

For hybrid systems additionally:

```text
escalation frequency
escalation precision
value gained after escalation
controller confidence
controller calibration
```

---

# 12. Scalability Requirements

JevResearch should scale along three independent axes.

## Controller scalability

Swap:

```text
Random
→ Jev
→ TPE
→ CMA-ES
→ Hybrid
```

without touching tasks.

---

## Domain scalability

Swap:

```text
CIFAR
→ language modeling
→ tabular
→ segmentation
```

without touching controllers.

---

## Search-space scalability

Start with:

```text
predefined operators
```

then move toward:

```text
parameterized operators
↓
dynamically generated operators
↓
LLM-generated operators
↓
open-ended code edits
```

The architecture should support this progression naturally.

---

# 13. Important Design Rules

### Rule 1

**Metrics are ground truth.**

Do not ask Jev whether an experiment is objectively good when the real metric already exists.

---

### Rule 2

**Jev decides; it does not need to generate code.**

Its core value is fast structured decision-making over experiment state.

---

### Rule 3

**Autoregressive LMs should be optional.**

The system must remain useful without them.

---

### Rule 4

**Do not entangle the controller with a modality.**

A controller should consume generic state.

---

### Rule 5

**Do not start with arbitrary code editing.**

Structured operators make early experiments:

- reproducible
- interpretable
- safe
- easy to benchmark

Open-ended code generation comes later.

---

### Rule 6

**Every decision must be logged.**

Store:

```text
controller input
candidate set
controller output
confidence/probabilities if available
selected candidate
ExperimentSpec (full config, seed, dataset split, code version, parent ID)
SearchState
experiment result
failures
training time, controller time, total timings/cost
incumbent changes
campaign progress
```

Persist everything needed to reconstruct and resume the campaign from the first working loop, including generator/controller random state where needed. Explicitly record when probabilities/confidence are unavailable.

This will matter enormously for analysis and publication.

---

# 14. Relationship to Autoresearch

Karpathy-style autoresearch broadly follows:

```text
LLM inspects state
→ proposes/edits code
→ experiment runs
→ metric returned
→ LLM reasons again
```

JevResearch asks whether the expensive generative reasoning step is necessary every time.

The alternative is:

```text
cheap decision model
→ routine search

expensive generative LM
→ only when genuinely needed
```

This creates a more hierarchical research agent:

```text
System 1:
fast search/control

System 2:
open-ended reasoning/invention
```

---

# 15. Research / Paper Direction

The project should initially be built as a strong open-source system.

However, every component should be implemented in a way that supports rigorous experimentation.

The scientific contribution should **not** be framed as:

> "We use Jev for AutoML."

That is too implementation-specific and too dependent on one model.

A stronger framing is:

> **Hierarchical System-1/System-2 control for autonomous ML experimentation.**

Possible research hypothesis:

> A fast decision-specialized controller can handle most experiment-selection steps, while expensive autoregressive reasoning is only needed when the search space itself must be expanded.

Potential comparison:

```text
Random
TPE
CMA-ES
Autoregressive LM every step
Jev-only
Jev + classical optimizer
Jev + adaptive LLM escalation
```

Important measurements:

```text
final performance
performance vs compute
performance vs inference cost
number of autoregressive LM calls
search efficiency
failure rate
escalation usefulness
controller calibration
```

The most interesting version is not merely cheaper AutoML.

It is the broader question:

> **When does autonomous research actually require System-2 reasoning?**

If JevResearch can reach similar or better performance while using dramatically fewer autoregressive LM calls—and this generalizes across vision, language modeling, and tabular tasks—that becomes a much stronger paper direction.

---

# 16. Immediate Development Scope

Do not build everything at once.

The current target should be only:

```text
Phase 1
+
Phase 2
```

Meaning:

```text
generic core abstractions
+
CIFAR-10 task with one fixed small ConvNet
+
learning rate, weight decay, optimizer operators only
+
RandomController
+
reliable experiment logging
```

First milestone:

> **CIFAR-10 + generic experiment loop + predefined operators + random selection.**

Once that works cleanly, expand outward rather than redesigning inward.
