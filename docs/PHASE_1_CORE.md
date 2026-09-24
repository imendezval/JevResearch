# JevResearch — Phase 1: durable generic search loop

## Mission and scope

Build a small Python package that runs a deterministic synthetic task through a task-independent experiment loop. A seeded random controller selects from valid, complete experiment proposals. The loop records what it offered, selected, attempted, and measured; it can resume safely after a pause or crash.

This phase does **not** train a neural network, download a dataset, or call Jev. The attached JevResearch architecture describes the long-term direction; this brief is the implementation contract for Phase 1. Inspect the repository and its instructions first. Adapt to an existing codebase instead of imposing a new scaffold. Do not add placeholder optimizer, LLM, or modality modules.

## Core contract

- `Task` defines a versioned protocol, baseline configuration, config validation, deterministic candidate generation, and execution/evaluation. `Controller` sees a compact serializable `SearchState` and an ordered set of `Candidate`s, then returns one candidate ID. `Runner` owns the loop and budget. Storage owns the durable history. The runner and controller must not import a concrete task.
- A candidate has a stable ID within its offer, operator name/parameters, parent experiment ID, full resulting configuration, and the complete trial spec it would run. Validate and deduplicate configurations **before** offering them. Persist the exact ordered offer; do not regenerate an outstanding offer on resume. A later optimizer may generate candidate values through the same task validation boundary.
- `ExperimentSpec` is immutable: task/protocol version, full JSON-serializable config, data/split identity, fixed evaluation budget, training seed, source identity, and optional parent. Fingerprint canonical serialization. Separately track a config/protocol key so a different seed cannot disguise a repeat config in one search. A future replicate study uses a separate session.
- `ExperimentResult` records status, primary scalar objective and secondary metrics, duration, error type/message, and artifact references. Objective direction is explicit; reject missing, nonfinite, failed, or protocol-incompatible metrics as best. Ties retain the earlier completed experiment, then trial number as a stable fallback.
- A session freezes task/protocol version, objective, controller type/seed, total trial budget, seed schedule, and executable source identity. Budget means **attempted trials including failures and the baseline**; require budget >= 1. The baseline is trial 0, uses a recorded deterministic initialization decision, and consumes one slot. If it fails, end the session without inventing a current best. Stop cleanly on exhausted budget or no novel candidates.
- Source identity is the Git commit plus a content digest of relevant package source and dependency manifests (including local edits); store runtime/package versions as metadata. Freeze this identity when a session starts and compare the content digest on resume. A new commit with identical executable contents need not invalidate resume; code or protocol changes require a new session. Old sessions must remain inspectable. Do not promise bitwise reproducibility across hardware.

Prefer standard-library `sqlite3`, dataclasses/protocols, and a small package under `src/jevresearch` if starting fresh. Keep the core free of PyTorch. Add only dependencies with a concrete need and record them reproducibly.

## Durable state machine

Use SQLite constraints and transactions around each change in logical state. One local writer and one in-flight trial per session suffice. Make these transitions precise:

1. Create the immutable session and reserve the baseline trial **before** running it. Record it as a deterministic baseline decision.
2. Before a controller call, save the ordered candidate offer and the input snapshot needed to replay it. An offer without a committed selection stays outstanding.
3. Validate the selected ID against that saved offer. In one transaction, save the selection, the exact spec, the next controller RNG state (or equivalent replay position), and one pending trial reservation. A duplicate trial/config key is a visible invariant error, never a new attempt.
4. Mark the trial running before executing; atomically record its terminal status and result. A task failure/OOM is a failed attempt, with a short sanitized diagnostic; storage corruption or invariant failure stops the runner visibly.
5. On resume, execute an already selected **pending** trial without selecting again. Mark a previously **running** trial with no result interrupted and count it once; never launch it again automatically. For an unselected offer, reuse its exact candidates and controller input. Do not count an offer or failed controller call as a trial. A controlled pause between trials preserves the session's total budget and is not an interruption.

Keep an audit trail of session, offer, decision, trial, status, timestamps, result, and error. A failed offer/decision record may be needed in Phase 3; make room for it without building a provider abstraction now. Never persist credentials or raw environment dumps.

## Synthetic task and interface

Create a small deterministic task with a known optimum, two meaningful operators, bounded search space, and an explicit baseline. It should permit injected task failure and a controlled interruption so the state machine can be tested. Derive controller randomness separately from trial seeds. Candidate IDs and order must be reproducible from the saved session state. At a given trial index, all offered candidates receive the same predetermined trial seed; selection does not change the seed schedule.

Provide a usable CLI with `run`, `resume`, `show`, and a controlled `--max-new-trials` (or equivalent pause mechanism). A session's original `--budget` is a total, not an extra allowance on resume. Document exact commands and show status, attempted count, best result, and why a session stopped. Keep the phase runnable on CPU without network access.

## Verification and acceptance gate

Write focused CPU tests for spec/config identity, candidate mapping and deduplication, objective direction/ties, failure accounting, session compatibility, candidate exhaustion, and transactional uniqueness. Use a second tiny test task to check that adding a task needs no runner/controller edits. Test these three recovery boundaries explicitly: outstanding offer, committed pending trial, and running interrupted trial. Compare a fresh seeded run with a gracefully paused/resumed run; decisions and results must match. An interrupted run must have one additional counted attempt and no duplicate fingerprint.

Run the package's test command and two CLI demonstrations: a complete synthetic search and a paused/resumed search to the original total budget. Inspect the DB via the CLI or a small query: baseline present, exactly one trial per decision, correct attempted count, no repeated config, correct best, and linked errors. If an actual crash cannot be safely induced in the CLI demonstration, the automated recovery tests must exercise it. Report commands and outputs; a skipped or failed gate is not a pass.

Phase 1 passes when a generic runner plus `RandomController` meets the above gate and an independent task can implement the interface without modifying either one. Stop there.

## Long-run and Git instructions

- Inspect `git status`, existing instructions, tests, and user edits before working. Write a brief internal plan and execute it without pausing for routine decisions. If the target is a genuinely new project directory without Git, initialize Git there after checking it is not nested in another repository. Put SQLite DBs, run artifacts, logs, datasets, checkpoints, virtual environments, local `.env` files, and secrets in `.gitignore` before the first commit.
- Make genuine, reviewable commits after coherent passing units, e.g. contracts/storage, runner/synthetic task, CLI/recovery tests/docs. Stage only files from this work, inspect `git diff --cached`, and record test commands in commit messages or the final report. Do not manufacture granular history by splitting one unreviewed broken state; do not amend/rebase/reset existing history, push, or commit unrelated user edits. If author identity is unavailable, leave the work organized and report it rather than inventing one.
- Commit implementation changes **before** starting the final recorded smoke session. Do not change executable code or dependencies while a session runs. If a source change becomes necessary, end that session, commit the fix, and start a new one. A later doc-only commit may change the Git SHA; the frozen source digest still governs resume.
- Review the final diff for task-specific branches in the core, duplicate execution paths, lost decision records, false reproducibility claims, and accidental data/secret commits. Report completed commits, exact verification, any blocked gate, and the Phase 2 handoff. Do not start Phase 2.
