# Repository architecture

Read `docs/architecture.md`, especially sections 4 and 7, before changing the search loop or adding a component.

- Put implemented contracts and orchestration in `src/jevresearch/core/`, selection in `controllers/`, configuration moves in `operators/`, task-specific data and workers in `tasks/`, trial supervision in `execution/`, and durable records in `storage/`. Add future packages only when they have working code.
- Keep controllers and the core runner independent of ML modality. A task defines valid proposals and execution details; `CandidateGenerator` makes complete immutable specs; a controller selects from a persisted offer.
- Persist the selected pending trial before execution. The executor returns a structured result; storage records successes, failures, and interruptions, and the next `SearchState` is reconstructed from history.
- Generic execution code must obtain a process task's payload and command from that task. Do not import a specific task or worker into generic execution or storage.
- Keep the architecture document aligned with actual interfaces when those interfaces change. Preserve readable old campaign history and run the relevant regression tests before committing.
