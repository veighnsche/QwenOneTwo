# Redesign progress

This is a resumable index for the [implementation plan](learning-redesign-plan-2026-11-08.md)
and [queue protocol/prompts](redesign-queue-prompts-2026-11-08.md). Inspect actual
code, tests, artifacts, running jobs and pushed commits before trusting a row.
Preparing these prompts does not authorize implementation or training.

Delivery: `pending`, `in_progress`, `blocked`, `delivered`. Technical/learning
checks: `pending`, `pass`, `fail`, `unsupported`, or `not_applicable`. A delivered
experiment can fail its learning acceptance. No failed check becomes a pass
because a later queued task starts.

| Slice | Delivery | Required evidence / checks | Current evidence |
|---|---|---|---|
| RD-01 | pending | Coverage/panel/native-supervision manifests; model-free checks | None; plan only |
| RD-02 | pending | Runner preparation/rejection checks; identity/resume contracts | None; plan only |
| RD-03 | pending | Source/cache policy and static checks; explicit deferred GPU check IDs | None; plan only |
| RD-04 | pending | Versioned payload/admission contracts and implementation; deferred GPU IDs | None; plan only |
| RD-05 | pending | Joint objectives, owners, binding/mask tests; deferred GPU IDs | None; plan only |
| RD-06 | pending | Native wiring/eligibility report; deferred real-return/gradient IDs | None; plan only |
| RD-07 | pending | All required deferred/changed-path 3090 checks pass; actual counts/hashes | None; not executed |
| RD-08 | pending | Complete matched fit/evaluation receipts; useful-improvement outcome separate | None; training not authorized by prompt preparation |
| RD-09 | pending | Accepted RD-08 receiver plus eligible native run/evaluation receipts | None; training not authorized by prompt preparation |

Earliest unfinished slice: **RD-01**. No redesign implementation/model work has
started. No redesign job is recorded as running. Check current process evidence
when resuming rather than assuming this historical observation remains current.

For each attempted slice add a short entry with implementation/evidence paths,
exact commands/results, dependency hashes, relevant implementation commits and
verified push status. List deferred checks by stable IDs and their owning slice;
RD-07 resolves them. Record active run manifest/PID/log/checkpoint paths when a
long job starts, and update them when it finishes or is interrupted. Store raw
artifacts locally under `data/`; commit concise reports/portable evidence only.

Blockers: none encountered during prompt preparation. Known data/supervision and
layout limits are hypotheses/constraints in the plan, not completed checks.
