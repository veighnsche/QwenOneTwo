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
| RD-01 | in_progress | Coverage/panel/native-supervision manifests; model-free checks | [Report](redesign-rd01-results-2026-11-08.md), [manifest](evidence/redesign-coverage-v1.json); technical checks pass, push pending |
| RD-02 | pending | Runner preparation/rejection checks; identity/resume contracts | None; plan only |
| RD-03 | pending | Source/cache policy and static checks; explicit deferred GPU check IDs | None; plan only |
| RD-04 | pending | Versioned payload/admission contracts and implementation; deferred GPU IDs | None; plan only |
| RD-05 | pending | Joint objectives, owners, binding/mask tests; deferred GPU IDs | None; plan only |
| RD-06 | pending | Native wiring/eligibility report; deferred real-return/gradient IDs | None; plan only |
| RD-07 | pending | All required deferred/changed-path 3090 checks pass; actual counts/hashes | None; not executed |
| RD-08 | pending | Complete matched fit/evaluation receipts; useful-improvement outcome separate | None; training not authorized by prompt preparation |
| RD-09 | pending | Accepted RD-08 receiver plus eligible native run/evaluation receipts | None; training not authorized by prompt preparation |

Earliest unfinished slice: **RD-01**, pending commit/push verification only.
RD-02..09 are unstarted; no redesign model job was launched.

For each attempted slice add a short entry with implementation/evidence paths,
exact commands/results, dependency hashes, relevant implementation commits and
verified push status. List deferred checks by stable IDs and their owning slice;
RD-07 resolves them. Record active run manifest/PID/log/checkpoint paths when a
long job starts, and update them when it finishes or is interrupted. Store raw
artifacts locally under `data/`; commit concise reports/portable evidence only.

Blockers: none encountered during prompt preparation. Known data/supervision and
layout limits are hypotheses/constraints in the plan, not completed checks.

## RD-01 — 2026-11-08

Technical acceptance: **pass**. Learning outcome: **not_applicable**, no model or
training execution. RD-01 deferred GPU check IDs: **none**. Implementation:
`qwen_one_two/training/redesign_coverage.py`, shared audit recipe/entrypoint in
`architecture_audit.py`, `learning_report.py:summarize_coverage_learning`,
`scripts/audit_redesign_coverage.py`, `tests/test_redesign_coverage.py`.

Evidence: [report](redesign-rd01-results-2026-11-08.md),
[portable manifest](evidence/redesign-coverage-v1.json), raw inventories/panels/logs
under `data/redesign-rd01-20261108/accepted/`. Source, release, raw artifact and
reused historical report hashes are pinned and verified. 9,000 selected rows,
all 1,500 receiver training rows, E130/X180, zero whole-release selected-family
overlap. Two repeated full constructions are byte-identical with model imports
blocked. Historical results were reused without model execution.

Exact commands/results:

- `python -m scripts.audit_redesign_coverage --output data/redesign-rd01-20261108/accepted`: exit 0.
- `python -m unittest tests.test_redesign_coverage tests.test_architecture_audit tests.test_latent_semantic_audit tests.test_learning_report`: **23 pass**, 60.371 s, exit 0.
- Compilation of all six changed/new Python files: pass; exact command in portable evidence.
- `git diff --check`: pass. Dependency/artifact and historical release identity verification: pass.

Known downstream limits: no assessed training examples for negation/OR,
remaining/half-completion or denominator 200; no assessed familiar-range Score
E/X cohort. 224 interleaved calls lack their own Reason, 373 interleaved records
have ambiguous source occurrences; 44 writer/139 interleaved Choice calls do
not meet the current unique-selection continuation guard. Native targets remain
conditional on exact actual bindings/returns. These complete audit findings do
not block RD-01 delivery or authorize invented labels/guard changes.

Recovery: clean `main` at `70dc5b2` had no RD-01 implementation or matching run
manifest. Sandbox process inspection had limited host visibility; no service was
changed. Initial model-free implementation/test failures are retained in the
raw directory and superseded by the accepted run. No agents, locked data, new
records/reviews, GPU/model work, training or deployment. Handoff updated.
Implementation commit/push verification: pending finalization.
