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
| RD-01 | delivered | Coverage/panel/native-supervision manifests; model-free checks | [Report](redesign-rd01-results-2026-11-08.md), [manifest](evidence/redesign-coverage-v1.json); 23 tests pass; implementation `800c100` pushed and verified |
| RD-02 | delivered | Runner preparation/rejection checks; identity/resume contracts | [Report](redesign-rd02-results-2026-11-08.md), [evidence](evidence/receiver-preparation-v1.json); 12 final preparation tests +28 reused tests pass; `254f526` pushed and verified |
| RD-03 | in_progress | Source/cache policy and static checks; explicit deferred GPU check IDs | [Report](redesign-rd03-results-2026-11-08.md), [evidence](evidence/frozen-w-source-v1.json); 50 static tests pass; three GPU checks pending RD-07; push pending |
| RD-04 | pending | Versioned payload/admission contracts and implementation; deferred GPU IDs | None; plan only |
| RD-05 | pending | Joint objectives, owners, binding/mask tests; deferred GPU IDs | None; plan only |
| RD-06 | pending | Native wiring/eligibility report; deferred real-return/gradient IDs | None; plan only |
| RD-07 | pending | All required deferred/changed-path 3090 checks pass; actual counts/hashes | RD03-GPU-D-INDEPENDENCE, RD03-GPU-CACHE-PARITY, RD03-GPU-CONSUMER-GRADIENTS pending; no GPU execution |
| RD-08 | pending | Complete matched fit/evaluation receipts; useful-improvement outcome separate | None; training not authorized by prompt preparation |
| RD-09 | pending | Accepted RD-08 receiver plus eligible native run/evaluation receipts | None; training not authorized by prompt preparation |

Earliest unfinished slice: **RD-03**, awaiting commit/push verification only.
RD-04..09 are unstarted; no redesign model job was launched.

For each attempted slice add a short entry with implementation/evidence paths,
exact commands/results, dependency hashes, relevant implementation commits and
verified push status. List deferred checks by stable IDs and their owning slice;
RD-07 resolves them. Record active run manifest/PID/log/checkpoint paths when a
long job starts, and update them when it finishes or is interrupted. Store raw
artifacts locally under `data/`; commit concise reports/portable evidence only.

Current delivery blockers: **none**. RD-01 data/supervision limits below are
measured metadata findings; layout/technical checks for later slices remain pending.

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
Implementation commit: `800c100a56dfcc2afe33bdc004e965affef45331`.
`git -c 'credential.helper=!gh auth git-credential' push origin main`: exit 0,
`70dc5b2..800c100 main -> main`. A subsequent authenticated `git ls-remote`
returned `800c100a56dfcc2afe33bdc004e965affef45331 refs/heads/main`, identical to
local HEAD. This delivery/handoff finalization is committed and pushed separately
on top of that verified implementation; no RD-02 work is included. 141 local
Markdown targets resolve across the changed ledger/handoff/report. All required
RD-01 checks pass; learned capability remains unchanged.

## RD-02 — 2026-11-08

Technical acceptance: **pass** for this model-free slice. Learning outcome:
**not_applicable**. No required RD-02 check is deferred; new-profile tensor and
resume qualification remains future RD-07 work as those profiles are implemented.

Predecessor recovery: RD-01 source/test/raw/reused-audit hashes are unchanged;
its recorded 23 passing tests are reused. Both delivered commits are ancestors
of clean baseline `47c58a3`, verified as remote `main`. No matching RD-02 job or
run manifest existed; sandbox process inspection has limited host visibility.

Implementation: `scripts/train_receiver_probe.py` gains explicit prepare/execution
and named profiles; `training/receiver_plan.py` binds plans, caches and resumable
identities. `receiver_profiles.py` shares profile/owner declarations;
`checkpoint_metadata.py` exposes model-free inspection used by the existing CUDA
checkpoint loader; `loss_config.py` shares unchanged coefficients. The actual
`Window`, Adam/update and tensor checkpoint save/restore bodies remain unchanged.
Bound legacy resume uses the existing loop and restores complete-window state.

Evidence: [report](redesign-rd02-results-2026-11-08.md),
[portable manifest](evidence/receiver-preparation-v1.json), and raw plans/receipts/
logs/blocked-import launcher under `data/redesign-rd02-20261108/verified/`.
All three actual runner preparations pass with model imports blocked. Repeated
legacy plans are byte-identical. Proposed frozen-pooled/direct arms bind the
same 1,500 records, four passes /6,000 presentations /750 windows, initial masters/
moments/RNG, owners, objectives and dependencies, with distinct profile identities.
All preparation execution counters are zero. These proposed fits are not launched
or authorized by preparation; new profiles reject execution pending RD-03..07.

Commands/results:

- Four actual blocked-import runner preparations: exit 0. Exact commands, snapshot,
  environment, output paths, source/artifact hashes and host intervals are in evidence.
- `PYTHONPATH=tests:. python -m unittest tests.test_receiver_plan tests.test_receiver_learning tests.test_training_schedule tests.test_training_driver tests.test_host_metrics.HostMetricsTests`: **40 pass**, 62.390 s.
- After final dependency/admission additions, `PYTHONPATH=tests:. python -m unittest tests.test_receiver_plan`: **12 pass**, 47.402 s; unchanged results for the other 28 tests reused.
- Compilation of all nine changed/new Python files and `git diff --check`: pass.
  Existing coefficient values, unchanged Window/tensor checkpoint ASTs, plan source
  hashes and all reused RD-01 hashes verified. Exact compilation command in evidence.

A broader initial invocation also found the **pre-existing**
`HostMetrics.ProfilingPlanTests` setup error `profiling receipt input drift`:
130/130 rows match original records and historical input hashes, 0/130 match
current v2 view hashes. Relevant code/tests are unchanged from baseline. Its raw
failure/proof is retained; required Recorder and RD-02 checks pass independently.
No stale receipt was relabeled valid. This older diagnostic fixture remains a
known limitation, not an RD-02 implementation blocker.

No model/GPU execution, fitting, optimizer update, agents, new annotations/reviews,
locked access, deployment, service or Operating Table changes. RD-01 data/native
supervision limits remain as recorded. Handoff updated. Implementation commit:
`254f5268eacedd920b1c4219b79bcda8744a4524`. Authenticated
`git -c 'credential.helper=!gh auth git-credential' push origin main`: exit 0,
`47c58a3..254f526 main -> main`. Subsequent authenticated `git ls-remote`
returned `254f5268eacedd920b1c4219b79bcda8744a4524 refs/heads/main`, identical to
local HEAD. This ledger/handoff delivery finalization is committed and pushed
separately on top of that verified implementation. 148 local Markdown targets
resolve across the changed ledger/handoff/report. Stop at RD-02; next is RD-03.

## RD-03 — 2026-11-08

Staged implementation acceptance: **pass**. GPU qualification: **pending RD-07**.
Learning outcome: **not_applicable**; no model execution or fit.

Predecessors: RD-01/02 implementation/test/raw/reused-evidence hashes were
unchanged at clean baseline `850a2da`; delivered commits are ancestors of main
and remote main was verified equal. Valid earlier checks were reused. No matching
RD-03 code/run manifest existed; sandbox process inspection has limited host
visibility. No existing service was modified.

Implementation: `training/frozen_source.py`, explicit W mode in
`g1_backbone.py`/`codec/backbone.py`, source-policy/cache plumbing in `clean.py`
and `receiver_learning.py`, pending-check registration in `receiver_plan.py`,
`tests/test_frozen_source.py` and implemented/unrun `test_frozen_source_gpu.py`.
The [clean construction owner](clean-test-construction-v1.md) versions the opt-in
policy. Default D source and pooled roles remain controls; native source states
retain their actual prefill graph. W source encoding never changes ownership or
merges/replaces/zeros adapter parameters. Mutable D sources cannot be cached.

Evidence: [report](redesign-rd03-results-2026-11-08.md),
[portable manifest](evidence/frozen-w-source-v1.json), raw preparation/validation
logs and source migration under `data/redesign-rd03-20261108/accepted/`.
All 1,810 receiver training/E/X permitted view hashes are rechecked against the
unchanged RD-01 inventory during preparation. Old/new source hashes and unchanged
CleanView/tokenizer/native/dense computation bodies explain predecessor reuse;
old plans/checkpoints are not relabeled compatible with new code. Raw previous
evidence is retained unchanged.

Commands/results:

- `PYTHONPATH=tests:. python -m unittest tests.test_frozen_source tests.test_training_inputs tests.test_receiver_plan tests.test_receiver_learning tests.test_latent_semantic_audit tests.test_learning_report`: **50 pass**, 73.251 s.
- Actual frozen-pooled, repeated frozen-pooled and legacy CLI preparations with
  model imports blocked: exit 0; exact commands and fingerprints in evidence.
  Repeated frozen plans are byte-identical; all model/execution counters zero.
- Compilation of all eight changed/new Python files and `git diff --check`: pass;
  exact compilation command in evidence. GPU file was compiled/inspected only,
  never executed or reported as a passing skipped suite.

Deferred checks (all **pending**, owner **RD-07**, exact 3090 UUID only):

- `RD03-GPU-D-INDEPENDENCE` → `FrozenSourceGPU.test_frozen_features_ignore_D_updates`.
- `RD03-GPU-CACHE-PARITY` → `FrozenSourceGPU.test_cache_matches_uncached_and_preserves_bindings`.
- `RD03-GPU-CONSUMER-GRADIENTS` → `FrozenSourceGPU.test_consumer_gradients_and_frozen_W`.

These methods use an existing reviewed assessed Noul record and actual parent/
reference artifacts. They are targeted machinery assertions, not full semantic
qualification. The exact future command/opt-in is recorded as **not executed**.
No source tensors/fingerprints, parity, gradients, timing improvement or learning
outcome were fabricated. New profile execution remains gated pending later slices.
RD-01 data/native supervision limits and the prior historical profiling-test drift
remain unchanged. Handoff updated. Implementation commit/push: pending finalization.
Stop after RD-03; next implementation slice is RD-04, retaining these GPU checks
as pending until RD-07.
