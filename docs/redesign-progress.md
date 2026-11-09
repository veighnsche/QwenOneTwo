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
| RD-03 | delivered | Source/cache policy and static checks; explicit deferred GPU check IDs | [Report](redesign-rd03-results-2026-11-08.md), [evidence](evidence/frozen-w-source-v1.json); 50 static tests pass; three GPU checks resolved by RD-07; `2cbfa41` pushed and verified |
| RD-04 | delivered | Versioned payload/admission contracts and implementation; deferred GPU IDs | Contract `2247619`, implementation `ae800a4` pushed/verified; [report](redesign-rd04-results-2026-11-08.md), [evidence](evidence/direct-role-states-v1.json); 61 static tests pass +6 refreshed; three new GPU checks resolved by RD-07; zero receiver support |
| RD-05 | delivered | Joint objectives, owners, binding/mask tests; deferred GPU IDs | `fe90af0` pushed/verified; [report](redesign-rd05-results-2026-11-08.md), [evidence](evidence/joint-receiver-objectives-v1.json); 74 static tests pass +14 refreshed; two new GPU checks resolved by RD-07; Reason masks preserved |
| RD-06 | delivered | Native wiring/eligibility report; deferred real-return/gradient IDs | `6f74680` pushed/verified; [report](redesign-rd06-results-2026-11-08.md), [evidence](evidence/direct-native-flow-v1.json); 86 static tests pass +10 refreshed; two new GPU checks resolved by RD-07; zero complete corpus formation/semantic continuation/direct reader support |
| RD-07 | delivered | All required deferred/changed-path 3090 checks pass; actual counts/hashes | [Report](redesign-rd07-results-2026-11-08.md), [evidence](evidence/redesign-qualification-v1.json); ten deferred +three new GPU checks pass; twelve1024 traces; zero updates; `7da99c6` pushed and verified |
| RD-08 | in_progress | Complete matched fit/evaluation receipts; useful-improvement separate | [v2 report](receiver-matched-v2-results-2026-11-08.md), [pause evidence](evidence/receiver-matched-v2-pause-2026-11-09.json); full1500 support,18 GPU IDs resolved,104 distinct static test results; control paused at599 updates,500 on disk; candidate/evaluation unstarted; user GPU hold |
| RD-09 | blocked | Accepted RD-08 receiver plus eligible native run/evaluation receipts | [Report](redesign-rd09-results-2026-11-08.md), [evidence](evidence/native-demonstration-rd09-v1.json); conditional authorization recorded; blocked on RD-08/no accepted receiver and zero native supervision; execution unstarted, zero updates; blocker `a864b65` pushed/verified |

RD-07 is delivered; technical acceptance passes and implementation `7da99c6` is pushed/verified.
All ten deferred GPU checks are resolved; no required RD-07 check remains pending.
Earliest unfinished slice: **RD-08**, paused following the user's GPU-off constraint on2026-11-09. Format recovery provides1500/1500 shared support and changed-path qualification passes. Control has599 completed updates,500 saved on disk; candidate/evaluation/RD09 remain unstarted. No automatic GPU resume.

For each attempted slice add a short entry with implementation/evidence paths,
exact commands/results, dependency hashes, relevant implementation commits and
verified push status. List deferred checks by stable IDs and their owning slice;
RD-07 resolves them. Record active run manifest/PID/log/checkpoint paths when a
long job starts, and update them when it finishes or is interrupted. Store raw
artifacts locally under `data/`; commit concise reports/portable evidence only.

Current RD-01..07 technical implementation blockers: **none**. The historical
direct-v1 zero-support blocker is resolved by the explicitly authorized v2 format
revision and full audit. Current execution holds for the user's GPU-off constraint.
RD-01 data/native supervision limits remain; qualification does not waive them.

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
remain unchanged. Handoff updated. Implementation commit:
`2cbfa41518a0433ea868e2f0ca2c0bad08568008`. Authenticated
`git -c 'credential.helper=!gh auth git-credential' push origin main`: exit 0,
`850a2da..2cbfa41 main -> main`. Subsequent authenticated `git ls-remote`
returned `2cbfa41518a0433ea868e2f0ca2c0bad08568008 refs/heads/main`, identical to
local HEAD. This ledger/handoff finalization is committed and pushed separately
on top of that verified implementation. 165 local Markdown targets resolve.
Implementation is delivered under staged protocol; all three GPU checks remain
pending and no GPU qualification or learned result is claimed.
Stop after RD-03; next implementation slice is RD-04, retaining these GPU checks
as pending until RD-07.

## RD-04 recovery — 2026-11-08

Clean main `483c256`; RD-01/02 verified through RD-03 source migration, RD-03
source/test/raw hashes unchanged. Valid recorded model-free results reused.
Three RD-03 GPU checks remain pending RD-07. No matching RD-04 implementation
or run manifest; sandbox process visibility is limited. No model execution.

RD-04 contract phase: `codec/layout.py`, `docs/direct-role-states-v1.md` and
seven named owner specifications define version 2 metadata, 32-state fields,
variable exact-attribute keys and complete 256/64 admission/return costs.
`python -m unittest tests.test_role_layout`: four model-free tests pass.
Contract commit/push precedes representation implementation as required.

Contract commit: `22476193bb517b503e927cd141feddf56e02af7b`, pushed and remote
main verified before implementation edits.

## RD-04 implementation — 2026-11-08

Staged implementation acceptance: **pass**, delivered. GPU qualification: **pending RD-07**.
Learning outcome: **not_applicable**; no model execution or training.

Implementation/evidence: [report](redesign-rd04-results-2026-11-08.md),
[portable evidence](evidence/direct-role-states-v1.json),
[coverage manifest](evidence/direct-role-layout-coverage-v1.json), raw
`data/redesign-rd04-20261108/`. Existing registry/runtime/renderer support ordered
variable role states, immutable profile/length metadata, absent-midpoint omission,
exact attributes, shared identity-initialized FP32 residual projection and explicit
writer length/capacity owners. Full all-status echo/largest-candidate reservations
retain 256/64 envelopes. Preflight rejects before source encoding/component
prefill, and runtime sealing rejects atomically before dispatch.

Predecessor evidence/raw artifacts remain unchanged, and delivered commits are
ancestors. Current source migration is explicit in the evidence. Native computation
methods, producer/binding, CleanView/token mapping, checkpoint and numeric kernels
remain unchanged; Session.open adds explicit profile selection before prefill.
Current preparation revalidates all 1,810 permitted view identities. Earlier
prepared plans are stale and cannot be reused as current.

Commands/results:

- `PYTHONPATH=tests:. python -m unittest tests.test_role_layout tests.test_codec_exact tests.test_direct_role_packets tests.test_direct_role_encoding tests.test_frozen_source tests.test_training_inputs tests.test_receiver_plan tests.test_receiver_learning`: **61 pass**, 83.413 s.
- Final exact selected-candidate attribute return assertions:
  `PYTHONPATH=tests:. python -m unittest tests.test_direct_role_packets`: **6 pass**,
  0.069 s; unchanged results for the other tests reused.
- Four actual CLI preparations (direct/repeat/frozen-pooled/legacy), with model
  imports blocked: exit 0. Direct repeated plans are byte-identical; all execution
  counters zero. Exact argv/source/raw hashes/receipts are in portable evidence.
- Metadata-only Rust BPE/Jinja full-panel audit: 1,810 rows, zero supported.
  All 105 saved source token lengths from 32 historical probe rows match pinned
  record/input hashes. No neural tokenizer/model forward or feature extraction.
- Compilation including GPU test file and `git diff --check`: pass. GPU methods
  inspected/compiled/collection-imported only, never executed or counted passing.

Every excluded row, family, operation, field length, mapping identity and observed
whole-packet cost is retained. All 1,810 rows exceed the worst-case return bound;
1,354 rows also have overlength questions, 93 have overlength alternatives, and
three E rows exceed the request bound. No rows were cropped or removed and no
bound increased. Preparation binds this explicit future-fit blocker while
preserving the complete proposed exposure schedule. No learning claim follows.

New deferred checks, all **pending**, owner **RD-07**:

- `RD04-GPU-IDENTITY-PROJECTION` → `DirectRolesGPU.test_identity_projection_and_legacy_wire`.
- `RD04-GPU-DIRECT-ROUNDTRIP` → `DirectRolesGPU.test_direct_packets_returns_and_immutable_bindings`.
- `RD04-GPU-DIRECT-GRADIENTS` → `DirectRolesGPU.test_live_projection_and_consumer_gradients`.

These declared engineering latent fixtures check mechanisms only; they are not
admissible corpus features or semantic qualification. Exact 3090 UUID/opt-in is
required before model imports. All three earlier RD-03 checks remain pending.
No GPU/CPU model workload, optimizer update, agents, new data/reviews, locked
access, deployment, service or Operating Table change. Earlier RD-01 label gaps
and historical profiling-test drift remain. No RD-05 implementation is included.

Implementation commit: `ae800a4bf675fb58e6419ca411054fa9cabb973a`.
`git -c 'credential.helper=!gh auth git-credential' push origin main`: exit 0,
`2247619..ae800a4 main -> main`. Authenticated `git ls-remote` returned
`ae800a4bf675fb58e6419ca411054fa9cabb973a refs/heads/main`, identical to local HEAD.
261 local Markdown targets resolve. This ledger/handoff delivery finalization is
committed and pushed separately on top of the verified implementation. Six GPU
checks remain pending; zero-support coverage blocks a future fit, with no waiver.
Stop after RD-04. Next unfinished implementation slice is RD-05.

## RD-05 recovery — 2026-11-08

Clean main `7576540` matches authenticated remote main. RD-01..04 source/test/raw
evidence and migration chains verified; delivered commits are ancestors. Six
RD-03/04 GPU checks remain pending RD-07. Valid unchanged results reused. No
matching RD-05 manifest or job was found; process inspection is limited to the
sandbox namespace. No service changed. The zero-admissible-row direct-layout
coverage finding still blocks a future fit; it is not an unfinished RD-04 static
implementation check. RD-05 makes no layout or allowance revision.

## RD-05 implementation — 2026-11-08

Staged implementation acceptance: **pass**, delivered.
GPU qualification: **pending RD-07**. Learning outcome:
**not_applicable**; no model execution, feature extraction, fitting or update.

Evidence: [report](redesign-rd05-results-2026-11-08.md),
[portable manifest](evidence/joint-receiver-objectives-v1.json),
[contract](prepared-receiver-component-v1.md), raw `data/redesign-rd05-20261108/`.
`component_admission.py` binds reviewed clean inputs to the actual Runtime/
snapshot, exact ordered roles/candidates, source states/coordinates and complete
spans. Labels attach only after the complete forward and revalidation; record IDs
or predicted answers do not admit arbitrary writer vectors. Receipts explicitly
claim no reference qualification or accepted receiver behavior. Diagnostic-reference
rejection and unsupported native masks remain. Failed empty attempts still occupy
scheduled exposures with no loss credit; unadmitted gradients reject.

Existing joint status/Noul/Choice/Score/Boolean feasibility losses and coefficients
are reused. Eligibility was factored unchanged into a model-free module; existing
actual-status and old supported Reason/isolated-reader exceptions remain. New-profile
Reason targets are unsupported/masked, never encoded or zero-filled. All 832
reviewed Reason annotations are retained. Metadata potentials after exact admission:
status1500 /Noul256 /Choice158 /Score254 /feasibility497; actual eligible exposures
are zero because no model was run. All 13 operation/reference-status cells remain
visible. Direct layout admission remains 0/1500 train, 0/130 E, 0/180 X; a future
fit still requires explicit layout revision/re-audit, without hidden subsets or
raised limits.

Exact projection owners are shared by preparation and actual construction. W, R,
startup row and reference remain frozen; writer length head is frozen for receiver
work. `extend_profile_window` retains parent masters/moments/RNG/counters and adds
fresh zero-delta projection owners without an update. Existing Window/checkpoint
math is unchanged. Joint reports separate heads/delivered success, raw/status
cells, eligible counts, Reason masks and paired startup retention. Frozen private
rows do not protect shared D features. Existing VJP diagnostics report each named
owner directly from saved buffers, with no extra forward/backward/update.

Commands/results:

- `PYTHONPATH=tests:. python -m unittest tests.test_component_admission tests.test_joint_receiver tests.test_learning_report tests.test_training_inputs tests.test_receiver_learning tests.test_receiver_plan tests.test_training_schedule tests.test_episode_producer tests.test_frozen_source`: **74 pass**, 120.531 s.
- After fixing failed-empty exposure retention,
  `PYTHONPATH=tests:. python -m unittest tests.test_component_admission tests.test_joint_receiver`:
  **14 pass**, 22.151 s, including one added test; other unchanged results reused.
  Together 75 distinct model-free tests are covered.
- Final GPU method collection/registration guard test: **1 pass**, 11.813 s.
  GPU methods were inspected/compiled/collection-imported only, never executed
  or counted as passing skipped suites.
- Four actual final `verified-final/` CLI preparations, with model imports blocked:
  direct/repeat/frozen-pooled/legacy all exit 0. Direct repeated plans are
  byte-identical; current source hashes match; all execution counters zero.
  Earlier preparation artifacts remain retained as superseded after source edits.
- Compilation and `git diff --check`: pass; exact commands/hashes in evidence.
  Predecessor migrations and 34 unchanged boundary/algorithm checks pass.

New deferred IDs, all **pending**, owner **RD-07**, exact 3090 UUID only:

- `RD05-GPU-GRADIENT-ACCUMULATION` → `JointReceiverGPU.test_joint_VJPs_and_eligible_accumulation`.
- `RD05-GPU-RESUME` → `JointReceiverGPU.test_complete_window_resume_and_frozen_owners`.

The first uses an existing reviewed frozen-W/pooled component and real joint VJPs;
the second checks parent/direct owners, real moments/RNG and engineering checkpoint
restoration. Neither steps an optimizer or claims corpus/direct semantic support.
All six earlier GPU IDs remain pending. No actual joint delivered behavior,
startup retention or owner gradient was measured in RD-05. RD-01 label/native
gaps and historical profiling receipt drift remain unchanged. No agents, new
records/reviews, locked access, deployment, service or Operating Table changes.
Stop after RD-05; no RD-06 implementation is included.

Implementation commit: `fe90af01540fab0962167f2b4375d07cb169f84b`.
`git -c 'credential.helper=!gh auth git-credential' push origin main`: exit 0,
`7576540..fe90af0 main -> main`. Authenticated `git ls-remote` returned
`fe90af01540fab0962167f2b4375d07cb169f84b refs/heads/main`, identical to local HEAD.
217 local Markdown targets resolve. A staged new-file EOF whitespace warning was
fixed with identical AST before pushing; four fresh `verified-final/` preparation
receipts bind the corrected bytes. Prior preparations remain unchanged and
explicitly superseded. This ledger/handoff delivery finalization is committed and
pushed separately on top of the verified implementation. Eight GPU checks remain
pending; the zero-support direct layout still blocks a future fit, without waiver.
Stop after RD-05. Next unfinished implementation slice is RD-06.

## RD-06 recovery — 2026-11-08

Clean main `0cd19cd` matches authenticated remote main. RD-01..05 implementation,
source/test/contract/raw evidence, recorded migrations and pushed ancestors verified.
Valid unchanged results reused. Eight GPU checks remain pending RD-07. No matching
RD-06 job/manifest existed; process inspection is limited to the sandbox namespace.
No model workload or service change. The zero-support direct receiver layout remains
a future-fit blocker, not an unfinished predecessor static implementation check.

## RD-06 implementation — 2026-11-08

Staged implementation acceptance: **pass**, delivered.
GPU qualification: **pending RD-07**. Learning outcome:
**not_applicable**; no model workload, source extraction, actual Session or update.

Evidence: [report](redesign-rd06-results-2026-11-08.md),
[portable manifest](evidence/direct-native-flow-v1.json),
[contract](direct-native-flow-v1.md), raw `data/redesign-rd06-20261108/accepted/`.
The opt-in EpisodeProducer uses actual original-prefill states, predicted ordered
writer fields/lengths, one live cache, actual Qev results and bound latent returns.
Optional direct annotation targets preflight/encode before Session creation; no
offline re-encoding or gold return enters the live direct call. Length supervision
is discrete; continuous fields remain connected. Explicit NativeEpisode admission
preserves decision/Reason/post-call masks and rejects dev accumulation. Forcing,
returned values/Refs and free/teacher-forced native tokens are recorded separately.

Full metadata inventory: 5,590 records, 3,220 writer/interleaved calls. All calls
exceed the unchanged worst-case return64 bound, so complete direct formation
eligibility is **zero**. Questions exceed32 in 569 writer/2023 interleaved/33 W
calls; complete requests exceed256 in 157/989/13 calls. 67 interleaved records
exceed full-source spans. All affected rows/calls/fields/costs/families remain.
Unverified native semantics imply zero decision/post-call continuation targets.
No new explanations/corrections, allowance increases, cropping or subset were used.

Direct supplied-reader support: **zero** across 750 training /120 development
R/X9 records. Legacy metadata potentials (not executed/qualified): Noul257,
Choice199, Score250 in training; 51/36/31 in development. Separate approved no-call
direct500/language-replay1000 training potentials do not establish return use.
Original 224 missing interleaved call Reasons, 746 ambiguous-source calls in373
records and nonunique Choice writer44/interleaved139/W4 remain. Actual eligible
gradient exposures are zero because no model was run. The earlier zero receiver
layout support still blocks its fit; native semantic/formation/reader gaps are
additional later-fit limits, not missing audit work.

Evaluation plan: 770 held-out cases, separated into native W/F470, prepared-X180
and supplied-reader120; 100 X pairs remain 90 prepared +10 reader, with zero
actual-native-final pair labels. Explicit W/F free evaluation has no target,
effort, control or override and cannot train. Final typed values/identities use
original rules/endpoints; heads/consumed returns cannot substitute. Existing
prose continuations provide no verified typed final-answer extractor; no parser
or judge was invented. Missing observations/extraction provenance stay unsupported.

Commands/results:

- `PYTHONPATH=tests:. python -m unittest tests.test_native_redesign tests.test_episode_producer tests.test_component_admission tests.test_joint_receiver tests.test_direct_role_encoding tests.test_direct_role_packets tests.test_frozen_source tests.test_training_inputs tests.test_receiver_plan tests.test_learning_report`: **86 pass**, 116.312 s.
- `PYTHONPATH=tests:. python -m unittest tests.test_native_redesign`: **10 refreshed
  pass**, 14.224 s, after final native telemetry/typed-field changes; unchanged
  other results reused. GPU collection/registration guard: **1 pass**, 13.047 s.
- Actual `scripts.audit_native_redesign` full metadata audit with model imports
  blocked: exit0; 5,590 records retained. Rust BPE/Jinja only, no neural features.
  Reader-op counts derived from unchanged audited rows without rerunning BPE.
- Four actual `accepted/verified/` final CLI preparations with model imports
  blocked: direct/repeat/frozen-pooled/legacy pass; repeated direct plans identical,
  current source hashes match, all execution counters zero. Earlier preparations
  are retained as superseded after typed observation rejection tightened.
- Compilation, staged whitespace, source/artifact identities and 33 unchanged
  boundary/algorithm checks pass. Audit-time producer/binding source bytes were
  recovered/hash-verified; native_plan/annotation ASTs remain unchanged by later
  telemetry/early-unsupported-return changes. Old evidence is not relabeled current.

New deferred IDs, all **pending**, owner **RD-07**:

- `RD06-GPU-NATIVE-RETURN` → `NativeRedesignGPU.test_actual_native_return_and_prefill_source`.
- `RD06-GPU-CONNECTED-GRADIENT` → `NativeRedesignGPU.test_native_loss_through_actual_return_receiver_and_writer`.

The first prepares Noul/Choice/Score actual-return/source/accounting fixtures;
the second isolates the Noul return-specific VJP through frozen receiver/codec
weights to continuous writer states and R. Guided controls, Assessed status,
existing continuation token and budget512 forcing are disclosed. No optimizer
is used, and these are not valid corpus semantic continuations. Methods were
compiled/inspected/collection-imported only, never executed or counted passing.
All eight earlier IDs remain pending, exact 3090 UUID only; no CPU/3060 fallback.

No model execution, feature extraction, training, agents, new data/reviews, locked
access, deployment, service or Operating Table change. Historical profiling input
drift and earlier data/native supervision gaps remain. Stop after RD-06; no RD-07
GPU execution is included. Next unfinished slice is RD-07 technical qualification.

Implementation commit: `6f74680c4c6a866999d271d15e681d239d9ff588`.
`git -c 'credential.helper=!gh auth git-credential' push origin main`: exit0,
`0cd19cd..6f74680 main -> main`. Authenticated `git ls-remote` returned
`6f74680c4c6a866999d271d15e681d239d9ff588 refs/heads/main`, identical to local HEAD.
245 local Markdown targets resolve; all current source/test/contract/raw hashes
and staged whitespace pass. This ledger/handoff delivery finalization is committed
and pushed separately on top of the verified implementation. Ten GPU checks remain
pending and no learning outcome is claimed. Layout/native support limits remain
concrete blockers for later fits. Stop after RD-06; next is RD-07 qualification.

## RD-07 recovery/run — 2026-11-08

Clean main `42d6fa4` matches authenticated remote main. RD-01..06 source/test/
contract/raw evidence, migrations and pushed ancestors verified. Sandbox GPU
visibility failed; authorized host access sees exact 3090 UUID, PyTorch2.15/cu132
and one selected CUDA device. Existing repo server PID2094 and ComfyUI PID565871
are independent services, not matching qualification jobs; left unchanged.
No matching RD-07 job/manifest existed. GPU qualification now authorized, without
optimizer steps; source/data/native-support limits remain separate fit blockers.
Raw qualification manifest/log: `data/redesign-rd07-20261108/deferred-1/`; actual
PID and source hashes are written by `run_deferred.py`. One CUDA process, one
resident test W at a time; class cleanup verifies all320 W tensors/tying/reference.
All optimizer step entry points are blocked by the diagnostic launcher.

## RD-07 qualification — 2026-11-08

Technical acceptance: **pass**. Learning outcome: **not_applicable**. All ten
RD-03..06 deferred IDs plus RD07-GPU-PROJECTION-FP64, RD07-GPU-LONG-PARITY and
RD07-GPU-CANCELLATION-ENVELOPES pass on exact3090; no required technical check
pending. [Report](redesign-rd07-results-2026-11-08.md) and
[portable evidence](evidence/redesign-qualification-v1.json) bind executed
source/profile/checkpoint hashes, exact commands/results, failures and raw files.
Raw `data/redesign-rd07-20261108/`; final runs `deferred-final/` PID574507 and
`changed-2/` PID572811 completed, no matching job remains.

Final ten deferred tests pass in155.915 s; three new GPU tests pass in198.589 s.
Twelve direct1024 traces/nine comparisons pass, including final cache states;
all13 operation/status cells, exact256 request/257 atomic rejection/64 return,
four cancellation scenarios, native continuous gradients and complete-window
resume/RNG/sampler/owners pass. All320 W tensors/physical tying/reference checked
in five final class cleanups. Zero optimizer guard calls or updates; historical
188 counters are restored only. Unchanged324/108 numeric matrices and raw child
hashes verified; reuse applies only to unchanged kernels. New paths have actual
FP32/FP64/VJP coverage under unchanged tolerances.

Initial failure: reported Noul raw node was outside delivered-value graph;
shared subtraction node repairs the return-specific VJP. Subsequent cancellation
fixtures found pre-request58/50 and pre-return67/58 ledger/cache divergences;
pre-charge guards plus completing/discarding charged work repair them. Initial
failed logs remain unchanged. Current normal tensor path is pinned by AST tests.
Model-free suite88/89 initially pass (116.988 s); source-contract whitespace
assertion fixed, all three new tests pass (0.061 s), yielding89 distinct current
passes with unchanged86 results reused. Four current-source, model-import-blocked
CLI preparations pass; direct repeats identical, all preparation execution
counters0. Compilation/staged whitespace pass. Predecessor migration verified.

No training, learned result, agents, new data/reviews, locked access, deployment
or service/Operating Table changes. Native semantic/Reason masks preserved.
No technical waiver or capability claim. Direct receiver/native supervision
limits still block affected fits; RD-08/09 unstarted. Implementation/evidence commit `7da99c652fba8729df247c62fcb029fb931d5cea`
is pushed: authenticated push exit0 (`42d6fa4..7da99c6 main -> main`); subsequent
authenticated `git ls-remote` returned the identical full SHA for main. This
ledger/handoff/evidence delivery finalization is committed and pushed separately
on top. 190 current local Markdown targets resolve; exact source/test/raw hashes
verify. Stop after RD-07, no required technical checks pending.

## RD-08 recovery/preflight — 2026-11-08

Submitted authorization: finite two-arm receiver comparison only, four passes over
all1500 supported records per arm,6000 exposures/750 eight-example windows.
Started on clean main `de5807391f6ae36e5fda32480900188db908d99a`, matching
authenticated remote main. Host read-only process inspection found only existing
server PID2094 and no matching RD-08/receiver fit or qualification process. Exact
3090 is accessible on host; no GPU-access blocker. No service changed.

RD-01..07 evidence/source/raw/commit verification and common-support preflight
are in progress. No fit, optimizer update or development evaluation launched.

## RD-08 coverage blocker — 2026-11-08

Delivery: **blocked before fitting**, not a completed failed experiment.
RD-01..07 remain delivered; RD-07 technical qualification: **pass**, all13 IDs
and executed source/test/raw hashes unchanged. Seven predecessor implementation
commits and their original source/test/contract/raw evidence verified in order.
Valid unchanged tests/measurements reused; no unnecessary GPU rerun.

[Report](redesign-rd08-results-2026-11-08.md),
[portable evidence](evidence/receiver-comparison-rd08-v1.json), raw
`data/redesign-rd08-20261108/`. Model-import-blocked command
`PYTHONPATH=. python data/redesign-rd08-20261108/verify_and_record.py > data/redesign-rd08-20261108/preflight.log 2>&1`:
exit0. Both prior current-byte plans have identical full order:1500 records ×4,
6000 exposures/750 windows per arm. Full coverage sources/raw/tokenizer/release
and all1810 record identities verify. All1500 training rows exceed return64
(observed89–166);1136 questions also exceed32. All501 Noul/497 Choice/502 Score
and288 families retained. E0/130 and X0/180. Common support is zero regardless
of control support; no pooled-control support count was invented.

Prompt8 explicitly says: "If complete shared support is unavailable, record the
coverage blocker and do not silently substitute a smaller experiment."
Therefore neither fit or development evaluation was launched. Actual per arm:
0 exposures/windows/updates/measurements. Learning acceptance **not_evaluated**;
no receiver artifact or new checkpoint. No training failure or useful-improvement
pass/fail claimed. No envelopes, semantic masks, data, grading or exposure changed.

Next recovery: explicit revised complete-role layout contract, full training/E/X
coverage audit and affected qualification, then the exact existing-runner full
matched schedule/initialization/moments/RNG/owner binding. Existing finite RD-08
authorization persists; no automatic redesign search or smaller fit. RD-09 stays
unstarted and blocked. No model workload, agents, new records/reviews, locked
access, deployment or service/Operating Table change. Blocker record commit `2c977ce81caa7235bf8b602d221d0c4b40e1b1de` is pushed;
authenticated push exit0 (`de58073..2c977ce main -> main`) and authenticated
`git ls-remote` returned the identical SHA.197 local Markdown targets, artifact
hashes and staged whitespace pass. This ledger/handoff/evidence finalization is
committed/pushed separately on top. RD-08 remains blocked rather than delivered.

## RD-09 request / predecessor recovery — 2026-11-08

Conditional RD-09 submitted. Clean main
`9af92fd93114469466933582e6dbbe5f60af60d1` matches authenticated remote main.
Required recovery is rechecking earliest unfinished RD-08 before dependent native
work. RD-09 execution remains unstarted. Host3090 is accessible; read-only process
inspection found existing server PID2094 and no matching receiver/native fit or
qualification job. No service changed. No model execution/update launched.

## RD-09 prerequisite result — 2026-11-08

[Report](redesign-rd09-results-2026-11-08.md) and
[evidence](evidence/native-demonstration-rd09-v1.json); raw
`data/redesign-rd09-20261108/`. Earliest incomplete predecessor remains **RD-08**:
0/1500 shared training support, neither arm started, acceptance not_evaluated,
accepted receiver absent. This is not a completed failed fit. A completed failed
RD-08 would also block native training without automatic reruns/search.
RD-01..07 evidence/raw hashes and pushed ancestors verify; all13 RD-07 checks
remain pass against current unchanged source/tests, no pending technical check.

`PYTHONPATH=. python data/redesign-rd09-20261108/verify_prerequisites.py > data/redesign-rd09-20261108/prerequisites.log 2>&1`:
exit0, model imports blocked. Verified RD-08 row-level proof/dependencies and
reused RD-06 native audit:5590 records,3220 calls (3150 train/70 dev),0 complete
formation,0 native decision/post-call continuation,0 direct supplied-reader
support across750 train/120 dev. Typed native-final extraction unsupported and
actual-native X pairs0. Separate no-call/replay potentials500/1000 do not qualify
actual-return supervision. No annotations, masks, limits or acceptance changed.

Conditional RD-09 authorization recorded; execution remains unstarted and
blocked. Native schedule/eligible manifests cannot be fixed as executable without
valid supervision/accepted receiver. Actual training attempts/exposures/windows/
updates/model workloads/evaluations/checkpoints0; no frozen-receiver retention
measurement or learning outcome. No source change or redundant GPU test run.
No agents, new data/reviews, locked access, deployment or service/Operating Table
change. No new slice delivered. Recover layout/full support and RD-08 acceptance
first, then valid native supervision; commit exact finite native manifests before
any conditional execution. Blocker-record commit `a864b65515831daada0616e2a19b7db0b8175bff` pushed:
authenticated push exit0 (`9af92fd..a864b65 main -> main`); remote main returned
the identical full SHA.207 local Markdown targets, dependency/raw hashes and
staged whitespace pass. Ledger/handoff/evidence finalization is committed/pushed
separately on top. RD-08/RD-09 remain blocked; native execution unstarted.

## RD-08 explicit format recovery — 2026-11-08

User explicitly authorized revising the format to fit all1500 existing examples,
full coverage recheck, changed-path qualification and resuming fixed RD-08.
RD-09 must not start; no new data. Clean pushed main4770032 recovered; host3090
accessible, existing server2094 untouched, no matching fit. Versioned compact
returns will bind immutable existing roles by Ref instead of re-echoing content.
Both matched arms use compact returns; candidate carries all role states with
explicit63-state field capacity. Existing256/64 envelopes and old profiles stay.
Native v2 writing is unsupported/unstarted; no old native length owner is resized.
Contract is committed before runtime implementation. No fit launched yet.

## RD-08 v2 implementation/qualification — 2026-11-08

Contract c1021a3 committed/pushed before implementation. All1500 original training
examples now fit without cropping, new data or request/return envelope increases.
Both arms use compact bound returns; direct fields explicitly1..63, old native
length head unchanged/frozen. Full1810 metadata audit and field input/mapping
parity with v1 pass; common train1500/1500, E127/130, X180/180. Three E oversize
requests retained as failures, no performance-based selection.

103 final static tests pass (137.464 s). GPU1:16 tests,15 pass/1 pooled-Score
admission failure; whole-rubric exact normalization repairs it. GPU2:3 pass on
final source: real input/endpoint delivery, resume, exact both-format8-example
accumulation.24 passing1024 traces/18 comparisons and all-status/cancellation/
FP64 methods are unchanged and reused with pinned source migration.17 required
IDs resolved; all320 W/tying/reference checks pass,0 qualification optimizer calls.
No numeric/optimizer kernel or tolerance changed. Evidence:
[coverage](evidence/compact-role-layout-coverage-v2.json),
[qualification](evidence/compact-role-qualification-v2.json),
[finite matched plan](evidence/receiver-matched-v2-plan.json).

Current-source model-import-blocked preparations pass, both full6000-element
orders/initial checkpoints/shared owners/coefficients match exactly; candidate
only adds identity projection owners with fresh absent moments. Existing runner
loop consumes these full plans, retains750 windows/arm and initial/final actual
head+delivered measurements. Both fits finish before dev evaluation. All47
historical passing Choice IDs in the fixed50 retention panel are pinned before
fitting. RD09/new data/deployment excluded. Implementation/evidence is committed
and pushed before fitting; no training update yet at this entry.

## RD-08 control running — 2026-11-08

Implementation/qualification/finite plans4137167af95449452975233411cf932a1de494cc
pushed and exact remote main verified before fitting. Control actualPID597060,
`data/redesign-format-v2-20261108/control/report.json`, log`control.log`, initial
complete-window checkpoint`control/checkpoint-0188/`. Explicit3090 UUID; standard
`python -m scripts.train_receiver_probe --execute-gpu --input-profile frozen-w-pooled-compact-v2 --prepared-plan data/redesign-format-v2-20261108/plans/control/plan.json` with pinned snapshot/output.
Initial full1500-row no-grad head+delivery measurement is active;0 updates at
launch. Candidate/evaluations unstarted. Reconnect to this process, never duplicate.
No RD09/new data/services change. Run source archives/plan and exact PID are local.

## RD-08 exact first-update recovery — 2026-11-08

Control PID597060 completed initial1500 measurements and1 update/8 exposures;
exact first-update model/moments/CUDA+CPU+Python RNG round-trip passed. Next input
failed `frozen reference storage/version changed`: identical checkpoint copies
advance tensor counters. Original run is retained FAILED_PROBE, not a completed
failed learning comparison. Checkpoint0189 df5dafd3817404e15bd669944ab8f56deb680140e56377414498d9e435170c5a
preserves the exact complete window/cursor8; no missing state.

Cache guard now rebinds version metadata only after complete original reference
checksum/frozen-ownership verification. Changed bytes/artifact/ownership reject.
All payload/model/optimizer methods and full1810 BPE mappings unchanged; coverage
receipts explicitly migrate the source guard.18 qualified IDs: targeted real3090
restore/cache parity1 pass32.961s, heads/losses exact and cache hits2,0 updates,
W320/tying/reference pass. Three compact static tests pass0.109s (104 distinct
model-free tests including103 prior unchanged results). Exact metadata-only
checkpoint migration and retained observation/source receipts are in
`data/redesign-format-v2-20261108/control-migration.json`. New fixed plans bind
this qualified source; cursor8, one completed update,749 windows remaining.
Code/qualification/migration are committed/pushed before resuming. Candidate,
all development evaluation and RD09 remain unstarted. No fitting is repeated.

## RD-08 resumed control active — 2026-11-08

Cache repair/migration a8c33a0fbd5cc4b8f8faa7c3c23caf45f766b073 pushed and remote
verified before resume. ActualPID618710, log`data/redesign-format-v2-20261108/control-resumed.log`,
manifest`control-resumed/report.json`; exact `--resume .../migrated-control/checkpoint-0189`
manifestSHAe4c4604247a1a64f572e522cc3460dc50882451ab4d00fd17426b9d863b60066,
new qualified `plans-cache/control/plan.json`. Starts cursor8, preserves first
update and all1500 initial observations; no repeated model baseline. Unchanged
baseline copy and SHA proof`baseline-reuse.json`. Second cumulative update passed,
749 resumed windows scheduled total (global750). Candidate/evaluation/RD09 not
started. Source/plans stay frozen during fit; no automatic extra training.

## RD-08 user GPU hold — 2026-11-09

The user reported being told to turn off the GPU now and asked about lost
progress. Verified PID618710 was sent SIGSTOP only; process state T (stopped),
no new GPU/model work or probes. No service/fan/device change or automatic resume.
Control599 completed windows/updates,4792 presentations,1500 distinct records,
zero rejected exposures. Original first update plus598 resumed updates; all1500
initial observations and matched initial receiver/moment/RNG receipts preserved.
Candidate/final measurements/development/RD09 unstarted; learning not_evaluated.

Last disk checkpoint `control-resumed/checkpoint-0688`, SHA
`2eaea42f7bbbbc0a125c51f0b06016afbda2fa1011c913a05c912202269d91aa`, all four
file hashes verified model-free. Historical688 counters mean500 new RD08
updates/cursor4000.99 additional completed updates/792 presentations are only in
live process/GPU memory, plus possible partial window600. Their measured training
time is3848.013s/64.13 minutes. Power/reset/termination loses that live state;
ordinary pause retains it while process/device memory survive. Do not fabricate
exact599 recovery from500 checkpoint or duplicate a stopped matching process.

See [pause evidence](evidence/receiver-matched-v2-pause-2026-11-09.json),
[updated report](receiver-matched-v2-results-2026-11-08.md) and handoff. Raw paths
`pause-20261109.json`, `control-resumed/report.paused-20261109.json` and
`active-performance-observation.json` pin exact observations. Source remains
a8c33a0, previously pushed/verified; qualification18 IDs resolved and104 distinct
model-free test results retained. No new model workload for this pause record.

Utilization check: process122% CPU/about1.2 cores; host9.54% busy on12 logical
CPUs over2s, sparse per-process GPU samples10/1/17%. Last125 windows38.817s mean,
14.272 component VJPs mean, correlation0.826. Separate backwards/host integrity
readbacks remain; no component profiler or measured optimization. No fit setting
or source change. Current execution hold supersedes earlier GPU reopening.
