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
| RD-03 | delivered | Source/cache policy and static checks; explicit deferred GPU check IDs | [Report](redesign-rd03-results-2026-11-08.md), [evidence](evidence/frozen-w-source-v1.json); 50 static tests pass; three GPU checks pending RD-07; `2cbfa41` pushed and verified |
| RD-04 | delivered | Versioned payload/admission contracts and implementation; deferred GPU IDs | Contract `2247619`, implementation `ae800a4` pushed/verified; [report](redesign-rd04-results-2026-11-08.md), [evidence](evidence/direct-role-states-v1.json); 61 static tests pass +6 refreshed; three new GPU checks pending RD-07; zero receiver support |
| RD-05 | delivered | Joint objectives, owners, binding/mask tests; deferred GPU IDs | `fe90af0` pushed/verified; [report](redesign-rd05-results-2026-11-08.md), [evidence](evidence/joint-receiver-objectives-v1.json); 74 static tests pass +14 refreshed; two new GPU checks pending RD-07; Reason masks preserved |
| RD-06 | in_progress | Native wiring/eligibility report; deferred real-return/gradient IDs | [Report](redesign-rd06-results-2026-11-08.md), [evidence](evidence/direct-native-flow-v1.json); 86 static tests pass +10 refreshed; implementation ready to commit/push; zero complete corpus formation/semantic continuation/direct reader support |
| RD-07 | pending | All required deferred/changed-path 3090 checks pass; actual counts/hashes | RD03-GPU-D-INDEPENDENCE, RD03-GPU-CACHE-PARITY, RD03-GPU-CONSUMER-GRADIENTS; RD04-GPU-IDENTITY-PROJECTION, RD04-GPU-DIRECT-ROUNDTRIP, RD04-GPU-DIRECT-GRADIENTS; RD05-GPU-GRADIENT-ACCUMULATION, RD05-GPU-RESUME; RD06-GPU-NATIVE-RETURN, RD06-GPU-CONNECTED-GRADIENT pending; no GPU execution |
| RD-08 | pending | Complete matched fit/evaluation receipts; useful-improvement outcome separate | None; training not authorized by prompt preparation |
| RD-09 | pending | Accepted RD-08 receiver plus eligible native run/evaluation receipts | None; training not authorized by prompt preparation |

Earliest unfinished implementation slice: **RD-06**, implementation/checks complete,
awaiting commit/push and delivery finalization. RD-07..09 are unstarted.
Ten RD-03..06 GPU checks remain pending RD-07; no redesign model job was launched.

For each attempted slice add a short entry with implementation/evidence paths,
exact commands/results, dependency hashes, relevant implementation commits and
verified push status. List deferred checks by stable IDs and their owning slice;
RD-07 resolves them. Record active run manifest/PID/log/checkpoint paths when a
long job starts, and update them when it finishes or is interrupted. Store raw
artifacts locally under `data/`; commit concise reports/portable evidence only.

Current RD-06 implementation delivery blockers: **none**. Future real-data fit
blocker: the initial direct-role layout admits zero receiver rows (train1500,
E130, X180). An explicit layout revision and full re-audit are required before
that fit; no subset, allowance increase or waiver was applied. RD-01 data/native
supervision limits and pending later technical checks also remain.

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

Currently due model-free acceptance: **pass**; staged delivery awaits commit/push
and finalization. GPU qualification: **pending RD-07**. Learning outcome:
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
