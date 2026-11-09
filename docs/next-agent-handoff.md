# Next-agent handoff — 2026-11-08

Read this first for current status, then the
[implementation specification](architecture-and-training-plan.md) and its linked
topic contracts. Historical reports describe their recorded sources and runs.
This handoff records measured mechanisms, not learned capability.

## Current candidate progress — 2026-11-09

CandidatePID878908 completed the full1500 initial observations and250 of750
updates. Shared initial weights/moments/CUDA/Python RNG match control exactly;
first-update restore round-trip passes. Checkpoint0438 stores250 effective
updates/cursor2000, manifestSHA9415d6f32b455c356167acc8ef0f13a55c90ba9b1c85c3b093450fc67fe32c8b;
all four files verify. Prior125 checkpoint0313 retained. Raw verification receipts
`candidate-start-verified.json`, `candidate-checkpoint-{125,250}-verification.json`.
No rejected exposures. ActualPID878908, host session62708, same output/log/plans
as below. Do not duplicate or change sources/qualified evidence/plans midfit.
Continue to750, then all1500 final observations; only then evaluate both arms.
CPU/language questions did not authorize a midfit rewrite or extra learning.

## Current execution — control complete; direct candidate starting, 2026-11-09

Control `data/receiver-performance-20261109/control/report.json` is
COMPLETED_RD08_TRAINING_NOT_EVALUATED, process exited0.750 retained windows/
updates,1500×4/6000 effective presentations,0 failures; final1500 head/delivery
observations complete.250 actual updates in the recovery process. All320 W/
tying/reference/R/nonreceiver owners frozen, grad=None; initial common masters/
moments/CUDA/Python RNG match original parent receipts. Final checkpoint0938 SHA
f616f4c9154ff2a4c4b7d5b58b5f8d67b9afd55baf80d758e5e234a0b3073452.
Report SHA3e0cd4bad2efcf315c683c8266d707859979fed029db3fedfad9fa9395c9d091,
verification`control-verified.json`. Prior625 checkpoint remains. Replay99
windows/792 examples match preserved original receipts exactly; physical lost/
replayed work and unknown unflushed work remain explicit.

Candidate was launched from original parent188, not control, with exact3090,
qualified retained policy and `plans-qualified/candidate/plan.json`. New output
`data/receiver-performance-20261109/candidate`, log`candidate.log`, host exec
session62708, actualPID878908. Reconnect to manifest/PID; never duplicate. It first measures
all1500 initial heads/deliveries, then exactly750 windows/6000 presentations,
then final1500 observations. Both fits must finish before either E/X evaluation.
No dev fitting/checkpoint selection/extra pass/data/agents/RD09 or service change.
Sources/qualification/plans remain frozen. Final evaluation uses current runner
`--evaluate-gpu`/exact final checkpoint/hash/completed peer report; reduction is
`data/receiver-performance-20261109/analyze.py`. No evaluation has started.

## Previous execution — optimized RD-08 control active, 2026-11-09

Qualification/recovery commit6793f252b4259ff2a19bc0ffb54e349a68374d2c was pushed
and its exact remote SHA verified before launch. Active controlPID829981,
`data/receiver-performance-20261109/control/report.json`, log`control.log`, host
exec session23580. Explicit3090 and `--execution-policy retained-activations-v1`;
prepared plan`plans-qualified/control/plan.json`, migrated checkpoint0688 SHA
00928d0150dc6e16055206651a0fb865b2b888ce2b94deca298075e7bd0ed968. It loaded
500 retained updates/cursor4000 and executes the remaining250 windows. All1500
original baseline observations are copied unchanged into control/panel-00.jsonl;
receipt`baseline-reuse.json`, no baseline model rerun.

The99 lost logged windows have now been replayed;792 original input/head/loss/
eligibility/active-owner/update-norm receipts match exactly. Original recorded
window time3848.013s; recovery3502.353s with fresh caches, not a matched-cache
benchmark. This does not fabricate an old599-state checkpoint. Raw proof
`replay-parity.json`; only500 were recoverable on disk. No lost work is hidden.
The control is now past599 retained updates, still targeting750 total.

Do not change production source, qualified evidence or prepared plans during
fitting. Reconnect, never duplicate. After control COMPLETED_RD08_TRAINING_NOT_EVALUATED
with750 windows/6000 effective exposures, launch candidate from original188
using plans-qualified/candidate/plan.json, same retained policy, new output
`data/receiver-performance-20261109/candidate`. Both fixed fits must finish before
either E/X evaluation. Existing runner `--evaluate-gpu`, final checkpoint/hash and
completed `--peer-report` enforce that. Never start RD09. Current reduction script
`data/receiver-performance-20261109/analyze.py` preserves predeclared acceptance,
original graders/cohorts and explicit physical lost/replayed-work accounting.

## Qualification — measured performance passes; RD-08 recovery, 2026-11-09

GPU permission reopened. The retained execution policy now passes PERF-01..03:
six alternating paired windows, pooled48.322→35.763s (−25.990%),
direct52.837→39.992s (−24.311%); CPU reductions37.367%/34.766%. Exact component
VJPs, updated masters/Adam moments, eligibility/None structure, frozen R/startup/
reference and transactional resume.12 discarded diagnostic updates/96 existing
presentations are separate from RD08. All19 current-source core checks pass with
0 optimizer steps:24 long1024 traces/18 comparisons, all statuses/bounds,
cancellation/FP64/source isolation, resume and eight-record largest-request
windows in both interfaces. Maximum request256; largest direct peak4.18GiB.
Every class cleanup verifies all320 W tensors/tying/reference.35 final static
checks pass109.098s with model imports blocked. Sources/kernels/limits unchanged;
no full-catalogue speedup or100% GPU-utilization guarantee is claimed.

Updated qualification and measured performance evidence must be committed/pushed
and exact remote SHA verified before fitting. Qualified metadata plans use
`data/receiver-performance-20261109/plans-qualified/{control,candidate}/plan.json`.
The full1500×4/6000-presentation/750-window effective schedule, shared parent0188,
owners/moments/RNG/objectives and Reason masks remain. Candidate/evaluation/RD09
unstarted. Runtime policy must be selected explicitly as `retained-activations-v1`.

Resume control from checkpoint0688's500 retained updates/cursor4000 after a
verified metadata-only source/policy binding transition. Preserve every tensor,
Adam/RNG byte and sampler receipt. The prior599 logged updates include99 lost
windows after the last checkpoint; recovery replays that lost work rather than
claiming a599-state artifact or retaining extra effective passes. Record physical
completed-window work separately: prior599 plus250 recovery windows gives at
least849 actual completed optimizer calls versus750 retained updates; any work
between the last flushed receipt and process loss is unknown. Never hide it or
replace the effective schedule with a smaller experiment. Keep original logs,
failed prefix,1500 baseline observations and old manifests intact. Both fixed
fits finish before either development evaluation. No new data/agents/RD09 or
services/fans/Operating Table changes.

See [measured report](receiver-performance-v1.md),
[performance evidence](evidence/receiver-performance-v1.json),
[current core qualification](evidence/compact-role-qualification-v2.json).

## Previous work — GPU reopened; performance validation active, 2026-11-09

The user reopened GPU work with "GPU is open again." The declared finite
PERF-01..03 validation is active on the exact3090 UUID. No matching RD08 trainer
existed at launch; serverPID2094 remains untouched. Source6f90391 is archived
under `data/receiver-performance-20261109/gpu-1/source/`; execution manifest and
`tests.log` are there. Reconnect to the actual `tests.test_receiver_performance_gpu`
process, never duplicate. Packed CUDA hash parity/mutation test has passed;
matched twelve-discarded-update/six-pair validation is running. No promotion,
RD08 continuation, lost-state replay or RD09 has occurred. Affected long traces,
maximum-corpus memory and current-source qualification still precede use.

## Previous status — performance implementation; GPU hold persisted, 2026-11-09

The user requested improved performance. Implemented explicit
`retained-activations-v1`: one episode's activations retained instead of layer
recomputation for each component VJP, recurrence scratch capacity32, packed32MiB
fingerprint readbacks with byte-exact original canonical hashing. Default
`checkpointed-v1` remains unchanged. No shared eight-episode graph, combined
loss/VJP, precision/optimizer/eligibility change, skipped check, hash cache or OOM
fallback. Existing runner/plans bind and report the policy, reject mismatched
peer policies, and reject the candidate before model loading until source-bound
performance evidence and affected v2 qualification pass and are committed/pushed.

**GPU hold remains.** No model workload, optimizer update, device probe, training
resume, source/protocol checkpoint migration or service change in this task.
PERF-01 packed-hash parity, PERF-02 retained joint update/resume and PERF-03 paired
throughput/memory are pending. No actual speedup or retained-memory fit is claimed.
See [performance contract/report](receiver-performance-v1.md) and
[pending evidence](evidence/receiver-performance-v1.json). Future GPU test code
uses exactly12 discarded diagnostic updates/96 existing presentations, both
interfaces and three alternating pairs; it is not automatic RD-08 continuation.
Worst-case memory and affected long v2 qualification also remain prerequisites.

35 distinct model-free tests pass (34 initial plus21 refreshed including one new
peer-policy check); two GPU classes skipped. Model imports blocked. Real opaque
checkpoint bytes preserve all1024 tensors' canonical hash and original reference
artifact exactly;32MiB packs use4 groups for the full bundle and2 for reference.
These are planned byte groups, not measured CUDA copies/speedup.13 arithmetic/
optimizer/episode/checkpoint files, Reference class and full training-loop AST
are unchanged. The unrelated historical profiling-plan input-drift failure is
reproduced on baseline8e63aaa; retained, not waived.

**Recovery state changed externally:** former pausedPID618710 is now absent.
Read-only host process inspection found no matching replacement trainer; cause
unknown. No newer checkpoint exists. The control log still ends at599 updates,
but only500 are recoverable from checkpoint0688; all four file hashes verify.
The99 unsaved updates are not reconstructible from known artifacts. This task
did not signal/restart/kill the trainer. The serverPID2094 remains untouched.
Raw process receipt: `data/receiver-performance-20261109/process-state.json`.
Do not follow older instructions to SIGCONT the vanished PID, claim599-state
recovery or automatically restart/rollback to500. After GPU permission returns,
resolve missing-state/rollback accounting explicitly, qualify current sources,
and bind any compatible checkpoint migration. Candidate/E/X/RD09 unstarted.

## Previous operating permission — RD-08 GPU work paused, 2026-11-09

The user reported being told to turn off the GPU now and asked what progress
would be lost. Only the verified RD-08 trainer PID618710 was sent SIGSTOP; its
state is T (stopped). No new GPU work/probes or automatic resume. Existing
services, fan controls and devices were untouched. Pausing retains VRAM and
process memory; it does not power off the card or save a new disk checkpoint.

Control has599 completed updates/windows,4792 presentations across all1500
records, no rejected exposures, and1500 initial observations. Last complete disk
checkpoint is `data/redesign-format-v2-20261108/control-resumed/checkpoint-0688`,
manifest SHA `2eaea42f7bbbbc0a125c51f0b06016afbda2fa1011c913a05c912202269d91aa`.
All four checkpoint file hashes verify. Its historical counter688 equals500
RD-08 updates; sampler cursor4000. Destroying the process/GPU memory loses99
completed updates/792 presentations, measured3848.013 seconds (64.13 minutes),
plus any partial window600. The logs alone cannot reconstruct that missing
optimizer/RNG state. Continuing the same stopped process preserves live progress,
but requires the user to reopen GPU work first. Do not duplicate it.

Candidate, final training measurements, E/X evaluations and RD09 remain unstarted.
RD-08 is in progress with a user GPU hold, not delivered or a completed failed
learning comparison. No accepted receiver exists. Format support1500/1500,
18 GPU qualification IDs and104 distinct model-free test results remain recorded;
three oversized E cases and native/Reason supervision limits remain explicit.
See [pause evidence](evidence/receiver-matched-v2-pause-2026-11-09.json),
[v2 report](receiver-matched-v2-results-2026-11-08.md) and [ledger](redesign-progress.md).
The earlier running-process instructions below are historical and superseded.

Read-only performance observations found process CPU122% (about1.2 cores),
whole-host busy9.54% over2 seconds on12 logical CPUs, and sparse GPU activity.
Last125 windows averaged38.817s with14.272 separate component VJPs/window;
runtime/VJP correlation0.826. Separate backwards and per-example integrity
readbacks remain in the active path. These are observations, not a component
profile or measured optimization. No training setting/source was changed.

## Previous operating permission — GPU and fan-noise hold lifted, 2026-11-08

After temporarily closing both GPUs and banning work that ramps up the fans,
the user explicitly reopened work with "everything is allowed again." The
temporary noise restrictions on GPU workloads, device probes, benchmarks,
CPU-heavy jobs and test suites are lifted. Both GPUs are available within their
existing scopes: ordinary model/training paths use the 3090, and the 3060 is
limited to the explicit read-only two-GPU diagnostic scope. No CPU model fallback.
Existing no-agents/no-new-data constraints and finite experiment scopes remain;
this reopening does not launch or extend a completed training experiment. Do not
change fan controls, live services or The Operating Table. Older hold statements
below are historical. The completed comparison and next unfinished slice remain
as recorded below.

## Current requested work — explicit format recovery and RD-08, 2026-11-08

The user explicitly authorized fixing the format for all1500 existing examples,
rechecking coverage and resuming the finite RD-08 once qualified. **No new data
and do not start RD-09**, superseding earlier conditional RD-09 execution scope.
See [v2 report](receiver-matched-v2-results-2026-11-08.md),
[plan](evidence/receiver-matched-v2-plan.json),
[coverage](evidence/compact-role-layout-coverage-v2.json),
[qualification](evidence/compact-role-qualification-v2.json), [ledger](redesign-progress.md).

Full shared training support is now1500/1500: complete direct request roles with
explicit1..63 field capacity and compact bound returns for BOTH matched arms.
Metadata3/4; request256/return64 and other resource/numeric/semantic bounds stay.
All1810 original field input/mapping/length and record identities match v1. E
coverage127/130, X180/180; retain three E oversized requests as failed cases.
Native length33 head unchanged/frozen; new native semantic/reader paths unsupported.

103 static tests pass;17 GPU IDs pass through explicit source migration and delta
qualification. All24 traces reach1024 with18 paired comparisons; all-status,
exact256/257/39 bounds,8 cancellations, FP64/VJPs, joint accumulation, resume/
RNG/owners, all320 W/tying/reference pass. GPU1's pooled Score normalization
binding failure is retained and repaired exactly; GPU2 final-source3 checks pass.
No qualification optimizer step. Unchanged kernel matrices are reused only for
unchanged kernels. No source or service change outside this authorized scope.

Both prepared plans have identical1500×4 order,6000 exposures/750 windows, exact
parent0188 shared initialization/moments/RNG/schedule/objectives/owners. Candidate
projection is fresh zero-delta identity; W/R/budget/reference and length head stay
frozen. Existing runner loop is generalized; no copied loop/framework. Initial/
final full training heads AND actual delivery measured without fitting them.
Both fixed fits must complete before either E/X model evaluation. Protocol fixes
familiar-cohort Noul/Score, per-cell status and47 historical Choice retention IDs
before fitting. No dev selection, retries/extra passes, new data or deployment.

Earliest unfinished RD-08 is **in_progress**.4137167 implementation/qualification
was pushed and verified before control launched. Control PID597060 has now exited
FAILED_PROBE after full1500 baseline observations and1 saved update/cursor8.
Its exact first-update model/moment/RNG round-trip passed; next input failed only
a stale reference version-counter guard after identical checkpoint restoration.

The guard is repaired with full originally bound reference checksum/ownership
revalidation before rebind; changed content still rejects. Targeted3090 cache/
restore test passes32.961s, exact heads/losses/cache hits2 and W320/tying/reference,
0 steps; three compact static tests pass.18 qualified IDs now resolved. Payload/
model/optimizer math and metadata audit methods unchanged; original measurements
and first update remain valid with explicit source migration. New qualified plans
`data/redesign-format-v2-20261108/plans-cache/{control,candidate}/plan.json`.

Recovery checkpoint and exact file/protocol migration:
`data/redesign-format-v2-20261108/control-migration.json`, destination
`migrated-control/checkpoint-0189/`. All tensor/optimizer/RNG files unchanged;
original full order/owners/objectives/receipts retained. Resume at cursor8 with749
windows remaining; do not repeat the saved update. New output `control-resumed/`;
copy original `control/panel-00.jsonl` unchanged there after output creation,
preserving its original source provenance/initial0188 measurement. Original
control logs/report/checkpoints stay immutable. Candidate output `candidate/`
still unstarted and must use new plans-cache/candidate plan only after control
completes. Neither development evaluation has run. RD09 stays unstarted.

This cache fix/qualification/migration is committed/pushed before resume. Do not
alter source or plans during fits. Reconnect to actual report PID/log/checkpoint,
never duplicate. Both fixed fits finish before either evaluation; failure is an
actual recorded learning outcome, never permission for retries/extra passes or
native learning. Historical zero-support/source-pending statements below are
superseded by the qualified v2 recovery above.

## Previous requested work — conditional RD-09, 2026-11-08

The user submitted conditional finite RD-09 authorization. **RD-09 execution
remains unstarted/blocked**; earliest unfinished predecessor is **RD-08**. See the
[RD-09 report](redesign-rd09-results-2026-11-08.md),
[evidence](evidence/native-demonstration-rd09-v1.json) and [ledger](redesign-progress.md).

RD-01..07 evidence/raw hashes/pushed ancestors verify and all13 RD-07 GPU IDs
remain pass for unchanged current sources. RD-08 still has0/1500 complete shared
support, no fit, no useful-improvement result and no accepted receiver artifact.
RD-08 is blocked before fitting, not a completed failed experiment. If completed
RD-08 fails acceptance, native training stays blocked; no automatic receiver
rerun/search. Existing finite RD-08 and conditional RD-09 authorizations persist
only within their stated conditions.

Verified native audit:5590 records;3220 calls (3150 train/70 development writer),
complete formation0, native decision/post-call continuation0, direct supplied
reader0 across750 train/120 dev. No verified typed-final extractor or actual-native
X pair labels. Direct/no-call500 and language-replay1000 training potentials cannot
substitute for actual-result use. Existing semantic masks/limits remain intact.

Model-import-blocked prerequisite verifier exits0, records dependencies and reuses
valid unchanged tests/audits without GPU reruns. Actual new model workloads,
training attempts/exposures/windows/updates, evaluations and checkpoints are all0.
No accepted receiver exists for retention measurement; learning not_evaluated.
No native schedule was fixed as executable and no dependent implementation begun.
Host3090 accessible, no matching fits/qualification job, existing server PID2094
untouched. No agents/new data/reviews/locked access/deployment/service changes.

Recovery order: explicit complete-role layout revision/full training/E/X re-audit
and affected qualification; full authorized RD-08 comparison with accepted receiver;
valid native supervision; then fix/commit exact finite native eligibility/owners/
coefficients/forcing/evaluation manifests before training. Do not crop, invent
labels, weaken acceptance, substitute a smaller experiment or start automatic
search/extra passes. Blocker record `a864b65515831daada0616e2a19b7db0b8175bff` is pushed and its exact
remote SHA verified; ledger/handoff/evidence finalization is committed/pushed
separately on top.207 current local Markdown targets resolve; dependency/raw
hashes and staged whitespace pass. Historical
statements below about absent RD-09 authorization are superseded here.

## Previous requested work — RD-08 receiver comparison, 2026-11-08

The user submitted/authorized finite RD-08 training, but **RD-08 is blocked before
execution**, not delivered as a failed experiment. See the
[report](redesign-rd08-results-2026-11-08.md),
[evidence](evidence/receiver-comparison-rd08-v1.json) and [ledger](redesign-progress.md).
All RD-01..07 deliveries were verified; RD-07's thirteen GPU passes remain valid
for unchanged current source, with raw evidence and pushed ancestors verified.
No technical qualification check is pending; no GPU rerun was needed.

Required complete shared support is absent: direct training0/1500, E0/130, X0/180.
Every training return reservation exceeds64 (89–166 positions);1136 questions
also exceed32. All501 Noul/497 Choice/502 Score rows and288 families remain in
the derived manifest, with all1810 training/E/X IDs/hashes/costs/exclusions.
Prompt8 explicitly forbids substituting a smaller experiment when complete shared
support is unavailable. Both plans still declare the identical full1500×4 order,
6000 exposures/750 windows per arm; those are planned counts only.

Actual per arm:0 exposures,0 windows,0 updates,0 measurements/evaluations. No fit
or checkpoint exists; learning acceptance **not_evaluated**. Model-import-blocked
verification exits0. Existing full-panel audits/tests reused after dependency
verification. Host inspection found3090 accessible, no matching fit/qualification
job, and existing server PID2094 untouched. No model workload ran in RD-08.

Earliest unfinished slice: **RD-08**. Recovery needs an explicit layout revision
carrying complete existing roles, full training/E/X re-audit and affected GPU
qualification, then exact existing-runner full matched schedule/owners/moments/RNG.
Do not weaken limits, crop, invent labels, silently select a subset or start an
automatic architecture search. The user's existing finite RD-08 authorization
persists after prerequisites pass; no RD-09 authorization was given. RD-09 remains
unstarted and no fitted receiver qualifies it. No agents, new data/reviews, locked
access, deployment or service/Operating Table changes. Earlier statements that
RD-08 was unsubmitted are historical and superseded here. Blocker record `2c977ce81caa7235bf8b602d221d0c4b40e1b1de` is pushed and exact
remote main SHA verified. Ledger/handoff/evidence finalization is committed and
pushed separately on top. Do not mark RD-08 delivered.

## Previous requested work — RD-07 qualification, 2026-11-08

RD-07 is **delivered**, with technical acceptance **pass**. Implementation/evidence
commit `7da99c652fba8729df247c62fcb029fb931d5cea` is pushed to main, and exact
authenticated remote SHA verified. Ledger/handoff/evidence delivery finalization
is committed and pushed separately on top. See the [report](redesign-rd07-results-2026-11-08.md),
[portable evidence](evidence/redesign-qualification-v1.json) and
[ledger](redesign-progress.md). RD-01..06 predecessor evidence was recovered and
verified from clean pushed main42d6fa4; all ten deferred GPU IDs are now resolved.

Actual final execution: ten deferred tests pass (155.915 s) plus three changed-path
checks (198.589 s), on exact RTX3090 UUID only, PyTorch2.15/cu132. Twelve direct
traces reach1024 across all three ports and functional/checkpoint/serving/FP32;
FP64 projection forward/VJP, all13 operation/status returns, exact256/257/64
boundaries, cancellation, original-prefill/native-return gradients and complete
window save/resume/RNG/owner integrity pass. All320 W tensors/tying and reference
identity pass at every final class cleanup. Zero optimizer steps/updates;
existing historical counter188 was restored, not incremented. No training ran.

Routine failures were repaired: Noul's reported raw node now feeds its delivered
value; cancellation checks precede ledger charges, while already charged work
finishes before late output is discarded. Failed attempts, executed sources,
metrics and exact commands are retained in `data/redesign-rd07-20261108/`.
All89 distinct current model-free tests pass (86 unchanged results reused plus
three refreshed contract tests); current CLI preparation repeats identically.
Unchanged historical324/108 numeric matrices are hash-verified and reused only
for unchanged kernels. No tolerance, envelope or semantic mask was relaxed.
Preparations still conservatively block nonlegacy fitting; this qualification
does not remove execution/admission guards or authorize a fit.

**No pending RD-07 technical checks. No learning outcome.** Engineering fixtures
force controls/status/budget/existing tokens; these do not prove semantic native
invocation or final-answer improvement. Direct receiver support remains0/1500
train,0/130E,0/180X; direct formation0/3220 calls; native decision/continuation and
direct supplied-reader support0. Typed native-final extraction and actual-native
X-pair labels remain absent. Explicit layout revision/re-audit and valid labels
are needed before affected fits, preserving32/256/64 bounds. Historical profiling
receipt/input drift remains unmodified and is not reported passing.

RD-08 is the earliest unfinished slice; RD-08/09 remain unstarted and outside
this submission. No agents, new data/reviews, locked access, deployment, service
or Operating Table change. Older pending statements below are historical and
are superseded by this section and the RD-07 evidence.

## Previous requested work — RD-06 native flow and supervision, 2026-11-08

RD-06 staged implementation is delivered. Implementation/evidence commit
`6f74680` is pushed to main and its exact remote SHA was verified; ledger/handoff
delivery finalization is committed and pushed separately. Read the
[RD-06 report](redesign-rd06-results-2026-11-08.md),
[evidence](evidence/direct-native-flow-v1.json),
[native contract](direct-native-flow-v1.md) and [ledger](redesign-progress.md).

The opt-in direct EpisodeProducer now uses the original prompt prefill states,
live ordered writer fields/predicted lengths, actual Qev evaluation and bound
latent return in one cache. Direct offline annotation targets are prepared before
Session creation; no re-encoding or gold return enters live invocation. Explicit
native admission permits applicable control/formation objectives but keeps native
decision/Reason/post-call continuation masks and rejects dev accumulation. Forcing,
actual returned values/Refs, native phases and teacher-forced tokens are explicit.
Default Session/runtime, coefficients, optimizer/checkpoint and numeric kernels
remain unchanged. Legacy controls remain available; no reference qualification
was fabricated or transferred to the new native profile.

**Actual supervision eligibility limits:** all 3,220 reviewed writer/interleaved
calls exceed return64, so zero complete direct formation targets; unverified
native meanings also leave zero decision/post-call continuation targets. Direct
supplied-reader support is zero across training750 /development120. All 5,590
relevant records, fields, costs and exclusions remain in metadata. Existing no-call
direct/language replay and legacy supplied-reader potentials remain separate,
not evidence of actual-return understanding. Native fits cannot proceed on
invented labels or control-only/zero-semantic-loss demonstrations. The earlier
zero direct receiver support also remains a fit blocker requiring explicit layout
revision/re-audit. No bounds, panel, tolerance or data/review was changed.

Held-out plan: native W/F470, prepared receiver X180 and supplied-reader R/X9
120; all 100 X pairs retain their actual exposure (90 prepared /10 supplied).
There are no actual-native-final X pair labels. Explicit native W/F evaluation
uses free control and no targets/effort/override and cannot enter training.
Typed-final grading is separate from receiver heads/consumed returns; extraction
of typed values/identities from existing prose remains unprovided/unsupported,
with no invented parser/judge. No actual final answer or pair result was measured.

All **86 model-free tests pass** (116.312 s), plus ten refreshed native tests
(14.224 s) after final telemetry/typed-field changes; final GPU registration guard
passes. Full Rust BPE/Jinja audit and four current-byte CLI preparations pass
with model imports blocked. Repeated direct plans are identical; all actual
execution counters zero. Source migrations/old raw artifacts and 33 unchanged
boundary checks were verified. Historical profiling input drift remains unchanged.

**Ten GPU checks remain pending RD-07**, never executed. The eight earlier IDs
below plus `RD06-GPU-NATIVE-RETURN` and `RD06-GPU-CONNECTED-GRADIENT` are implemented
in `tests/test_native_redesign_gpu.py`. Exact 3090 UUID and explicit submitted
RD-07 opt-in precede model imports. Engineering forcing: guided controls, Assessed
status, one existing token and budget512. The first covers Noul/Choice/Score;
the second isolates native-loss -> actual return -> frozen receiver -> continuous
writer/R gradients. No optimizer update or corpus semantic qualification is claimed.

No model execution, actual Session, extraction, fitting, agents, new data/reviews,
locked access, deployment, service or Operating Table change. Stop after RD-06;
**RD-07 is unstarted**. Later fits remain blocked by support limits and unrun checks.
Current evidence hashes, compilation, staged whitespace and 245 local Markdown
targets pass.

## Previous requested work — RD-05 joint objectives and binding, 2026-11-08

RD-05 staged implementation is delivered. Implementation/evidence commit
`fe90af0` is pushed to main and its exact remote SHA was verified; ledger/handoff
delivery finalization is committed and pushed separately. Read the
[RD-05 report](redesign-rd05-results-2026-11-08.md),
[portable evidence](evidence/joint-receiver-objectives-v1.json),
[prepared-component contract](prepared-receiver-component-v1.md) and
[ledger](redesign-progress.md).

The new profiles have explicit experimental prepared-receiver admission bound to
an exact reviewed permitted view, independent encoding, actual sealed snapshot,
source states/coordinates and ordered candidates. Labels attach after the complete
forward and revalidation. Null reference qualification/unaccepted behavior are
explicit. Native/writer semantics keep their masks; the diagnostic-reference guard
remains. Empty failed attempts retain their scheduled slot without losses.

Existing joint status/Noul/Choice/Score and reviewed Boolean feasibility losses,
coefficients and eligible-example means remain. Eligibility was extracted without
changing its AST. New-profile Reason targets are unsupported and masked, never
encoded/zero-filled; the old supported post-forward Reason/reader exception stays.
All 832 reviewed Reasons remain in metadata. W, R, startup row and reference stay
frozen; writer length head is frozen during receiver work. Exact owner specs and
parent-to-direct Window construction preserve old masters/moments/RNG/counters;
new projection moments stay absent until actual eligible gradients. No step occurs.

Joint reports separate head predictions from actual delivered grades, per-cell
status/value evidence, eligibility and paired startup retention. Private row freeze
does not protect shared D features. VJP diagnostics read per-owner norms/counts
from actual Window buffers without another model/gradient call. No actual delivery,
retention or owner gradient has been measured for this redesign.

All **74 model-free tests pass** (120.531 s), plus **14 refreshed binding/dispatch
tests** (22.151 s) after a failed-empty-exposure fix, including one added test;
75 distinct tests covered. Final GPU collection/registration guard test passes.
Four final CLI preparations under `data/redesign-rd05-20261108/verified-final/` pass
with model imports blocked, repeated direct plans byte-identical and all actual
execution counters zero. Earlier preparations are retained as superseded after
source changes; do not relabel them current. Predecessor deliveries/raw artifacts
and their current source migrations were verified. Core loss/Window/checkpoint,
Session/runtime, native producer/binding and numeric algorithms remain unchanged.

**Eight GPU checks remain pending RD-07**, never executed: the six RD-03/04 IDs
below plus `RD05-GPU-GRADIENT-ACCUMULATION` and `RD05-GPU-RESUME` in
`tests/test_joint_receiver_gpu.py`. They require submitted RD-07, exact 3090 UUID
and explicit opt-in before model imports. The VJP fixture uses an existing
frozen-W/pooled reviewed component; the resume fixture uses parent/direct owners,
real moments/RNG and declared engineering perturbations. No optimizer update or
semantic qualification is claimed.

**Future-fit blocker unchanged:** direct layout supports 0/1500 train, 0/130 E,
0/180 X. An explicit layout revision and complete re-audit are required before a
real-data fit; no allowance increase, selected panel or invented annotation was
used. Metadata objective potentials are not admitted or measured examples.
Existing RD-01 supervision/native gaps and historical profiling-test drift remain.
No model execution, feature extraction, fitting, agents, new data/reviews, locked
access, deployment, service or Operating Table changes. Stop after RD-05;
**RD-06 is unstarted**. Compilation, final staged whitespace checks, evidence
hashes and 217 local Markdown targets pass.

## Previous requested work — RD-04 direct role states, 2026-11-08

RD-04 staged implementation is delivered. Implementation/evidence commit
`ae800a4` is pushed to main and its exact remote SHA was verified; ledger/handoff
delivery finalization is committed and pushed separately. The layout/admission
contract was committed and pushed first as `2247619`.
Read the [RD-04 report](redesign-rd04-results-2026-11-08.md),
[implementation evidence](evidence/direct-role-states-v1.json),
[coverage evidence](evidence/direct-role-layout-coverage-v1.json),
[versioned contract](direct-role-states-v1.md) and [ledger](redesign-progress.md).

The opt-in profile carries ordered complete content-token states with explicit
lengths/owners, no role pooling/cropping, a shared zero-delta FP32 residual
projection, version-2 metadata, native length heads and 32-state field capacity.
Exact attributes and immutable bindings retain their ABI. Complete all-status
return reservations include echoes and the largest candidate; envelopes remain
256 request /64 return. Legacy renderer fixtures stay exact; live native Session
computation, cache lineage, budget/cancellation and numeric kernels are unchanged.
RD-01..03 implementation/raw evidence and source migration were verified.

**Known future-fit blocker: zero admissible receiver rows**: train0/1500,
E0/130, X0/180. All rows, per-field lengths, observed costs and exclusions remain
in the metadata inventory. No smaller panel or raised allowance was substituted.
Preparation binds the blocker and retains the full proposed exposure schedule.
A direct-profile real-data fit requires an explicit layout revision and full
coverage re-audit first. This initial layout has no measured useful-learning result.

All **61 model-free tests pass** (83.413 s); six packet tests were refreshed
after adding exact selected-attribute return assertions (0.069 s). Four actual
CLI preparations pass with model imports blocked; direct repeated plans are
byte-identical and all execution counters zero. Rust BPE/Jinja metadata-only
coverage keeps all 1,810 rows; 105 historical source token lengths match. No
hidden-state extraction, model execution or optimizer update occurred.

**Six GPU checks remain pending RD-07**, never run/counted passing:
`RD03-GPU-D-INDEPENDENCE`, `RD03-GPU-CACHE-PARITY`,
`RD03-GPU-CONSUMER-GRADIENTS`, `RD04-GPU-IDENTITY-PROJECTION`,
`RD04-GPU-DIRECT-ROUNDTRIP`, `RD04-GPU-DIRECT-GRADIENTS`. RD-04 methods in
`tests/test_direct_roles_gpu.py` use declared engineering latent fixtures and an
existing prompt/parent/reference to check technical mechanisms, not corpus
semantics. Exact 3090 selection and submitted RD-07 opt-in precede model imports.
New-profile execution remains gated; old prepared plans are stale.

No agents, new records/reviews, locked access, fitting, deployment, service or
Operating Table changes. Existing RD-01 coverage/native supervision gaps and
historical profiling-test drift remain. Stop at RD-04; **RD-05 is unstarted**.
Compilation, whitespace, evidence hashes and 261 local Markdown targets pass.

## Previous requested work — RD-03 frozen-W sources, 2026-11-08

RD-03 staged implementation is delivered. Implementation/evidence commit
`2cbfa41` is pushed to `main` and its exact remote SHA was verified; the
[ledger](redesign-progress.md) records commands and delivery finalization. Read the
[RD-03 report](redesign-rd03-results-2026-11-08.md),
[portable evidence](evidence/frozen-w-source-v1.json) and updated
[clean-source contract](clean-test-construction-v1.md).

`CleanEncoder` has explicit `current-d-source-v1` and opt-in
`frozen-w-source-v1` policies. Frozen source forwards skip both adapters directly
under v3 arithmetic, with complete permitted bytes/token mappings and fresh
state; no merge, weight replacement or ownership toggle. Existing pooled role
payloads and D controls remain. Cache keys bind W/buffer content, tokenizer/
template/configuration, arithmetic/software, permitted bytes and full mappings,
without mutable D/receiver hashes. Fresh local sources and original parent audits
are rebuilt; only detached body states are cached. Actual native Session, writer,
binding and cache graphs remain unchanged.

All **50 model-free tests pass** (73.251 s); compilation/whitespace and 165 local
Markdown targets pass. Actual
frozen-pooled/repeated/legacy preparation passes with model imports blocked;
frozen plans are byte-identical, all execution counts zero, and all 1,810 existing
receiver permitted input hashes revalidate. RD-01/02 deliveries and remote main
were verified before edits. Historical evidence is retained; explicit old/new
source migration supports unchanged boundary reuse. Old prepared plans are stale
for this implementation and must not be relabeled current.

**GPU qualification remains pending RD-07**, with implemented assertions:
`RD03-GPU-D-INDEPENDENCE`, `RD03-GPU-CACHE-PARITY`,
`RD03-GPU-CONSUMER-GRADIENTS` in `tests/test_frozen_source_gpu.py`.
No GPU test was run or counted as passing. Execution needs explicit RD-07 opt-in
and exact 3090 UUID before model imports. Fixtures use an existing assessed Noul
record/parent/reference and no optimizer; broader semantic qualification remains
separate. New profiles remain blocked for normal execution pending later slices.

No model execution, source extraction, fitting, new data/reviews, agents, locked
access, deployment, service or Operating Table changes. Existing RD-01 data/native
supervision gaps and historical profiling-test drift remain. **Next unfinished
implementation slice: RD-04**; it may proceed when submitted while these exact
GPU checks remain pending RD-07. Stop after RD-03 in this task.

## Previous requested work — RD-02 receiver preparation, 2026-11-08

RD-02 is delivered. Implementation/evidence commit `254f526` is pushed to `main`
and its exact remote SHA was verified; the [ledger](redesign-progress.md) records
commands and the delivery finalization. Read the
[RD-02 report](redesign-rd02-results-2026-11-08.md) and
[portable evidence](evidence/receiver-preparation-v1.json). The existing probe
runner now requires explicit preparation/execution, provides named legacy,
frozen-W pooled and direct-role profiles, and binds panels, permitted views,
frozen dependencies, initial checkpoint/moments/RNG, owners, objectives and
scheduled exposures in deterministic plans. New profiles remain unimplemented/
unqualified and reject execution; no model was run.

Bound legacy execution/resume uses the existing loop, Window and transactional
checkpoint code. Model-free guards reject stale releases/profiles/caches/
checkpoints, wrong GPU/regime, ambiguous actions and incompatible cursors/owners.
Failed attempts stay scheduled; actual updates remain separate. Preparation
receipts have zero encodings, Sessions, prefills, positions and optimizer updates.
Existing host timing/call instrumentation distinguishes actual execution from plans.

All three actual CLI preparations pass with model libraries blocked; repeated
legacy output is byte-identical. The proposed comparison arms share the same
1,500 records /four passes /6,000 presentations /750 windows, initial state,
objectives and existing owners, with separate profile/checkpoint identities.
The RD-04 projection owner remains explicitly pending, without fabricated tensors.
40 model-free tests pass (62.390 s); final guard changes are covered by a fresh
12-test pass (47.402 s), reusing the other 28 unchanged results. Compilation and
whitespace and 148 local Markdown link targets pass. No optimizer/tensor algorithm
was changed or exercised.

The broader initial run exposed a pre-existing historical profiling-test setup
failure: all 130 saved E receipt view hashes use the historical schema and reject
current clean-view-v2; original row hashes still match. Its unchanged source and
raw failure/proof are recorded. Required Recorder tests pass. Do not treat that
older diagnostic fixture as passing or substitute its stale views into new plans.

RD-01 implementation/raw evidence and pushed commits were verified and valid
23-test results reused. All RD-01 missing operator/range/native binding labels
remain limits on later work. No new model execution, training, data/review,
agents, locked access, deployment or service change occurred. No required RD-02
check remains pending. **Next unfinished slice: RD-03**; stop at RD-02 here.

## Previous requested work — RD-01 metadata coverage, 2026-11-08

RD-01 is delivered. Implementation/evidence commit `800c100` is pushed to `main`
and its exact remote SHA was verified; the [progress ledger](redesign-progress.md)
records commands and the delivery finalization. Read the
[RD-01 report](redesign-rd01-results-2026-11-08.md) and
[portable manifest](evidence/redesign-coverage-v1.json). The deterministic audit
resolves 8,000 training +1,000 development selected records, accounts for all
1,500 receiver training rows, fixes complete E130/X180 panels and 50 assessed
Choice retention cases, and inventories conditional native targets. Existing
semantic/architecture audit receipts are reused, with original raw hashes checked.

Known limits for later work: no assessed receiver training coverage for negated
AND/OR, remaining/half-completion or denominator 200; no assessed Score E/X cohort
meets the declared familiar numeric-range condition. Unclassified prose remains
unclassified. The native inventory records 224 interleaved calls without their own
Reason, 373 interleaved records with ambiguous source occurrences, and 44 writer/
139 interleaved Choice targets that cannot satisfy the current unique-selection
continuation guard. Supplied-return reader annotations do not qualify actual-return
use; native binding/continuation conditions remain enforced. These are completed
audit findings, not a waiver to train or evidence of learned behavior.

All **23 model-free tests pass** (60.371 s); compilation/whitespace and 141 local
Markdown targets pass. Acceptance covers repeat determinism with model imports blocked,
complete accounting, split overlap, ambiguity, stale identities, hidden-label
invariance and report denominators. Exact command/results and dependency/artifact
hashes are in the evidence. No model/GPU workload, training, new data/review,
agents, locked access, deployment or service change occurred. There are no deferred
RD-01 GPU checks. **Next unfinished slice: RD-02**, runner preparation and resumable
identities, requiring its own submitted prompt. Stop after RD-01 in this task.

## Previous requested work — learning redesign plan and queue preparation

Subsequent request: prepare one implementation prompt per slice, so later queue
items recover unfinished predecessors instead of skipping them. The
[copyable prompts and required protocol](redesign-queue-prompts-2026-11-08.md)
and [progress ledger](redesign-progress.md) are now present. Each submitted task
must verify code/evidence/push status, recover the earliest incomplete delivery,
reuse an existing active run/checkpoint and preserve actual qualification gates.
Deferred GPU checks from RD-03..06 are resolved by RD-07; a delivered but failed
RD-08 cannot unlock RD-09. AGENTS.md points to this task-specific protocol.
Preparing prompts did not authorize any execution. RD-01..07 prompts are
non-training; submitting RD-08/09 explicitly authorizes their stated finite fits.
At prompt preparation all redesign slices remained pending. No agents, data/model work or training ran.
Prompt-preparation validation: nine ordered prompts/nine copyable blocks, all
referencing the recovery protocol; 200 local links across the four affected
Markdown files resolve; whitespace checks pass. No queue manager was added.

The user requested a concrete implementation plan reusing working infrastructure
and existing data, with W frozen and latent-only communication; no agents or
training yet. The [plan](learning-redesign-plan-2026-11-08.md) records RD-01..09,
code/contract owners, per-slice acceptance and evidence, and explicit data and
native-supervision blockers. It supersedes older next-slice suggestions as the
proposed work order, without changing current runtime or historical results.

Completed this turn: implementation/report inspection and the plan only. All 197
local Markdown link targets across three files resolve; whitespace checks pass.
No model/GPU execution, training,
new data/reviews, agents or service changes. No redesign implementation ran during that plan-only turn.

**At plan preparation, next unfinished implementation slice: RD-01**, the metadata-only coverage,
comparison-panel and native-supervision manifest. RD-02..06 add runner/profile,
stable component features, richer latent roles, joint objectives/binding and
native wiring; RD-07 performs targeted no-update 3090 qualification when requested.
RD-08/09 describe future receiver/native learning and require subsequent training
authorization. Current learned capability remains exactly as reported below.

## User direction: MVP gates — 2026-11-06

Keep the following tests and their recorded results, but reconsider their role
as mandatory blockers for MVP progress. Do not spend further work satisfying a
diagnostic merely because the current pipeline makes it a prerequisite; first
establish how it demonstrates a necessary product behavior.

- **Exact text reconstruction from internal vectors:** retain as a diagnostic.
  Identical wording has not been established as essential to the MVP. The
  architecture asks for semantic recovery; the current evaluator uses exact
  canonical reconstruction as a stricter sufficient criterion. Evaluate whether
  relevant meaning survives and is correctly used downstream. Preserve exact
  values, identities and bindings where the task requires them.
- **Broad domain/status coverage quotas:** retain as coverage reporting and
  broader qualification goals; justify any quota made a prerequisite for an MVP
  demonstration. This does not remove supported operations or status handling.
- **Tight thinking-budget prediction targets:** retain as quality measurements;
  justify blocking the MVP on their precision. Preserve the budget mechanism,
  numerical contract, separate limits and accounting.
- **Passing two checkpoints exactly 100 updates apart:** retain as an
  experimental stability protocol; justify requiring that specific spacing for
  MVP acceptance rather than treating it as an inherent product requirement.

The intended MVP still includes **Noul, Choice and Score**, learned invocation,
internal vector communication, and correct use of actual results in a normal
answer. Assess that complete behavior on examples outside training, including
meaningful input changes. Correctness, valid binding, numerical stability,
single-prefill/cache behavior and cancellation remain necessary.

This records a requested change in acceptance policy, not an implemented gate
bypass or a declaration that the model works. Before changing enforced gates,
document their product justification or an appropriate replacement check.
Existing failed reports remain failures under their original protocols. This
documentation change does not interrupt or extend an active experiment, launch
training, promote weights, or authorize deployment.

## Current checkpoint — matched quantity/operator receiver comparison complete

The user authorized the matched receiver-learning comparison on existing data,
3090 only, W/reference frozen, development held out, no agents or new data.
Both fits and all measurements are complete, with no promotion. Implementation
commit `57a3cf8`; read the [fixed protocol](quantity-receiver-learning-v1.md),
[results](quantity-receiver-learning-results-2026-11-08.md) and
[portable evidence](evidence/quantity-receiver-learning-v1.json).

**Matched scope:** literal UTF8 versus INT64, all 254 assessed Score training rows
/96 families, four passes /1,016 presentations /127 updates per arm, A4 positions
189..315. Identical initial receiver/moments/RNG, same source-bound count fields,
eight extra request positions, zero failed training exposures. Existing Score
loss plus auxiliary count regression through D/packet representations; count
predictions never enter Score/runtime. Status/Reason/Noul/Choice losses absent;
their private heads receive no updates, but shared representations change.

**Training values learn:** Score mean interval distance literal
**0.4048570293→0.0405827365**, INT64 **0.4036024587→0.0403374007**. Exact head hits
38/254 and 57/254 are all endpoints; interior errors improve but remain 0/84
exact. Required/completed count MAE **5.1608/8.0563** versus **13.1336/6.4100**;
both counts nearest-integer correct only **2/254 versus 0/254**. Correct Assessed
status collapses **253→89** and **254→75**, so head learning is not usable behavior.

**Held-out transfer fails:** only after both fits, each fixed checkpoint runs all
130 E source-only +180 applicable X cases. X Score mean error **0.3374772806 /
0.3383736139**, worse than frozen literal/INT64 **0.2346320546 /0.2348618014**.
Both remain **0/40 exact head/delivered Score, 0/20 both-correct pairs**; correct
Score status **7/40 /12/40**, down from 40/40. Reversal directions **5/10 both**;
half-scale **5/10 /1/10**. Literal ordinary-completion error improves modestly
**0.2651124950→0.2383401253**; INT64 worsens **0.2677185292→0.3289889189**.
Required/completed X count MAE **98.5304/29.3868 /105.5418/48.3076**; both integers
recovered **0/40** in both arms. No count clipping was added.

Training has denominators 20/100, counts<=100, ordinary completion only. X always
requires 200 and adds new units/reference meanings, reversal and half-scale.
No assessed receiver training examples for those operators exist. INT64's 128
bit is unseen during training. These are concurrent transfer shifts, not proof
of a unique cause or unlearnability; no missing examples/annotations were invented.

**Retention is mixed:** Choice X stays **40/40** in both arms; assessed Choice E
changes **7/10 parent→10/10 literal /4/10 INT64**. Assessed Noul E+X was already
**0/80** and remains zero; correct abstentions increase. All-operation behavioral
passes **83/310 parent→99/310 /117/310** must not be mistaken for assessed Noul/
Score competence. All **620/620** new development sessions pass actual latent
transport/binding/cache/accounting, with no runtime failures. Historical controls
were reused with explicit row/view bindings, not rerun. Locked/reader cases remain
excluded; no development fitting, stopping, selection or promotion.

**Validation:** 38 CPU tests pass in 15.203 s; compilation/whitespace pass. Receipt
audit PASS over 88 archived source files, all exposure/atom/optimizer/checkpoint
receipts and recomputed summaries. Initial non-attribute wire/source/reference/
ledgers match; each arm's first eight training/no-grad forwards match exactly.
First-update model/moments/all RNG roundtrips exact. Same-graph separate/combined
VJP max relative RMS **3.7994001e-6 /2.5423570e-6**, minimum cosine
**0.9999999999928306 /0.9999999999968955**; nonzero quantity-only D/atom/readout
gradients. All 320 W tensors, physical tying, reference/R/nonreceiver owners and
grad=None pass. Unchanged v3 qualification reused; no expensive matrix rerun.

Exactly **3,668 Sessions /395,482 positions**: 2,032 training, 1,016 training
measurements, 620 development. One prefill/lineage each; max 209 positions.
Offline **11,140 source encodings /275,608 tokens**, **954 frozen reference
encodings /25,982 cache hits**. First/fourth exposures of all 254 records per arm
have identical reference payloads and changed D-dependent source states.
Measured run region **7,666.574491 s**, peak CUDA **3,622,856,704 bytes**, exit 0.
Raw: `data/quantity-learning-20261107/run/`; audit/validation/exit/GPU evidence
in parent. Only original PID2094 /3,852 MiB remains. No failed GPU model runs,
3060, agents, new records/reviews, native/general training, deployment, live
service or Operating Table changes.

**Next unfinished slice:** preserve/recover status and assessed behavior while
retaining value learning, then qualify count-range/operator transfer. The tested
objective omitted status/retention losses; its failure does not establish failure
of a joint or isolated repair. Missing operator/range training coverage is a
separate limitation. No additional fit or data generation is authorized by this
result. Reference fidelity, Reason meaning, learned invocation and complete native
use remain unqualified. Earlier next-slice entries below are historical.

## Previous checkpoint — direction hash fixed; numeric-access comparison complete

The user authorized the direction-hash repair and one matched numeric-access
comparison on the 3090, existing data, frozen W/reference, no agents, new data
or training. Both slices are complete. Read the
[protocol](numeric-access-comparison-v1.md),
[results](numeric-access-results-2026-11-07.md) and
[portable evidence](evidence/numeric-access-comparison-v1.json).

**Hash fix (`dcc3206`):** `clean-view-v2` explicitly hashes rubric direction;
direction-only changes differ, invalid directions fail, and request construction
rejects a direction different from the hashed view. Old receipts stay unchanged.
Diagnostic reuse explicitly maps old/new hashes with full row/reference/source
bindings; no implicit fallback. Historical launchers may reject source/hash drift
and need their recorded snapshots. Model/packet arithmetic is unchanged.

**Comparison (`dbd94c0`):** all 40 X Score cases /20 pairs. Original source-state
baseline reused; two new matched arms add source-literal UTF8 versus parsed INT64
attributes for required/completed counts. Both add eight request positions,
preserve all original evidence and use identical source-bound Key substrings.
Two derived Key encodings copy existing wording; no released rows/labels/reviews
are created. No ratios, complements or desired outputs enter the input.

**No Score improvement:** baseline /literal /INT64 mean interval distance
**0.2338902032 /0.2346320546 /0.2348618014**. All are **0/40 exact head and delivered
Score**, with 40/40 correct status and transport. INT64 lowers error versus
literal on 20 cases and raises it on 20. All paths retain **5/10 reversal and
0/10 half-scale correct directions; 0/20 both-exact pairs**. No promotion.

Four source-only anchors (Noul, Choice, Score higher/lower) exactly reproduce
saved wire, h, outputs and full ledgers. All 40 matched cases have exact
non-attribute wire rows, original source/reference tensors, source literal atoms
and request event lengths. All matched h differ; median relative L2 **0.091647**,
which shows sensitivity, not correct numeric use. Noul/Choice anchors are not
fresh full-operation retention qualification.

**35 CPU tests pass in 0.236 s**, compilation/whitespace pass. Independent CPU
receipt audit passes and verifies 85 executed-source files. All 320 W tensors,
physical tying, receiver/reference/cache identities and grad=None pass on 3090.
No failed GPU attempts; unchanged v3 numeric/gradient matrices reused.

Exactly **84 new Sessions /10,950 positions** (498 anchors, 5,226 per arm), one
prefill/cache each, exact accounting, no invocation text. Offline: **31 source
encodings /656 tokens +222 cache hits; two reference encodings /53 tokens +657
cache hits**. Frozen source/ref cache reuse is process-local and reported
separately from Sessions. Model-run measured region **78.289277 s**, peak CUDA
**3,622,856,704 bytes**, excludes metadata prep/final serialization. Raw:
`data/numeric-access-20261107/run/`, receipt audit/tests metadata in parent.
Only original PID2094 /3,852 MiB remains. No 3060, agents, data/reviews, training,
locked-set access, deployment, service or Operating Table changes.

**Next unfinished slice:** test learning of quantity/operator use with matched
training-side exposure using existing training cases, keeping development held
out. Distinguish raw-count access from the requested operation. This frozen
comparison shows no immediate format benefit; it cannot prove typed values
unlearnable, particularly because receiver training did not use attributes.
No further learning or format search is authorized or launched by this result.
Reference fidelity, Noul transfer, Reason meaning and complete native use remain
unqualified. Earlier next-slice entries below are historical.

## Previous checkpoint — training-to-X/meaning audit complete; no new model work

The user authorized a read-only audit using existing data/saved features, frozen
W/reference, no agents, new data or training. Read the
[findings](training-x-semantic-audit-2026-11-07.md),
[scope](training-x-semantic-audit-v1.md) and
[portable evidence](evidence/training-x-semantic-audit-v1.json).
**Completed, no promotion or new model execution.**

The training task is narrower than X: assessed Noul is **256 positive AND** only;
X adds ten negations and ten OR cases. Assessed Score training is **254 ordinary
completion fractions**, source denominators 20/100, with **170 endpoint targets**.
All 40 X Score denominators are 200, with 40 distinct nonendpoint/nonhalf targets;
ten reverse the rubric and ten halve its scale. Training has 96 unit strings, X
one. Same six domain labels do not imply matched task/numeric coverage.

**Historical correction:** the original 500 reference rows contain 50 meanings,
but the current pass-12 reference trained on **3,376 distinct views**, including
all **6,499 receiver-training role requests /1,643 unique annotations**. X's
Question/Target/Alternative/Conditions canonical inputs have zero exact matches
in that catalogue; only 70/640 role requests match common anchors/dimension.
Canonical novelty does not prove novel meaning or an invalid evaluation. It
does confound attributing train/X differences only to the final head.

Existing direct-readout subgroup results: ordinary Score distance improves
**0.269677→0.228196 /20**, but still 0 exact; remaining-fraction error
**0.194187→0.478984 /10**, half-completion **0.202020→0.411399 /10**. All tolerances
unchanged. Direct positive-AND Noul 37/50 is below the 40-case majority reference;
balanced accuracy 57.5%, AUC 0.665. Both sides correct on just **1/10 negation and
3/10 OR pairs**. The parent's 10/10 negative-side success is its all-false bias,
not negation proof.

Saved boundaries: all 70 changed-input pairs change h; 20 excluded-input twins
retain identical wire/h/q. This establishes sensitivity/invariance, not correct
meaning. Choice's 10 renamed/reordered pairs correctly select A→X, useful limited
evidence. Tie pairs all retain A, which is accepted on both sides. Category 10 returns
are excluded from receiver input, so its invariance does not test contradiction
understanding. Native reader category 9 remains excluded; Reason meaning unqualified.

**Concrete input/audit gaps:** the component builders do not populate numeric
attributes; all 180 saved request role sequences have zero ATTRIBUTE_DESC/INT64
positions. Counts reach the receiver through source states; exact bytes/bindings
outside the model do not establish numeric access. Rubric direction is passed to
the renderer separately from `CleanView.materialize`'s hash. That hash alone is
not full packet identity; existing full-row/wire evidence remains valid and no
existing hash group has conflicting directions. No production fix was made.

Validation: **5 model-free tests pass in 0.128302 s**, compilation/whitespace pass.
Audit binds **1,003 training features, 180 X captures, 620 predictions, 90 pairs**,
checks **3,474 source byte-coordinate entries /640 X reference entries**, and
confirms zero train/X family/view overlap. Training receipts have source tensor
hashes and view hashes, not individual reference tensor hashes; the initial
CPU checker assumption was corrected. The initial RF-only exposure report is
retained and superseded by the final catalogue audit. No new GPU attempt.
Raw: `data/training-x-audit-20261107/`; checker:
`scripts/audit_latent_semantics.py`. Tensor artifacts are hashed as opaque bytes,
never executed on CPU. **Zero new Sessions/prefills/positions/encodings/fits**.
No W/reference loading, agents, new reviews/data, locked access or service changes.

**Next unfinished slice:** close the rubric-direction input-identity gap, then
scope a matched source-quantity-access diagnostic using existing bound counts
versus source-state-only inputs, with question/rubric/receiver/scoring fixed.
Keep X for development, never training selection. No follow-up experiment,
input-path modification or further fit is launched by this audit. Numeric access,
reference interpretation, status/value integration and native use are still
unresolved; do not attribute the failures to a single layer or automatically
scale training. Earlier next-slice entries below are historical.

## Previous checkpoint — direct-readout diagnostic complete; weak development transfer

The user authorized a simple direct-readout diagnostic on the 3090 using saved
features, frozen W/reference, separate training/development, no agents or new
data. Read the [protocol](direct-latent-readout-v1.md),
[actual results](direct-latent-readout-results-2026-11-07.md) and
[portable evidence](evidence/direct-latent-readout-v1.json).
**Completed; training signal is decodable, but held-out transfer remains weak.**
No production installation, additional fit or promotion.

Two standalone affine ridge probes directly read all 1,024 h coordinates with
an intercept, bypassing fixed heads/prototypes and the residual bridge. Training
uses **256 assessed Noul /254 assessed Score**, **96 families per operation**, from
the saved 1,003-feature corpus. Three-fold family-grouped training-only validation
chooses lambda among 1e-9/1e-6/0.001/1; both choose **0.001**. Completed **24 CV
solves +two final fits**, entirely CUDA FP64, with means/scales fitted inside each
training subset. Both probes/export are finalized before opening development
tensors. No development-based calibration/selection, iterations or time cutoff.

Development coverage is the saved **X** h: **70 assessed Noul /40 assessed Score**,
20 families per operation. E h was not saved and was not regenerated; do not
compare these directly with the earlier 80/50 E+X denominators. No train/development
family or permitted-input overlap, all selected training inputs unique. No
non-assessed value labels invented, locked data unopened. Exact scalar ABI,
Noul threshold and Score intervals retained for diagnostic metrics.

**Direct development:** Noul **48/70**, Brier **0.226305**, AUC **0.558**. This
imbalanced subset has 50 true/20 false labels: **43/50 true, 5/20 false** correct,
balanced accuracy **55.5%**, and 48 is below the **50-case majority-label reference**.
Do not sell the headline gain from prior 20/70 as strong transfer. Prior X
parent/joint/value-only AUCs were 0.600/0.621/0.618. Score **0/40 exact**, mean
distance **0.336694**, worse than parent **0.233890**, joint **0.250675**, value-only
**0.254892**. Error worsens in both directions: parent→direct higher
**0.247125 →0.289264 /30**, lower **0.194187 →0.478984 /10**.

Of 90 intervention pairs, 50 have two assessed scalar targets. Direct gets both
correct in **12/50**: four of 20 changed-truth Noul pairs and eight of ten
unchanged-target Noul pairs; Score zero of 20. Direction on changed targets
**20/40**, parent22/joint21/value-only18. These are scalar checks, **not** full
behavior/transport successes. Choice/status/Reason/native delivery are not tested
or modified; prior receipts remain historical, not fresh qualification.

Training fit /grouped OOF: Noul **197/256 /177/256**, balanced accuracy
**75.14% /66.21%**, Brier **0.154134 /0.166796**, AUC **0.866071 /0.785038**.
Score mean distance **0.263342 /0.310479**, exact hits **1/254 /8/254**.
Weak regularization gives very poor OOF raw MSE (Noul12.224/3.313,
Score3.103/2.036 at 1e-9/1e-6), versus selected0.167704/0.143547. This is evidence
for regularization, not a reason to select using development. Training or OOF
success does not establish reference fidelity or complete learned use.

**26 CPU tests pass in 14.118 s**, compilation/whitespace checks pass. CUDA analytic
affine/primal-versus-dual checks pass (max absolute difference **1.387778781e-17**);
constant features rejected. Maximum equation residual across all fitted systems
**1.547262027e-11**, versus declared1e-8. Exact export/reload predictions on
**510 training +110 development** records, unchanged feature tensor/file hashes
and parent manifest. Independent CPU receipt audit recomputes CV selection,
family splits, all scalar/pair/direction metrics, provenance and accounting.
Production numeric qualification reused; new diagnostic arithmetic is explicitly
FP64 with FP32 scalar reporting, not a change to v3 or a new backpropagation path.

Raw: `data/direct-readout-20261107/run/`, logs/audit/test receipts in parent;
`prepared/` is model-free preflight. Export `readouts.safetensors` SHA256
`a82f043469c837417aaab37ce78ef32d273d5ea3b09b6430d53f5094b098b969`, with
`export.json` marking production-incompatible/no promotion. No failed GPU attempt.
GPU fit/check/evaluation wall **2.247410 s**, peak CUDA **46,622,720 bytes**;
excludes release preparation/imports, not a matched transformer benchmark.
**Zero new Sessions, prefills, causal positions, source/reference encodings**.
W/reference/codec model instances never loaded. Only original PID2094 (3852MiB)
remains after exit. No 3060, agents, new annotations/reviews, native training,
deployment or Operating Table/service changes.

**Next unfinished slice:** audit the existing training-to-X task/input mismatch
and the semantic/numeric information preserved by the latent path before another
receiver fit. Reuse the caches and these diagnostics. The affine result neither
proves h is uninformative nor rules out nonlinear decoding; it does not isolate
architecture, optimization and distribution differences. Do not automatically
scale training, repeat a loss sweep, promote probes or generate examples.
Reference meaning, status/value integration, Reason meaning, learned invocation
and full native use remain unqualified. Further model experiments need new scope.
Earlier “next slice” entries below are historical.

## Previous checkpoint — matched value-only comparison complete; no value transfer gain

The user authorized one matched value-only comparison on the 3090 using the saved
1,003 features, exact initialization/exposure, frozen W/reference and unchanged
Choice, no agents or new data. Read the
[protocol](receiver-value-only-comparison-v1.md),
[actual results](receiver-value-only-results-2026-11-07.md) and
[portable evidence](evidence/receiver-value-only-comparison-v1.json).
**The comparison completed; removing status loss did not fix development values.**
No promotion, extra fit or schedule extension.

The prior joint repair is the matched control. New checkpoint-0000 **weights,
optimizer and RNG files are byte-identical** to its saved initialization. All
1,003 initial heads/losses/metrics and feature tensor bytes match. All original
model/loss/optimizer/ownership/schedule implementations match archived sources;
the launcher adds an explicit `--value-only` mode. The only objective change is
removing status before eligible-count accumulation. Forward routing is unchanged:
status/scalars share the residual, Reason/structure use original h, Choice bypasses.
All existing W, R/D, codec/reference parameters stay frozen; only the same 133,184
new residual parameters train. Exact values, bindings, thresholds and numeric
tolerances are unchanged.

Exactly **2,000 scheduled windows /16,000 visits**, same order and 955×16/48×15
per-record exposure counts. Scalar eligibility exactly **Noul 4,087 /Score 4,049**,
status zero. Ten predetermined windows have no eligible scalar target and correctly
skip AdamW: **1,990 updates**. Inactive branches get no gradient/moment/decay update
under the existing rules; no replacement examples or attempts to equalize updates.
All **1,003 features reused**, zero new bulk extraction.

**Development joint control → value-only:** behavior **123 →78 /310**, correct
status **182 →129 /310**, complete intervention pairs **30 →20 /90**. Noul truth
stays **25/80**, delivered success **9 →4 /80**, Brier **0.299799 →0.302809**,
descriptive AUC **0.675636 →0.638545**. All assessed development Noul predictions
remain false: 0/55 true positives, 25/25 true negatives, balanced accuracy 0.5.
Score exact delivered success stays **0/50**; interval distance worsens
**0.285469 →0.288103**, in both scale-direction groups. **All 90 Choice cases
exactly retain both controls' outputs and complete ledgers**, including 47/50
assessed successes. **310/310 transports**, zero execution failures.

Training joint → value-only: Noul truth **168 →169 /256** (eight true positives
gained, seven true negatives lost), BCE **0.626934 →0.507407**, Brier
**0.217993 →0.174319**, AUC **0.767519 →0.763934**; status **420 →358 /501**.
Score hits **0/254 unchanged**, mean distance **0.393886 →0.393377**, status
**502 →417 /502**. Better Noul training probability loss does not transfer;
status competition alone is not a sufficient fix in this configuration. No
impossibility/convergence claim or diagnosis of the remaining bottleneck.

**20 CPU tests pass in 13.860 s**, byte compilation/whitespace checks pass.
Targeted GPU full/minimal scalar loss/VJP equality, selected-branch nonzero gradients
and other-branch gradient isolation pass. Prior unchanged residual FP64/transformer
numeric qualifications reused. Exact first-update perturbed-state resume, four
fresh final h/head/ledger checks, both 320-tensor W/physical-tying checks, frozen
receiver/reference fingerprints, feature hashes and gradient ownership pass.
Independent CPU audit recomputes matched exposure, active owners, empty windows,
eligibility, source audits, summaries/grades, accounting and Choice retention.

Raw: `data/receiver-value-only-20261107/run/`, logs/audit/test receipts in parent;
`prepared/` is model-free preflight. No failed/retried GPU attempt. Final repair
`run/checkpoint-2000` manifest
`7d719824ec545cd9f12489cf61a9c231bc79085111a49501d65b591e82b32ffc`, fingerprint
`f98e0f742d105e553666e02a045ae1a251f0890a1b3145e3c0e24632cfba7245`.
It requires the pinned parent and explicit experimental installation.

New **4 fresh-check /310 development Sessions**, **399 /38,211 positions**,
total **314 /38,610**. One prefill/cache lineage each; no held positions or
invocation text. Cached fit/measurement adds zero positions/prefills. Separate
offline source/reference tokens **26,551 /16,316**, 483 reference encodings.
Model-run **479.425140 s**, peak CUDA **3,622,856,704 bytes**; not a matched
throughput benchmark. After exit only existing PID 2094 (3852 MiB) remains.
No 3060, agents, new data/reviews, locked-data access, reference/native training,
deployment or Operating Table/service changes.

**Next unfinished slice:** distinguish remaining scalar information/transfer
limitations from readout optimization using the saved feature corpus. A separately
specified direct-readout diagnostic is a candidate next comparison; additional
fitting needs new authorization. Do not repeat this status-loss comparison or
bulk extraction by default. Reference meaning, Reason meaning, learned invocation
and full native use remain unqualified. Earlier “next slice” entries are historical.

## Previous checkpoint — slice 4 complete; Choice retained, value repair unsuccessful

The user authorized one controlled Noul/Score receiver repair on the 3090,
existing data, frozen W/reference, no agents or new data. The
[protocol](receiver-residual-repair-v1.md),
[actual results](receiver-residual-repair-results-2026-11-07.md) and
[portable evidence](evidence/receiver-residual-repair-v1.json) record its scope.
**The finite experiment completed; its value-learning acceptance condition failed.**
No weights were promoted and no additional fit/continuation was run.

One opt-in 1024→32→1024 residual branch per Noul/Score operation, initially exact
identity, adapts only status/scalar readouts. All existing W, R/D, codec heads,
pooling/reference weights remain frozen; original Reason/structure and Choice
bypass the repair. Score retains its actual BF16-effective LM-head row. Production
defaults, Noul threshold, exact intervals, startup budget/S-curve and numeric
tolerances are unchanged. **133,184 new parameters**, fresh A4 component schedule,
**2,000 updates /16,000 visits**, all **1,003 existing Noul/Score receiver records**.
955 records have 16 visits, 48 have 15. Eligibility: status 16,000, Noul 4,087,
Score 4,049. No failed exposure or skipped update; development never enters fit.

**Development parent → repair:** total behavior **83 →123 /310**, status
**130 →182 /310**, full intervention pairs **23 →30 /90**. Noul assessed delivered
success **0 →9 /80**, but independent truth stays **25/80** and Brier worsens
**0.284675 →0.299799**. All assessed Noul probabilities remain below 0.5.
Descriptive Noul AUC improves **0.600727 →0.675636**, with overlapping positive
and negative ranges; this is a ranking signal, not solved truth/calibration.
Score exact delivered success stays **0/50**; raw interval hits **1 →0 /50**,
mean distance worsens **0.271739 →0.285469**, in both direction groups.
**All 90 Choice cases exactly retained**, including complete canonical ledgers;
assessed success remains **47/50**. **310/310 transports**, zero execution failures.
Status gains do not certify better value reasoning or Reason meaning.

Full-corpus training: Noul truth **168/256 unchanged**, Brier
**0.232134 →0.217993**, AUC **0.359037 →0.767519**; status **395 →420 /501**.
Score status **411 →502 /502**, but exact hits stay **0/254** and mean distance
only **0.405995 →0.393886**. The package comparison does not isolate capacity,
fresh optimizer/exposure, loss interference or reference fidelity as the cause.

**19 CPU tests pass in 14.162 s**; byte compilation and whitespace checks pass.
CUDA FP64 residual forward/VJP checks pass: maximum relative RMS
**2.712716892e-8 /1.867542228e-7**, versus unchanged 1e-5 tolerance. Minimal/full
head outputs and gradients agree exactly, with connected residual gradients.
All **1,003 initial outputs** exactly match parent receipts. Four fresh final
training Sessions match cached h, repaired heads and complete ledgers. Deliberate
weight/moment/RNG perturbation followed by first-update resume restores exactly.
Both 320-tensor W checks/physical tying, frozen receiver/reference fingerprints,
feature fingerprints and gradient ownership pass. Independent CPU receipt audit
recomputes exposures, masks, metrics, grades, provenance, accounting and retention.
Completed transformer numeric matrices are reused; their implementation is unchanged.

Raw: `data/receiver-repair-20261107/run/`, audit/log/test evidence in its parent.
Final repair bundle `run/checkpoint-2000`, manifest
`abc321bb1535656a931e7e5926e01def2417a3d90931235ae996e0a43feb6a8f`, requires
the pinned parent-0188 checkpoint; it is not a standalone or default model.
Repair fingerprint
`12262d41a02ef9d148a68f0888f038f2a4c67fa56c598121417774d61ff0ceae`.

Reused 24 training features. New **979 extraction /4 fresh checks /310 development
Sessions**, **93,964 /399 /38,319 positions**: total **1,293 /132,682**. One
prefill/cache lineage each, no held positions/invocation text. Cached fit visits
add zero positions/prefills. Offline source/reference tokens **102,528 /42,925**,
1,202 reference encodings, separately accounted. Model-run **1,619.862349 s**,
peak CUDA **3,622,856,704 bytes**. No failed/retried GPU attempt. After exit only
original service PID 2094 (3852 MiB) remains. No 3060, agents, new data/reviews,
locked-data access, native training, deployment or Operating Table changes.

**Next unfinished slice:** value learning/calibration that transfers while
preserving Choice. A discriminating proposed comparison is a value-only residual
fit on the **saved 1,003-feature cache**, matched in initialization, corpus and
exposure, to isolate status/value interference. That explanation is not proven;
do not present it as the fix. Further fitting needs new authorization. Do not
repeat feature extraction or automatically extend this failed repair. Reference
meaning, learned invocation, Reason meaning and complete native use remain
unqualified. Earlier “next slice” sections below are historical checkpoints.

## Previous checkpoint — slice 3 complete; simpler frozen-input candidate rejected

The user authorized one simpler latent-input comparison on the 3090, existing
data, frozen W/reference, no agents or training. Read the
[protocol](simple-latent-comparison-v1.md),
[actual results](simple-latent-results-2026-11-07.md), and
[portable evidence](evidence/simple-latent-comparison-v1.json).

Candidate **ordered-full-width-means-v1** replaces learned reference pooling
with ordered, disjoint means of the same frozen-W token states and existing
FP32 normalization. All tokens contribute exactly once. This bypasses the learned
256-wide pooling machinery, keeping 1024 width and existing slot counts; it is
**not an uncompressed-input/increased-capacity test**. The existing prepared-draft
R tick sees the new semantic records too. No fixed hidden prefix at D is claimed.

Reuse all **160** captured pre-pool/pooled role tensors and **180 X /90-pair**
control receipts from slice 2. **Three fresh control anchors** exactly match
saved outputs, full canonical ledgers, wire and h. All **180 candidate cases**
execute once with zero execution failures. W, D/R, codec, heads and reference
remain frozen; parent-0188 receiver/reference identities are unchanged. Candidate
provenance is explicitly a new basis, **not compatible reference targets**.

**Results, control →candidate:** behavior **49 →31 /180**; status **89 →149 /180**;
complete pairs **23 →0 /90**. Choice **40 →11 /40**, Score **0 →0 /40** and mean
Score distance **0.233890 →0.241228**. Noul assessed delivered success **0 →20 /70**,
but independent truth accuracy remains **20/70** and Brier worsens
**0.289148 →0.293918**. The candidate predicts false on all 100 Noul cases
(probabilities 0.413115..0.418366) and marks all Assessed, including all 30
Insufficient cases. Status gains are not better truth reasoning. Candidate gains
20 Noul cases, loses nine Noul and 29 Choice cases, retains 11 successes.

All **180/180 transports** pass. Source byte/offset/tensor lineage, permitted
views, request framing/length, startup budget and every nonsemantic wire row
match the control exactly. All **20 identical-input pairs** retain identical
wire/h. Every candidate role payload passes GPU FP64-reference comparison:
maximum relative RMS **8.288918964e−8** versus unchanged 1e−5 tolerance.

**10 CPU tests pass in 0.189 s**; all 320 W tensors/physical tying, frozen
fingerprints/cache, grad=None, byte compilation and whitespace checks pass.
Independent CPU audit recomputes receipts/grades/accounting, source snapshots and
160 saved payload byte hashes. No production forward/backward implementation or
numeric tolerance changed; completed matrices reused.

New execution: candidate **180 Sessions /20,856 positions**, anchors **3 /374**,
total **183 /21,230**. One prefill/lineage each, exact accounting, no held positions
or invocation text. Offline source tokens **14,212 +253** are separate. Zero new
reference encodings/pooler calls. Model-run wall **195.108165 s**, peak CUDA
**3,622,856,704 bytes**; not a matched throughput benchmark.

Raw: `data/simple-latent-20261107/run/`, with `audit.py`, logs and portable hashes
in its parent. No failed/retried GPU attempt. Only original service PID 2094
(3852 MiB) remains after exit. No 3060 model work, agents, data generation/reviews,
locked-data access, training, promotion, service/deployment/Operating Table changes.

**Decision:** retain learned pooling as the control; reject this fixed-capacity
replacement with the frozen receiver. An unfamiliar basis failing without
adaptation does not prove it loses necessary information or could never learn.
Reference semantic fidelity, receiver adaptation and complete native use remain
unqualified. No additional candidate was tried.

**Next unfinished slice:** slice 4, define the receiver-learning design/repair
using the observed Noul threshold/status and Score coverage/transfer failures,
preserving Choice. Fitting/adaptation needs new authorization; this comparison
does not justify an automatic pooler rewrite, threshold change or training run.

## Previous checkpoint — architecture slices 1–2 complete, no training

The user authorized architecture-review slices 1–2 only: existing data, 3090,
frozen W/reference, no agents or training. Read the
[architecture/data baseline](architecture-baseline-2026-11-07.md),
[read-only diagnostic protocol](information-loss-v1.md),
[measured findings](information-loss-results-2026-11-07.md) and
[portable diagnostic evidence](evidence/information-loss-v1.json).

Slice 1 metadata audit completed: 1,500 receiver records/288 families; assessed
Noul 168 false/88 true versus X 20/50. All 254 assessed training Score records
are increasing-scale exact points, 170 at endpoints; X includes ten reversed
scales. Reference training has only five distinct annotations per role. No
receiver train/development family/input overlap or contradictory identical-input
targets found. Ten CPU tests pass (14.330 s). Locked data remains unopened.
Portable evidence: `docs/evidence/architecture-baseline-v1.json`.

Slice 2 completed on the 3090 with **all W/D/R/codec/reference weights frozen**:
32 cached training features reused, **180 X cases /90 pairs** newly captured.
Every case exactly reproduces parent-0188 outputs, views, transport and complete
canonical ledger. Parent/final training scalar heads and all **140 saved final
X scalar-head values** reproduce exactly without installing or fitting heads.

**Findings:** saved final Noul head separates the eight training cases by ranking
(AUC **0.5 →1.0**), but all logits remain negative: false −0.318092..−0.134084,
true −0.080566..−0.013014, so fixed-threshold correctness stays **4/8**. Assessed
X ranking worsens **0.600 →0.373**, correctness **20/70** unchanged. This shows
training distinction plus threshold failure, not absence of learned distinction;
it does not establish that moving a threshold would fix generalization.

All **150 changed role-input comparisons** have different pooled slots; all
**70 changed-input pairs** have different h. The **20 identical-input pairs**
have bit-identical wire/h/q. No exact collision observed; semantic preservation
and useful interpretation are still unproven. Centered training Noul/Score ranks
are maximal for these tiny samples (7/9), q/k row ranks 256 with conditions ~3.
Parent Score changes in the reviewed direction in **5/10 reversal, 0/10 scale**
pairs; saved final head values in **5/10, 10/10**. Direction is not exact-value
or generalization success. Parent behavior unchanged **49/180**, statuses 89/180,
pairs 23/90, transport 180/180. Existing semantic acceptance remains BLOCKED.

Successful run **215.737326 s**, peak **3,622,856,704 CUDA bytes**, **21,244 Session
positions**, one prefill/lineage each. Offline source/reference tokens 14,212/5,894
are separate; 160 unique reference encodings, 480 cache hits. Cached visits add
zero positions. All 320 W tensors/physical tying, frozen fingerprints/cache and
grad=None checks pass. **14 CPU tests pass** (10 in 14.330 s +4 in 0.095 s), CUDA
analytic statistics checks and independent receipt/source/accounting audit pass.
Production numeric/backward code is unchanged; completed matrices reused.

Raw: `data/information-loss-20261107/run-v2/`. The first attempt stopped after
one request on a tuple/list comparison bug; canonical full-ledger serialization
fixed it. Keep its failed report/log/partial capture. It has no retained case
ledger; do not claim an exact physical-position total across both attempts.
After exit only original service PID 2094 (3852 MiB) remains. No agents, generation,
reviews, training, locked-data access, service/deployment/Operating Table changes.

**Next unfinished slice:** design and implement one versioned simpler latent
input comparison against this unchanged control (slice 3), explicitly separating
semantic interpretation/transfer from vector differentiation. Use the observed
coverage gaps and readout behavior to define its hypothesis. Do not assume pooler
collapse, rewrite the reference, change thresholds, launch fitting or expand data
automatically. No new representation or subsequent training ran in slices 1–2.

## Previous checkpoint — frozen-representation Noul/Score head experiment complete

The user authorized the head-only 3090 experiment with W/reference/representations
frozen, existing data, no agents or new data. Read the
[predeclared protocol](receiver-head-only-v1.md),
[measured results](receiver-head-only-results-2026-11-07.md) and
[portable evidence](evidence/receiver-head-only-v1.json).

Only **codec.prototypes and codec.score_row**, **3,072 FP32 master parameters**,
train. D/R, input encoders, shared Qev q/k, all status/Choice/Reason heads, budget,
W and reference stay frozen. Restore parent checkpoint-0188 strictly, project
its head moments/settings exactly, retain A4 schedule 188 and clip 1. The same
32 training records/families/order are reused. GPU-extracted detached features
support **1,812 head updates**, windows **189..2000**, **453 repeats /14,496 cached
visits**. These visits add **zero causal positions/prefills**; they are not latent
episodes. Noul eligibility 3,624, Score 4,530; no Choice/non-assessed losses.

There are **32 initial +32 final fresh extractions**, each set charging **3,332
session positions** with exact accounting, one prefill/lineage per case, no
invocation text or gold returns. Final h/candidate tensors, source-state hashes
and head outputs/losses exactly match cached replay. Initial heads/losses match
the saved joint control. Minimal-head forward/gradients match the actual Codec;
first-update receiver/optimizer/CPU-CUDA-Python RNG roundtrip is exact.

**Results:** Noul training truth stays **4/8**, Brier **0.256381 → 0.232680**,
BCE **0.706077 → 0.658275**. Choice training selection stays **4/4**. Score
training interval hits **0 → 3 /10**, mean distance **0.411528 → 0.000959**,
raw MSE **0.206481 → 0.000001851882**. All ten targets are zero-width point
intervals. The seven misses are roughly **0.000879..0.001711** from target;
the three hits are zero targets reached by the existing clamp. No interval or
threshold was relaxed. Approximate Score training fit is demonstrated; its
small residual cause/strict precision feasibility are not isolated.

The 32-update matched point is separately recorded: head-only Noul Brier 0.256086
and Score distance 0.300245 vs joint 0.247584 and 0.166759. The final 1,812-update
endpoint is not an equal-exposure improvement claim over the joint 32-update run.

All **130 E +180 X** cases execute, with **310/310 valid transports** and zero
failed evaluations. Compared with parent 0188 saved receipts (not rerun):

- Independent assessed Noul truth **25 → 24 /80**, Brier **0.284675 → 0.261361**.
- Independent assessed Score interval hits **1 → 0 /50**, mean distance
  **0.271739 → 0.319199**. E distance 0.458949, X 0.284261: training fit did not transfer.
- **All 90 Choice cases exactly retain actual results/statuses/views/grades**;
  assessed Choice delivery remains **47/50**.
- Status stays **130/310**; frozen parent status accepts **0/80** assessed Noul
  and 45/50 assessed Score cases. Delivered assessed success stays **Noul 0/80,
  Choice 47/50, Score 0/50**; overall behavior 83/310; X pairs 23/90. The existing
  semantic/behavior qualifier remains BLOCKED. The experiment itself completed.

**26 CPU tests pass in 29.597 s**, model imports blocked. Independent source,
release/control, replay/ledger, ownership, checkpoint and report audit passes.
All **320 W tensors/physical tying**, frozen representation/reference/cache and
grad=None boundary checks pass. All 12 checkpoints validate. No backbone/loss
arithmetic or tolerance changes; prior numeric matrices reused. Only original
GPU service PID 2094 remains after exit, 3852 MiB. No agents, new data/reviews,
services/deployment changes or Operating Table modifications.

Raw: `data/receiver-head-only-20261107/`. Unpromoted final checkpoint:
`run/checkpoint-2000`, manifest
`7f05e09f5c7a7b58a9dc69556ace3a38d1e9c4f71f46173c716b597e236991e2`;
receiver `618412959e691888215f208f75e1bad8c6d4d0d32bac18dfaf37cf5776191751`.
Frozen representation fingerprint
`94153808343ca8b2bb3b5763d00db94eac5c2c06a205d188d00018208726e64e`.
Reference remains
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.
These head bundles require the pinned parent for frozen backbone adapters;
they are not standalone native-model exports. **COMPLETED_HEAD_ONLY_NOT_PROMOTED**.

**Next unfinished slice:** measure frozen-feature information/separability and
conditioning on the 3090 to distinguish weak/ambiguous features from readout
optimization problems; inspect Score transfer separately from its small training
residuals. This is a proposed diagnostic, not another authorized fit. Do not infer
architectural impossibility, relax exact values/bindings, promote weights, extend
training or launch data generation automatically. Learned invocation/native
answer use, Reason meaning and the complete MVP remain unqualified.

## Previous checkpoint — controlled loss-balance comparison complete; candidate rejected

The user authorized one existing-data loss-balance comparison on the 3090 with
W/reference frozen, no agents or new data. Read the
[predeclared comparison](receiver-loss-balance-v1.md),
[measured results](receiver-loss-balance-results-2026-11-07.md) and
[portable evidence](evidence/receiver-loss-balance-v1.json).

The completed 32-update probe below is reused as the control, **not rerun**.
The candidate independently restores checkpoint-0188, with byte-identical initial
weights, optimizer/RNG/private-row files and identical ownership. All executable
source hashes match the control except the experiment runner. Only **status
1 → 0.1 and feasibility 0.5 → 0.05** change through an opt-in local Window
subclass; production defaults, loss primitives, eligible division, AdamW settings/
schedule, clipping and numeric model paths remain unchanged.

All **32 updates at 189..220**, **256 presentations /32 records /eight repeats**
complete with **zero failed exposures**. Initial heads/losses/input hashes match
exactly; first-window component gradient norms match after the declared scaling
and D cosines match exactly. Training session positions total **26,656**, with
exact accounting, one prefill/lineage, zero held positions/text invocation tokens,
and no gold returns. Eligibility: status 256, Noul 64, Choice 32, Score 80,
feasibility 64, Reason alignment/structure 59 each (control 74; predicted-status
eligibility changes). First-update checkpoint restore reproduces receiver,
optimizer tensors and CPU/CUDA/Python RNG exactly; all six checkpoints audit.

This candidate **does not improve overall behavior**:

- Training assessed accuracy stays Noul **4/8**, Choice **4/4**, Score **0/10**.
  Final Noul Brier improves control **0.247584 → 0.238634**, but all logits remain
  negative. Training true/false probability separation grows, with both classes
  below 0.5. Score distance initially improves at repeat four
  **0.316655 → 0.210957**, then regresses at repeat eight
  **0.166759 → 0.248728**. Do not claim convergence or threshold qualification.
- Held-out assessed Noul **23 → 22 /80**, Choice **47 → 47 /50**, Score **0 → 0 /50**.
  Raw Noul truth remains 25/80; Brier worsens **0.273479 → 0.284861**.
  Score interval distance worsens **0.261081 → 0.272387**.
- All-case behavior **127 → 123 /310**, status **230 → 226 /310**.
  E: Noul **18 → 17 /40**, Choice **14 → 11 /50**, Score **5 → 5 /40**.
  X: unchanged **50/100 Noul, 40/40 Choice, 0/40 Score**, pairs **30/90**.
  All **310/310** transports pass, zero failed evaluations. Existing X semantic/
  behavior qualifier remains **BLOCKED** on correctness/Score/interventions;
  the experiment itself completed, with no execution blocker.

**24 CPU tests pass in 28.838 s**, model imports blocked; independent control/
source/receipt/checkpoint/report audit passes. All **320 W tensors/tying**,
reference, R, budget/nonreceiver codec and grad=None boundary checks pass.
Backbone/loss-primitive sources and tolerances stay unchanged; numeric matrices
are reused. Only the original GPU service PID 2094 remains after exit, 3852 MiB.
No agents, new data/reviews, service/deployment changes or Operating Table changes.
No additional updates or coefficient candidates were launched.

Raw: `data/receiver-loss-balance-20261107/`. Unpromoted final checkpoint:
`run/checkpoint-0220`, manifest
`ee4421e60f0b788014ea733dd147c5c478d8d524f2327142a6ad37c890c22977`;
receiver `84ed1802cae0ecf011a3580baa22c7155409f7860e511a3e16d56df24338bab7`.
Reference remains
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.
Status **COMPLETED_PROBE_NOT_PROMOTED**. Keep coefficient defaults/live weights
unchanged. This rejects this pair at fixed exposures, not other schedules or
the architecture's learnability.

**Next unfinished slice:** define a head-only learnability comparison with fixed
D/input encoders, training only existing Noul prototypes and Score row on frozen
features, using existing records and independent held-out head checks. This can
separate readout learning from representation changes; it is a proposal, not
another authorized run or a proven fix. Preserve the 0.5 threshold, Score
intervals and exact binding. Full training, learned invocation/native result use
and Reason meaning remain unqualified. Do not continue coefficient sweeps or
extend this completed comparison automatically.

## Previous checkpoint — learning report and 32-update receiver probe complete

The user authorized this existing-data probe on the 3090, with W/reference frozen,
no agents or new data. Read the [fixed protocol](receiver-training-probe-v1.md),
[measured report](receiver-training-probe-results-2026-11-07.md) and
[portable evidence](evidence/receiver-training-probe-v1.json).

Implemented training-only deterministic panel selection and separate head/status,
calibration, eligible-loss and delivered-behavior reporting. Actual head telemetry
is captured before labels; an AST audit confirms unchanged producer objectives.
The source-bound runner forks checkpoint-0188 with exact moments/RNG/schedule,
validates original identities, and declares its new sampler/protocol explicitly.
All **32 updates** completed (windows **189..220**, **32 records × eight repeats =
256 presentations**, zero failed exposures). No further updates were launched.

Training heads: Noul truth **4/8 → 4/8**, Choice selection **4/4 → 4/4**, Score
interval **0/10 → 0/10**. Noul Brier improves **0.256381 → 0.247584** but all final
panel logits remain negative. Score mean interval distance improves
**0.411528 → 0.166759** and interval loss **0.206481 → 0.053827**. Status fits
quickly; this is neither a convergence claim nor proof of delivered behavior.

Actual post-update held-out evaluations cover **130 E + 180 X**, with **310/310
valid transports**, no failed cases, and frozen final receiver during evaluation.
Compared with preserved historical checkpoint-0188 receipts (no parent rerun):

- Combined assessed Noul **0 → 23 /80**, Choice **47 → 47 /50**, Score **0 → 0 /50**.
- Noul independent raw truth correctness stays **25/80**; its behavior gain comes
  from assessed status **0 → 76 /80**, not improved binary truth discrimination.
- E behavior: Noul **13 → 18 /40**, Choice **11 → 14 /50**, Score **10 → 5 /40**.
  X: Noul **9 → 50 /100**, Choice **40 → 40 /40**, Score **0 → 0 /40**.
- Overall behavior **83 → 127 /310**; X pairs **23 → 30 /90**. Existing semantic/
  behavior qualification remains **BLOCKED** on correctness/Score/interventions.

First/last-repeat VJP observations show an initial status/feasibility scale
imbalance; later Score D norms exceed status as status fits. Every sampled
combined gradient is clipped under the unchanged rule. These observations do
not isolate a causal fix. No coefficients, intervals or tolerances were relaxed.

**21 CPU tests pass in 29.697 s**, model imports blocked. Independent receipt/
source/checkpoint/report audit passes, including exact first-update receiver,
optimizer and CPU/CUDA/Python RNG recovery. All **320 W tensors/tying**,
reference/R/budget/nonreceiver codec and grad=None checks pass. Existing numeric
matrices are reused; model arithmetic/loss math did not change. No agents,
new data/reviews, service changes or Operating Table changes. PID 2094 remains
the only GPU process after exit (3852 MiB on the 3090).

Raw artifacts: `data/receiver-probe-20261107/`. Unpromoted final checkpoint:
`run/checkpoint-0220`, manifest
`ce1ab2bbcc946846a7b052c7f32cdc0a3e208dc6698e135cac7aa994bad15a41`;
receiver `5eabb8e580144bf0829a5311528a307671bc6568141e489d7277ae5e7fcd329d`.
Reference remains
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.
This is **COMPLETED_PROBE_NOT_PROMOTED**; do not install it into live services.

**Next unfinished slice:** define a controlled value-objective balance comparison
from the same parent/moments and fixed panel/exposures, measuring Noul separation,
Score intervals and retention of status/Choice. No extra run or full training is
authorized by completing this probe. Native learned invocation/answer use and
Reason meaning remain unqualified; do not equate component status gains with MVP.

## Previous checkpoint — projection candidate rejected; prioritize receiver learning

The requested next projection-dispatch experiment is complete on the 3090 using
existing cases, frozen W/reference, no agents, training, data/reviews, service
changes or elapsed-time cap. Read the
[projection screen](projection-dispatch-results-2026-11-07.md) and
[portable evidence](evidence/projection-dispatch-screen-v1.json).

Saved CPU-08 traces select 1,860 frozen projections / 5,582 matrix multiplications
in one Noul case. A small unoptimized Script-dispatch prototype preserves separate
ops and BF16 weight ownership, but **regresses CPU/case 4.4944%**
(1.046848 → 1.093898 s) and throughput **−4.3178%**. Every pair worsens
(4.6955%, 4.5689%, 4.2556% CPU). Peak allocation/reservation are unchanged.
Both same-case traces have 41,722 direct launches, 180 graph launches and 792
stream synchronizations. All six timed packs finish before any profiler starts.

All **147 forward presentations** match prior CPU-04 receipts exactly and pass
independent admission, including cold candidate reference payloads. All 320 W
tensors/tying, receiver/reference identities, frozen/grad=None and zero-update
checks pass. This does not qualify its gradients or learned behavior. The four
GPU operator test methods have one **checkpoint scope error**, preserved in two
runs: ordinary forward/gradient comparisons are exact, but checkpoint
recomputation observes JIT optimizations enabled even after extending the
disabled outer scope through backward. Its internal propagation cause is not
established beyond this guard observation.
No guard/tolerance was relaxed. Full 36-trace qualification is **NOT RUN** for
this rejected candidate. Do not claim this profile passed numeric qualification.

**All active prototype code/selectors/tests were removed.** Backend, scripts and
tests are byte-equivalent to starting commit `22e6646`; previous qualified paths
and defaults remain unchanged. Full prototype sources, patch, failures, timings,
all traces, source hashes, commands and restoration audit are retained in
`data/cpu-09-20261107/`. The installed deprecated Script API would add maintenance
without a measured gain. This rejects one approach, not all possible projection
optimization. QwenOneTwo PID 2094 and The Operating Table remain unchanged.

**Next priority: return to Noul/Choice/Score receiver learning**, rather than
another incremental throughput round. Existing checkpoint-0188 E receipts
(historical, no new evaluation) pass **Noul 13/40, Choice 11/50, Score 10/40**;
all 130 have valid transport. Status correctness is 13/40, 13/50, 15/40. Use these
as the recorded component baseline, not learned-invocation/native-answer proof.
Reuse the existing training release, keep development E/X out of updates, preserve
exact values/bindings/accounting, and evaluate semantic/behavioral changes in all
three operations. Current request forbids training; no updates were launched.
The next learning task needs its own instruction to execute. Keep existing
accepted performance options, without adding another prerequisite tuning project.

## Previous checkpoint — larger pure replay reduces CPU/case by 3.5924%

Implemented and qualified explicit `v3/linear-attention-block-v1` on the 3090.
No agents, training, new data/reviews, service changes or arbitrary time cutoff.
W/reference stay frozen; QwenOneTwo PID 2094 and The Operating Table stay
unchanged. Read the [measured report](linear-attention-block-results-2026-11-07.md),
[contract](linear-attention-block-v1.md) and
[portable evidence](evidence/linear-attention-block-v1.json).

`codec/attention_block.py` replays the original token-ordered recurrence in
32-position chunks, concatenation, RMS normalization and gating as one pure
1..1024-position operation. Actual q/k/v, beta/decay, incoming memory, norm
vector and gate activation are explicit inputs. Dense projections stay outside
capture; no permanent FP32 frozen matrix is retained. The independent gate
projection runs earlier; its admission stays outside capture and no finite or
publication boundary is crossed. VJPs save/copy actual inputs, and outputs are
independent clones. BF16 storage, FP32 transformer, compensated BF16 vocabulary,
immediate guards, one prefill/lineage, budget mapping and accounting stay intact.
Default `attention_block_backend='eager'` stays unchanged; `'graph-v1'` is opt-in.

**Six GPU operator tests pass in 8.218 s**, plus a strengthened saved-input
post-eviction gradient assertion **1/1 in 0.682 s**. **36 prescribed full-model
traces pass in 410.912892 s**: 128/512/1024 × Noul/Choice/Score × oracle /
functional / checkpoint / serving, seed 17. All 27 candidate state comparisons
are exact; all 18 parameter-gradient comparisons have zero RMS difference and
cosine 1 for R/D/wire/emitter. Strict FP32 and original gradient limits are
unchanged. All 320 W tensors/tying, diagnostic weights and grad=None checks pass.
Reuse existing 324/108-case matrices; no BF16 arithmetic shortcut was restored.

The actual frozen checkpoint-0188 reference has identical teacher-forced loss
**0.10903861373662949** and exact 8,192-value latent-input VJP on existing
`train-reference-0000`. Both receiver/reference fingerprints and frozen/grad=None
checks pass with zero updates. The data experiment also clears/recomputes
candidate reference payloads in a full parity pack, so it cannot hide behind
cached oracle outputs. **34 CPU tests pass in 13.223 s**, with model imports
blocked; four plan tests pass again after final scheduling changes in **1.959 s**.

Three alternating resident warm pairs, existing eight-case pack, one frozen
worker/model/context: median worker-case + parent-phase CPU/case
**1.109460 → 1.069604 s (−3.5924%)**. Every pair improves (3.7546%, 3.4705%,
3.4305%). Throughput **0.889370 → 0.921075 cases/s (+3.5649%)**; worker wall
falls 3.5975%. Peak allocation adds **71,241,216 bytes (67.94 MiB)**. Counter-
instrumented cold warmup pack CPU is almost equal (9.177257 / 9.156095 s);
no cold single-request speedup is established. Warm observations have zero
recaptures/evictions: 1,998 old versus 1,548
larger-block requests, with 22/29 cold shape keys fitting explicit capacity 32.
Normal retention remains eight. Warmup CPU samples include counters and actual
inference; they are not isolated capture cost. One case's direct launches **43,324 → 41,722
(−3.6977%)**; graph launches 198→180, device kernel events 72,682→72,502;
**792 stream synchronizations unchanged**. Traces are separate from timing.

The first 131-presentation run interleaved profiling with timing and showed a
large oracle CPU shift after profiling. Its entire timing comparison is excluded,
with all data retained. Corrected fresh-process timing completes all six packs
before profiling; **147 presentations** pass. Both runs together preserve **278
exact CPU-04 3090 receipts**, eleven distinct cases, zero updates. Existing
semantic failures/absence of learned invocation remain unchanged. Raw commands,
source snapshots, both timing sets, numeric/reference checks, traces and audit:
`data/cpu-08-20261107/` (chronological name, not new two-card qualification).
Numeric backend files are identical between qualifier and final benchmark;
only experiment preparation/scheduling changed, so numeric checks were reused.

**Next unfinished slice:** select the next pure projection/attention region from
the saved traces; most dispatch and every required synchronization remain. This
is a modest bounded CPU-06 improvement, not minimal CPU involvement. Do not
retain FP32 W duplicates or defer existing guards to broaden capture. >1024,
full 310-case/unseen-shape, new-profile 3060/shared-thread integration and full
training memory remain unqualified. No training, deployment or default switch.

## Previous checkpoint — shared weights/context tested; CPU/case does not improve

Implemented the requested bounded 3090 diagnostic with private module views,
shared frozen tensors and two thread/stream lanes in one process/CUDA context.
No agents, training, new data/reviews, service changes or time cutoff. W,
receiver and reference remain frozen. QwenOneTwo server PID 2094 and The
Operating Table were left unchanged. Read the
[shared-context report](shared-context-results-2026-11-07.md),
[contract](shared-context-diagnostic-v1.md) and
[portable evidence](evidence/shared-context-diagnostic-v1.json).

`training/shared_frozen.py` shares the exact 1,719 registered tensors
(1,656,951,700 bytes), with private module/parameter dictionaries, tokenizer,
precision owners and reference memo caches. It preserves embedding tying and
reference/live isolation; rejects trainable tensors and dynamic RoPE. Sessions,
registries and causal caches are private per case. Actual concurrent
cold-reference calls exercise the temporary `functional_call` substitutions.
All stream shapes are warmed serially before the process-wide
`recurrence.replay_only()` barrier forbids new forward/backward capture. Cache
capacity is bounded at 32/lane (64 shared); ordinary defaults remain eight.
All workers finish main work before final integrity checks start.

Three alternating replica/shared pairs use the identical existing eight-case
queue × three, with identical eleven-case warmup/cold-reference stress per lane.
All seven formal runs, including one serial control, exit zero. **Sharing fails
the CPU/case target:** median parent-plus-worker main CPU/case
**1.273707 → 1.378479 s (+8.2258%)**; main throughput
**1.506258 → 1.273485 cases/s (−15.4537%)**. CPU/case rises in every pair
(8.4832%, 5.8451%, 10.7591%). Ready allocation falls **46.7664%**
(3,587,922,944 → 1,909,981,696 bytes); setup-inclusive CPU falls **21.6540%**,
but end-to-end wall rises **26.9784%**. Serial control CPU/case is **1.079273 s**
at **0.921948 cases/s** (one sample). Keep this memory/setup option opt-in;
normal serving/training/two-GPU defaults do not change. No claim that the GIL
alone explains the slowdown. CPU intervals and Queue-feeder cutoffs are explicit
in the report; allocator peaks are not sampled physical-device peaks.

**454 formal presentations** independently admitted (168 main + 143 warmup +
143 concurrent cold-reference stress), eleven distinct cases, exact receipts
across all runs. **376** applicable presentations match the CPU-04 3090 baseline
exactly. An earlier 46-presentation probe also matches, retained separately and
excluded from medians. Initial/final 320-tensor W/tying checks, receiver/reference
hashes, frozen/grad=None ownership and zero updates all pass. Capture entries
remain 30/replica and 60/shared throughout stress/main. Existing semantic
failures and lack of learned invocation remain unchanged.

**30 CPU tests pass in 10.890 s**, with model runtimes blocked; initial AST
fixture namespace failure was corrected and retained in the logs. **Four
recurrence GPU tests pass in 0.946 s**, including exact connected gradients,
checkpoint/stream behavior, cold-capture rejection and nonfinite checks.
**Two sharing GPU tests pass in 0.324 s**, including deliberately overlapping
reference/live substitutions and rejection of trainable/duplicated tensors.
Existing recurrence definitions are AST-identical after removing only new cold
capture guards; other math/reference/evaluator files are byte-identical to
`ef78a4f`. Reuse completed 324/108-case numeric qualification; no tolerance
change or expensive matrix rerun. Raw plans, commands, sources, complete events,
tests, initial failures, all samples and audits: `data/cpu-07-20261107/`.
That chronological directory name does not complete task CPU-07.

**Next unfinished slice:** CPU-06 still needs an actual reduction in per-case
dispatch work within a larger pure GPU region, preserving exact operations and
all immediate guard/control boundaries. Shared weights reduce duplication and
setup CPU, not repeated operator dispatch. Full 310-case/long-context/threaded
gradient training and 3060 shared-context behavior remain unqualified. Do not
grow replicas, relax RMS tolerances, generate data, train or deploy by default.

## Previous checkpoint — opt-in resident pool measured; kernel shortcuts rejected

GPU throughput work remains authorized, with existing data and frozen W/reference.
No agents, new semantic data/reviews, training or live-service changes. QwenOneTwo’s
server **PID 2094 (`qwen-one-two.service`, port 3211)** was deliberately left running.
Read [pool results](resident-pool-results-2026-11-07.md),
[pool contract](resident-worker-pool-v1.md), [portable evidence](evidence/resident-worker-pool-v1.json),
and [rejected kernel screens](throughput-kernel-screens-2026-11-07.md).

The existing runner adds `--workers-per-device 1|2` and `--repetitions N`.
Defaults remain one process/card and one presentation. Logical worker IDs bind
physical UUIDs before Torch import; each process owns a complete frozen model
and whole independent Sessions. Repeated coverage/accounting uses `(repetition,
case ID)`. Reports distinguish worker/device counts, preserve all observations,
and score only the first repetition. Same-GPU receipts must match exactly.
No episode/cache/packet is split or shared across workers; no new coordinator.

Three alternating two-card pairs compare **two versus four processes** on the
same **24-presentation queue** (eight existing cases × three). Median main time
**16.858844 → 12.205966 s**, throughput **1.381197×**. End-to-end
**44.831816 → 42.568699 s**, only **1.053164×**. Worker CPU/case increases
**32.9089%** (1.333451 → 1.772276 s); parent-plus-workers setup-inclusive CPU/case
increases **81.2976%** (3.370115 → 6.109938 s). Allocation roughly doubles with
replicated BF16 models. This is a throughput/CPU/memory tradeoff, not reduced CPU
work. Keep the narrower default; do not multiply these gains by prior reports.

All three pairs improve main and end-to-end throughput. **198 final presentations**
(144 main + 54 startup) pass independent admission/receipt accounting. Every main
receipt matches its same-card CPU-04 baseline exactly. Initial/final W/tying,
receiver/reference fingerprints and grad=None pass, with zero updates. Six run
exit codes are zero. Model/codec/device/precision/source/reference/evaluator math
remains byte-identical to `c8e9daf`; unchanged numeric/gradient evidence is reused.
**30 CPU tests pass in 10.791 s**, with model imports blocked, including spawned
one/two/four-process echo fixtures. Final admission tests pass 7/7 in 0.007 s;
all six saved startup sets pass the strengthened same-card group check. Timing
sources precede that CPU-only checker refinement; only `compare_ready` changes,
with both versions/AST evidence retained. Raw sources, plans, receipts, all timing
samples and audits: `data/cpu-06-20261107/`.

The initial single-card probe (one pair, 24 cases) measured **1.638×** main
throughput for two 3090 processes, at higher CPU cost. It is exploratory support,
not a separate long-run qualification. Main queue timing may overlap earlier
workers’ final checks. Individual allocation peaks are not sampled device peaks.
Full-context/310-case scaling, training memory and four-worker CUDA profiling
remain unqualified. Existing semantic failures and lack of learned invocation
remain unchanged; no ReferenceEvidence/accelerator grant is produced.

**CPU-06 findings:** native fused RMS fails original strict FP32 states and
parameter-gradient limits in the first 128-position Noul pair (worst state RMS
7.98795e-5; wire gradient RMS 0.0055445). Other requested pairs are NOT RUN after
rejection. Replay/pointwise and finite-predicate variants have only modest isolated
gains and were not adopted. Their code/logs are archived; no prototype remains
active. No tolerance, precision, guard timing or model arithmetic was relaxed.
[Screening evidence](evidence/throughput-kernel-screens-v1.json).

**Next unfinished target:** reduce per-case host work without more replicas.
CPU-06 remains open. A larger pure region must retain exact operations or pass
original strict qualification; native RMS is not an approved shortcut. A shared
CUDA-context/weight approach would need reference/live and mutable-module
isolation first. Keep all immediate failure boundaries; do not start training,
deployment, unbounded pool growth or another numeric matrix merely by default.

## Previous checkpoint — CPU-04 retention measured; two-card integration passes

GPU use remains authorized for throughput diagnostics. No agents, new semantic
data, capability/reference training or services were used/changed. Two coherent
slices follow the earlier CPU-only instrumentation: CPU-03 baselines/guard fix
and CPU-04 cache-retention comparison. [CPU-04 report](cpu-04-results-2026-11-07.md),
[contract](recurrence-cache-retention-v1.md), [portable evidence](evidence/cpu-04-v1.json).

The normal diagnostic runner accepts `prepare --cache-limit 32`; model/training
defaults remain eight. This explicitly selected policy retains the same
device/stream/shape/dtype key and equations. Worker admission binds the complete
scheduled job. Parent main-phase wall time now separates resident throughput
from setup/final integrity. No finite boundary, tolerance or accounting changes.

Three alternating resident pairs on each card use the same existing eight cases
and frozen checkpoint/reference. Median summed case time improves **5.2948% on
3090 / 4.0411% on 3060**; CPU falls **5.0587% / 3.9190%**. All three pairs improve
both metrics. Independent warm observations show **37 → 0 recaptures/evictions**
over 1,998 requests. Allocated peak increases **53,906,432 bytes (51.41 MiB)**.
Timed passes have profiling/counters off; cold warmups and separate counter
observations are fully disclosed. Every repeated case receipt matches exactly.

The normal 32-entry path also passes a single finite integration comparison:
3090 main phase **11.418856 s**, both cards **7.132641 s**: **1.600930× throughput**.
Including setup, **37.941083 / 34.982324 s**, only **1.084579×**. Two cards cost
**6.6818% more worker CPU/case** and **65.8653% more setup-inclusive CPU/case**.
Do not conflate parallel throughput with reduced total CPU work or claim 2×.
This is eight-case first-pack evidence, not stable full-workload scaling.

CPU fixtures pass **25/25 in 8.698 s**, plus the final explicit-execution test.
The existing recurrence suite passes **8/8 per card at capacity 32**, including
exact eager values/derivatives, 512/1024 composition, checkpoint, streams and
failure guards. Recurrence/Session/neural/backbone/evaluator bytes are unchanged
from `4879331`. Both independent audits pass; all weights/ties/fingerprints and
grad=None checks pass. There are **271 CPU-04 model presentations, zero updates**.
No new 324-case matrix, full 3060 transformer-state qualification, training-memory
qualification or learned-flow certification is claimed. Noul/Score failures stay.

**Next unfinished performance slice: CPU-06 candidate selection**, using the
existing trace to isolate a larger pure backbone region between required guards.
The retained normalization-tail specification is unimplemented and need not be
chosen if profiling shows a better region. CPU-05 transfers are not the next
demonstrated bottleneck. Use explicit capacity 32 for this read-only pack; keep
future training/backward memory and semantics qualification separate. No larger
capture or training is queued.

## Previous checkpoint — CPU-03 hardware baselines complete

The user lifted the GPU hold on 2026-11-07 for throughput work. Both configured
cards are available for explicit read-only diagnostics; no agents, training,
new semantic data or service changes. [Measured CPU-03 report](cpu-03-results-2026-11-07.md)
and [portable evidence](evidence/cpu-03-v1.json). Earlier hold statements below
describe their historical turns and no longer block authorized throughput work.

Six eight-case runs complete: baseline/profile on each card, a first-job 3090
CUDA trace and a two-card trial. Same-card instrumentation receipts match
exactly; cross-card control/provenance match and scalar maximum difference is
2.384185791015625e-7. All weights remain frozen/unchanged with grad=None and zero
updates. No semantic or full 3060 numeric qualification is granted.

3090/3060 baseline summed main intervals are 12.646896/12.384660 s; main CPU
costs 1.567354/1.535788 seconds/case. Two-card end-to-end is 35.499134 s versus
39.483971 s on 3090, a local 1.112× throughput observation with setup included.
Do not claim a steady-state 2× benefit. CPU-03 adds singleton worker selection
and measured memory; separate parent main-phase timing is still needed.

Eight existing recurrence tests pass on each card. A failed initial 3060
checkpoint test exposed loss of thread-local diagnostic permission in autograd
engine threads. Admission and optimizer prohibition are now process-wide within
the dedicated non-nested diagnostic scope. The failed log is retained; the
corrected suite passes. CPU tests pass 24/24 in 6.770 s.

**Next unfinished slice: CPU-04**. Actual main-work counters show 83 recaptures
and evictions at capacity eight across 29 keys, costing about 0.7 CPU-seconds.
Compare a larger bounded cache with same-card repeated measurements and exact
output/gradient/cache/accounting checks. Do not remove finite checks or change
model arithmetic. Full warm distributions/transformer-state numeric comparison
across cards are not certified by the initial baselines.

## Previous checkpoint — CPU-01/02 implemented, GPUs remained off

The user authorized CPU-01/02 implementation, with both GPUs off, existing data,
no agents or training, and direct commit/push. The existing diagnostic runner
now records parent/worker process CPU and wall intervals, per-case phases,
dispatch counts, optional detailed call/copy/cache attribution and an optional
future first-job CUDA trace. [Results and limits](cpu-01-02-results-2026-11-07.md),
[portable evidence](evidence/cpu-01-02-v1.json), and
[synchronization map/capture specification](cpu-sync-boundaries-v1.md).

**88/88 CPU tests passed in 19.029 s** with model-runtime imports blocked;
the final targeted dispatch suite passes **11/11 in 0.746 s**. All 310 saved
final 3090 receipts pass identity/grade/accounting rechecks. Source-expression
and literal-byte fixtures pass. These are CPU validations, not GPU numerical,
gradient or performance qualification. One initial fixture namespace failure
and its fix are preserved in the evidence.

Four new source-bound plans under `data/cpu-01-02-20261107/` are
`full/plan.json` (310 cases/220 jobs), `cpu-baseline/plan.json`,
`host-profile/plan.json`, and `cuda-trace/plan.json` (each eight cases/seven jobs).
The subset pins six E min/max historical Session-position cases plus an intact
X pair; shared startup adds six presentations. Nothing was executed on either
GPU, including status probes. No new semantic data, agents, training, model
runtime imports or service changes occurred. Older plans are source-bound
historical evidence, not current execution plans.

CPU-02 specifies a normalization-tail capture candidate with its finite guard
left in place. It is disabled/unimplemented. No arithmetic, graph policy,
eligibility, accounting or model-facing packets changed. Detailed attribution
counts selected host calls and explicit audit-copy requests; it is not a full
kernel/transfer census. GPU compute/idle remains null. The separate CPU fixture
overhead measurement does not establish actual model profiling overhead.

**Next unfinished slice: CPU-03**, once the user explicitly lifts the GPU hold:
verify actual placement and 3060 compatibility, collect same-card baseline,
profile overhead and kernel/wait evidence, and check instrumented outputs and
accounting. No hardware execution is queued. Use those measurements to choose
CPU-04/05/06; do not implement every candidate automatically. Existing Noul/Score
semantic failures remain unresolved.

## Previous checkpoint — two-GPU diagnostic runner built, GPUs not run

The user subsequently requested a prioritized task list for substantially
reducing CPU involvement. See [CPU-involvement tasks](cpu-involvement-tasks.md).
It separates CPU-only preparation, deferred GPU measurement/optimization and
future training execution. The original task-list turn changed documentation
only; CPU-01/02 were implemented subsequently as recorded above.

**Both GPUs remain off, including status probes. No CPU model fallback.** The
user authorized building a 3090/3060 diagnostic runner, with existing data, no
agents or training, and commit/push. That build is complete and CPU-validated;
no model workload was launched. Read the [runner specification/build evidence](two-gpu-diagnostics-v1.md),
[portable evidence](evidence/two-gpu-runner-v1.json), and
`scripts/two_gpu_diagnostics.py`.

The prepared existing-data plan is
`data/two-gpu-runner-20261107/prepared-ready/plan.json`:
**220 main jobs / 310 cases** (130 E singles + 90 atomic X pairs), plus three
shared startup cases on each card (six declared repeated presentations). It pins
sources, release/record/view bytes, final receiver checkpoint 188, reference,
expected weight hashes and the saved GPU inventory. Status is
**PREPARED_GPU_NOT_RUN**; weight-file checks and actual names/placement are deferred.
The secondary UUID is taken from saved inventory; its expected 3060 name has not
been freshly verified. Actual startup rejects a wrong/unavailable card.

Future explicitly opted-in execution uses two independent read-only processes,
each with one visible UUID mapped to local cuda:0, frozen W/reference/live weights,
and whole Session/cache ownership. Jobs are assigned on completion, pairs remain
together, canonical report order is restored, and actual latent `evaluate_case`
is reused. The three shared-case readout/control comparisons precede main jobs;
they are not full 3060 numeric qualification. Per-case metadata/grades/accounting
and final immutable-weight receipts are checked. Errors stop owned children,
retain partial unqualified receipts and refuse silent retry/replay/overwrite.
Automatic resume is not implemented. Existing services remain unchanged.

`g1_device.py` provides a narrow explicit diagnostic scope for the secondary
UUID. Ordinary paths stay 3090-only; optimizer Windows reject diagnostic scopes
on either card. Guard changes preserve arithmetic/optimizer bodies and **82
other function definitions** by static AST comparison. Existing numeric reports
remain historical; **no new GPU numerical, gradient, memory or throughput
qualification** is claimed. Experimental reports cannot grant ReferenceEvidence.

**78 unique CPU tests pass**: broad **77/77 in 16.830 s**, final targeted
**11/11 in 0.745 s**, with model imports blocked and CPU-only echo workers.
A premature-exit fixture exposed unwrapped BrokenPipeError; channel errors are
now normalized and the failed log retained. All **310 historical final 3090
receipts** pass the new CPU validator. Final plan/source/pair/coverage/receipt
audit, compilation and whitespace pass. **Zero new GPU/3060 results**, no agents,
new semantic data/reviews, training or service changes. Raw evidence:
`data/two-gpu-runner-20261107/`. Do not invoke `run --execute-gpu` during the hold.

**Next unfinished slice:** after explicit permission to use GPUs again, verify
the secondary card and targeted model/numeric/cache/gradient compatibility, then
measure independent-case throughput. No execution/training is queued. Existing
Noul transfer and Score failures remain unresolved.

## Previous checkpoint — CPU asset lookup/audit profile complete

**Keep the GPU off at the user's direction, including probes/status tools. No
CPU model fallback. Leave existing services unchanged.** The requested CPU
profile and next concrete bottleneck fix are complete; no active job remains.
Read the [measured report](asset-audit-results-2026-11-07.md),
[portable evidence](evidence/asset-audit-v1.json) and
`scripts/benchmark_assets.py`.

The old asset accessor rebuilt a dictionary of all **17,809 training assets**
on every lookup. Initial CPU measurements found **7.76 / 7.69 ms** for eight
lookups, versus **0.40 / 0.45 ms** for actual saved audit JSON serialization.
The fix constructs one private read-only index from the validated bytes. The
original sorted asset tuple, loader checks, record copies and all audit
serialization/flush/hash behavior remain unchanged; payload bytes are shared,
not cloned. No new dependency or mutable data cache.

Five alternating CPU pairs (25 repetitions/path/operation/pair) give eight-view
preparation medians **15.269198→6.943565 ms** and **17.796844→9.404333 ms**:
**8.325633 / 8.392512 ms** saved, **54.5257% / 47.1573%** less CPU preparation.
Eight isolated lookups fall **8.108526 / 8.057267 ms→0.925440 / 0.924640 µs**.
These are fixed CPU batches, **not measured full training speedups**. Audit JSON
bytes match the initial profile. Tensor hashes/device transfers are excluded.

All **20,641 asset payloads** and full train/development snapshots match the
literal `c10c163` baseline. **110 prior input/provenance cases** reproduce their
committed hashes; **16 direct paired component cases** match. **52 model-free
tests pass in 16.179 s** with model-runtime imports blocked. Immutable index,
byte sharing, empty/Unicode/missing/invalid keys, no lookup filesystem access,
disk mutation isolation/reload rejection and existing provenance/Reason/sampling
checks pass. Loader and non-class module AST, source hashes, timing/receipt
recomputation, compilation and whitespace checks pass. Benchmark **exit 0**.
Raw receipts: `data/asset-audit-20261107/`.

No GPU tools, model loads, agents, new semantic data/reviews, training or service
changes. Prior numeric and learned-flow results remain unchanged. **Next
unfinished slice:** GPU checkpoint/cache-policy comparisons remain paused.
Independent CPU provenance traversal is now visible in profiles, but any change
must preserve mutation/contamination rejection; none is queued. Noul transfer
and Score numerical failures remain unresolved. No new data/training is queued.

## Previous checkpoint — CPU-only manifest optimization complete

**GPU work is on hold at the user's direction because of noise complaints.**
Do not activate the GPU for workloads, probes or status tools in follow-up work
without the user lifting that restriction. CPU model execution remains banned.
Leave existing services unchanged. The user authorized ordinary CPU work on
repeated manifest parsing; **that slice is complete, with no active job.** Read
the [measured report](manifest-access-results-2026-11-07.md),
[portable evidence](evidence/manifest-access-v1.json) and
`scripts/benchmark_manifest.py`.

The validated immutable release now exposes `purpose` and frozenset
`selected_ids`. Per-record component/reference/native/supervision admission uses
them; public JSON properties still return fresh copies. Loader validation is
AST-identical before the final constructor call. Source/provenance checks,
answer blindness, the Reason exception, record comparisons and sampling order
are unchanged. No mutable decoded manifest is cached or handed out by reference.

Literal prior sources from `8459e84` yield identical full training/development
snapshots (**9,728 / 1,600 records; 17,809 / 2,832 assets**) and **110 matching
existing input/provenance cases** (52 reference, 46 component, 12 native).
Five alternating CPU pairs on eight existing receiver records give median
**0.909318→0.015810 s** preparation time, saving **0.893508 s**; full-manifest
reads fall **16→0**. This is CPU input preparation, not full training timing.

**51 model-free tests pass in 16.263 s**, with model-runtime imports blocked.
An initial new test fixture omitted its operation; it was corrected and its
failed log retained. The final full run, benchmark **exit 0**, source/snapshot
hashes, loader AST, timing recomputation, compilation and whitespace checks pass.
Raw receipts: `data/manifest-access-20261107/`. No GPU tools, model loads,
CPU inference, new semantic data/reviews, agents, training or service changes.
Existing numeric and learned-flow results below remain unchanged; no new model
qualification or end-to-end training speedup is claimed.

**Next unfinished slice:** checkpoint/cache-policy comparisons remain pending
and paused until GPU work is allowed again. Independent CPU profiling of repeated
asset lookups/audit serialization is available as a separate next slice, not
launched here. Noul transfer and Score numerical failures remain unresolved.
No training or data generation is queued.

## Previous checkpoint — fused finite checks complete

The user requested Astra's fused finite checks, a 3090 benchmark, exact
output/gradient/accounting checks, frozen W/reference, no new data/agents/training,
and commit/push. **This slice is implemented and validated. No job remains
active.** Read the [measured report](fused-finite-results-2026-11-07.md),
[portable evidence](evidence/fused-finite-v1.json) and
`scripts/benchmark_finite.py`.

The runtime and training engine use read-only multi-tensor infinity norms and
one host finite decision at each existing boundary. All hidden/cache tensors,
each component's gradients, combined gradients and pre/post optimizer state
remain checked; no per-linear guard was deferred. Objective arithmetic,
per-component VJPs, eligible denominators, None masks, clipping/rollback,
precision, accounting and control behavior remain unchanged.

On the pinned RTX 3090, two fixed existing eight-record batches at final receiver
checkpoint 188 pass output/cache/loss-byte and exact gradient-value parity
against literal pre-change sources from `d767129`. There are **4,671 component
buffer / 809 combined-gradient tensor comparisons**, all seven eligible loss
types, **17 / 25 VJPs**, and identical source/accounting reports. Each Session
retains one prefill/cache lineage. The **32 parity / 96 timed presentations**
account for **14,576 objective-ledger positions**, excluding separate encodings/
rematerializations. Intentional masked -Inf logits remain unchanged.

Three alternating timed pairs per batch give median **36.844481→36.166659 s**
and **51.328583→50.424776 s**: **1.839684% / 1.760827%** less elapsed time.
The predicates themselves improve **7.4509× / 8.5798×**, with measured host
scalar decisions **49→1 / 405→1**. This is a small full-window improvement,
not a solution to the wider iteration-speed problem. Peak allocated tensor
memory is identical between paths. Optimizer update/clipping time is excluded.

**25 unique targeted CUDA tests pass**, including six boundary-fixture tests
against both literal legacy and fused classes (6+6, no failures/errors).
NaN/Inf, finite extremes, graph/version immutability, loss/derivative/buffer
failures, CPU-state rejection, partial-step rollback, cancellation, mutated
cache, lineage/position and accounting checks pass. The benchmark exits **0**
after **750.440664 s**; source/receipt/eligibility/timing recomputation passes.
All **320 W tensors / physical tying**, frozen reference, R/receiver/codec
fingerprints and final grad=None checks pass. **Zero model optimizer updates**;
no generation, agents, annotation changes or service changes. Raw artifacts:
`data/fused-finite-20261107/`. The old 324-case numeric matrix is reused;
no new broad-matrix or semantic/full-flow certification is claimed. Original
BF16 transformer arithmetic remains failed.

**Next unfinished slice:** separately measure checkpoint/cache-policy effects
with gradient/accounting equivalence before selecting a change. This is pending,
not launched. The Noul training/development transfer and Score numerical failures
below remain unresolved. No further training or data generation is queued.

## Previous checkpoint — receiver-learning experiment complete

**The requested receiver experiment and performance audit are complete. No job
remains active.** Read the [measured results](receiver-learning-results-2026-11-07.md),
[portable evidence](evidence/receiver-learning-v1.json),
[fixed protocol](receiver-learning-v1.md),
[training-path evidence](evidence/receiver-training-paths-v1.json),
[exact continuation](evidence/receiver-learning-resume-v1.json) and
[Astra performance audit](receiver-performance-audit-2026-11-07.md).

The user's explicit authorization allowed receiver-only capability learning on
the RTX 3090, superseding the earlier no-capability-training boundary for this
run. All **1,500 existing receiver records** were presented once, plus four
declared real-record repeats: **1,504 presentations / 188 optimizer windows**,
with **zero failed training exposures**. This is one initial full receiver pass,
not the full 2,000-window component phase or convergence. No reference pass 13,
writer/gate/reader training, new scenarios/reviews, promotion or service changes.

The isolated request-to-heads objective prevents malformed untrained Reason
feedback from discarding valid classifier losses. It uses actual latent requests,
answer-blind clean inputs and labels accessed after evaluation. It ends at the
heads with no supplied/claimed return or native answer. Frozen annotation caches
are bit-identical; D-dependent sources remain fresh. Actual evaluation continues
to decode and consume real model outputs without overrides.

| Operation | E baseline → final | X baseline → final |
|---|---:|---:|
| Noul | **4 → 13 / 40** | **13 → 9 / 100** |
| Choice | **8 → 11 / 50** | **31 → 40 / 40** |
| Score | **0 → 10 / 40** | **0 → 0 / 40** |

All ten final E Score successes are **non-assessed responses**. Numeric Score
agreement remains **0/50** assessable E+X cases, with large continuous errors.
Noul assesses **0/80** assessable development cases; its E gain mainly reflects
Unresolved responses. Choice reaches **40/40 controlled X cases / 20/20 Choice
pairs**, but missing/conflicting/no-feasible E statuses remain unsuccessful.
Do not certify all Choice or overall semantics from that subset.

Overall behavioral successes **56→83/310**, returned statuses **115→130/310**,
valid exchanges **280→310/310**, complete intervention pairs **16→23/90**.
All final decoding failures disappear, but Reason support meaning is untested.
Unsupported Assessed returns remain **46/130** reviewed non-assessable cases
(previously 95/130). The scoped intervention qualification remains **BLOCKED**.
No native answer consumption or learned invocation/writing is qualified.

A read-only final-checkpoint probe on the same 13 preflight training records
improves status accuracy **3→8/13** and all seven loss types on common eligible
examples. Status CE **1.975497→0.833762**; the one assessed Choice loss
**0.251437→0.00000309944**. Value probes are one example per operation, not a full
training recovery score. This demonstrates connected learning, not held-out
semantic competence or a unique cause of the remaining failure.

**All 320 W tensors / physical tying, frozen reference, R and frozen codec checks
pass.** All gradients finish None. Every training record/view hash, window,
eligibility count and Session accounting receipt passes audit; objective ledger
**160,853 positions**, excluding separate offline encodings/rematerializations.
All files in ten checkpoints pass hashes, and development grades/pairs are
recomputed. Training/development selected families are disjoint (**288/104**).
Locked labels remain unopened. Existing numeric matrices are reused; no new
trained-weight parity or full-flow certificate is claimed.

Tests: **25 unique model-free checks**, **10 CUDA loss/accumulation checks**,
**13 actual preflight operation/status gradient strata**, exact first-update
model/optimizer checkpoint roundtrip, and the final zero-update matched probe.
Sources/weights use v3 FP32 transformer / compensated BF16 vocabulary arithmetic;
original BF16 transformer arithmetic is still failed. Raw evidence:
`data/receiver-learning-20261106/`.

The original eligibility Counter summed tensors, retaining graphs; optimizer
counts were correct. It was fixed at checkpoint **75**, intentional SIGINT
**exit -2**; all weights/moments/RNG/order/cursor continue exactly at **76**.
No completed optimizer window was replayed; in-flight preparation can recur.
Continuation and matched probe both **exit 0**. Prefix and continuation source
identities/logs remain preserved. Active execution is over; do not restart it.

Final checkpoint: `continued/checkpoint-0188` in that evidence root, manifest
`0cd1d4bc55341e78339c1b0b05dec0b3650573d7501fb3f3d2fc6d2f649b6e58`.
Final receiver fingerprint
`94218aae6ec62a7745c349af8e9975fd4987520cf9f1345f38e44586e5e7742c`;
reference unchanged
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.

The user subsequently authorized **one Astra read-only performance subagent**.
It found one trainer, low CPU contention and repeated VJP/synchronization costs;
it changed no training source, job or service and launched no model workload.
The parent reproduced its timing statistics. The suggestions were initially
unbenchmarked; the finite-check slice above now has measured results. No other
agents are authorized by that request.

The subsequent finite-check slice is completed above; measured checkpoint/cache-
policy comparisons remain pending. Separately isolate Noul's training/development
transfer and Score's numerical failure before choosing further fitting or data
changes. No further training, generation or deployment is queued. Preserve the
known existing Reason annotation discrepancy; no corrected annotation/review is
claimed. Keep current failures and all previous reports intact.

## Previous checkpoint — semantic/behavioral gate replacement

**Implemented and evaluated; the tested configuration is BLOCKED by actual
receiver behavior, not by exact text reconstruction. No job remains active.**
Read the [new protocol](reference-behavior-v1.md),
[measured results](reference-behavior-results-2026-11-06.md) and
[portable evidence](evidence/reference-behavior-v1.json).

`load_reference_evidence` now accepts only the latent behavioral protocol;
recovery-v1 artifacts are diagnostics and cannot grant acceptance. Frozen,
fingerprint-checked candidate references may construct exactly bound experimental
supervision or initialize private codecs without a receiver pass. This is not
semantic acceptance or authorization to train. Nearby/mutated semantic contents,
foreign/incomplete sources, unadmitted snapshots and incompatible actual returns
remain unsupported. Diagnostic accumulation remains forbidden; Reason keeps its
separate approved status/support input exception.

On the specified RTX 3090, **180 existing development receiver cases / 90 pairs**
completed with actual latent requests/results, unmodified heads, neutral Session
prompts and no gold-return substitution. Noul passes **13/100**, Choice **31/40**,
Score **0/40**; complete pairs **16/90**; returned-status accuracy **91/180**.
There are **164 valid exchanges / 16 blocked invalid Reason decodes** (13 pointer
compatibility failures, three aggregate-bound failures). **All 180** receipt audits
pass single prefill/cache lineage/exact accounting. Rejected decodes are failures,
not wrongly admitted bindings. Source/value/unit bytes, candidate identities and
approved Score intervals are preserved without widening.

All **320 W tensors / physical tying** pass; reference and receiver fingerprints
are unchanged; gradients are None and optimizer updates zero. The loader
recomputes outcomes and rejects this failed report; no acceptance was granted.
The existing pass-12 reference is unchanged. No agents, new data, fitting,
capability training, promotion, service changes or pass 13. The separate 20
category-9 supplied-return reader cases are explicitly omitted, not certified.

**34 model-free tests pass**, the original seven GPU binding/gradient tests pass
in **43.426 s**, and four final strengthened GPU binding tests pass in **17.467 s**.
The input-gradient norm remains **0.25784578919410706**, with no live-adapter
gradients or optimizer updates. A test that mistook a protected registry copy for
admitted content was corrected; its failed log is retained. Compilation,
whitespace, source-snapshot and evidence-recalculation checks pass. Existing
numeric matrices were reused, since model arithmetic is unchanged.

Raw evidence: `data/reference-behavior-20261106/`. Setup/interruption exits are
**1 / 143 / 1**; the final run exits **2** for unsuccessful qualification. All
**29 completed partial-run cases were reused**, followed by **151 newly completed
cases**; no completed model measurement was repeated. Source/forward AST,
record/view, reference/receiver/profile identities and recomputed grades bind
reuse. Do not claim an exact physical-work total around interruption.

**Next unfinished slice:** distinguish the reference/pooler's contribution from
the receiver's using these saved behavioral failures before choosing further
fitting. Reason support meaning, Choice feasibility, other statuses, unforced
invocation/writing and native answer consumption remain unqualified. This result
is not full-flow certification and does not prove a unique failure cause. No
follow-on run is queued; further training/new data remain unauthorized.

## Previous checkpoint — complete-coverage endpoint

**The complete-coverage reference experiment is STOPPED after pass 12, exactly
as the user requested. Pass 13 never started. No experiment job remains active.**
Read the [measured report](reference-coverage-results-2026-11-06.md),
[portable receipt](evidence/reference-complete-coverage-v1.json),
[protocol and explicit user stop](reference-complete-coverage-v1.md), and
[parent-weight meaning inspection](reference-meaning-inspection-2026-11-06.md).

Implementation `caa1db5` adds complete shuffled passes and a reference-only
continuation schedule. User-stop implementation `c36d6ac` adds
`--stop-after-pass 12`, persists it in checkpoints and guards before another
iteration. It was installed via checkpoint 5,800 with unchanged model/optimizer
arithmetic. The original automatic loop had no fixed pass count; it continued
while recovery or loss improved. The user-selected endpoint overrides that
rule. **This is not a plateau or convergence claim.**

The experiment completed **12 passes / 5,064 new pooler updates / 40,512 eligible
presentations**, exactly **12 new presentations per each of 3,376 unique views**.
Cumulative optimizer update is **6,064**. W, reference T/codec and live weights
stay frozen; all model workloads used the specified RTX 3090, with no agents,
new source records/reviews, capability training, promotion or service changes.

| Diagnostic | Start | Final pass 12 |
|---|---:|---:|
| Fixed role-stratified training panel | 36/191 | **71/191** |
| Prior training diagnostic cases | 25/40 | **26/40** |
| Six exposure controls | 6/6 | **4/6** |
| Original development exact recovery | 0/100 | **0/100 at both final observations** |
| Original development contrasts | 100/100 | **100/100 at both final observations** |

Per-pass panel scores: **32, 35, 39, 47, 55, 60, 61, 60, 67, 69, 73, 71**.
Mean whole-pass CE declines from **0.6830126312468918** to
**0.3984135712201553**. Pass 11 is the highest panel observation (73/191), but
final pass-12 weights are retained without substituting an earlier checkpoint.
Final role scores: Question **4/32**, Target **3/32**, Alternative **25/32**,
Conditions **6/32**, Reason **2/32**; all 31 views in the five smaller roles
recover. The original 40 cases have three gains (`0019`, `0041`, `0039`) and two
losses (`0010`, `0001`), so retention is incomplete. This run genuinely continues
the six-success reference and regresses two controls; the older reset-to-common-
initial-state warning applies to the earlier broad control, not this continuation.

All **24 development observations** remain 0/100 exact and 100/100 contrasts.
Final required original-certificate observations are **5,964 / 6,064**. They
fail that conservative certificate. Exact wording and that checkpoint spacing
are not, by themselves, declarations of MVP failure; preserve the user's gate
policy above. No final-weight semantic equivalence score or full-catalogue
training recovery is claimed. The separate 20-case meaning inspection used the
original update-1,000 parent: 14 clear material substitutions/omissions, six
scale-anchor cases unresolved, not independent review or approval.

**The post-run audit PASSES all 75 checkpoints**, complete-pass counts, exact
resume, the persisted final-pass limit and no pass-13 update. Initial 1,396
saved tensors, optimizer and RNG bytes match the parent; **1,338 non-pooler
state tensors remain byte-identical**, including **483 T/codec tensors**.
All **320 frozen base tensors/physical tying** pass; decoder/live hashes remain
unchanged; gradients finish None. **Three metadata tests pass in 0.005 s;
eleven CUDA optimizer/checkpoint tests pass in 0.587 s.** Unchanged numeric
matrices were reused. The stop control is additionally proven by the actual
endpoint/ledger. Only a final logging statement failed after all artifacts and
checks were saved; the one-line fix was verified against the saved report,
without rerunning any model workload. Actual process exits remain **143, 143, 1**.
The final `report.json` is `COMPLETED` with `USER_REQUESTED_FINAL_PASS`.

Raw evidence roots: `data/reference-coverage-20261106/{run,continued,final-pass}`.
The prefix preserves updates 1–100; the continuation preserves 101–4,800; the
final segment preserves 4,801–5,064, with no repeated completed updates.
Two intentional checkpoint interruptions may contain unlogged in-flight GPU
work; do not claim an exhaustive physical-compute total. Completed-window
accounting is **1,401,924 charged / 1,361,412 teacher-forced forward input
positions**. Three frozen feature-cache preparations total **10,128 W prefills /
393,765 input positions**; the 20-case meaning inspection is separate.
Median completed update: **2.452649227998336 s**. Final process time:
**1,515.7522246669978 s**; final-process peak CUDA: **4,194,480,128 bytes**.

Final unqualified frozen reference:
`data/reference-coverage-20261106/final-pass/reference-006064`, fingerprint
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.
Final complete resume bundle: `final-pass/checkpoint-006064` in the same parent.
It stores the user final-pass limit. **Do not start pass 13 or additional fitting.**

At that earlier endpoint, unfinished work was reference meaning/recovery transfer, particularly the
weak Question/Target/Reason roles, and distinguishing encoder/readout limitations
from actual receiver use. Coverage improves training recovery but has not
established held-out transfer, a unique cause, or connected Noul/Choice/Score
competence. No additional training, generation, deployment or live promotion is
queued. Preserve all raw evidence and the original failed reports.

## Previous completed exposure comparison

**The exposure-controlled reference comparison is complete. Extra exposure
repairs all four tested Question/Target training failures, with no diagnostic
regressions; development recovery remains 0/100. No job remains active.**
Implementation/protocol `ade8246` preceded the run. Read the
[measured report](reference-exposure-results-2026-11-06.md),
[portable receipt](evidence/reference-exposure-comparison-v1.json) and
[fixed comparison protocol](reference-exposure-comparison-v1.md).

The completed broad control was reused. The treatment restored its exact initial
weights, optimizer and RNG, then completed **1,000 pooler-only updates / 8,000
eligible presentations** on the RTX 3090. Exactly **336 same-role repeat slots**
were reallocated, preserving all **3,376 distinct views**, every original first
occurrence and every slot's role. Questions 0000/0010 increased from 1 each to
86 each; Targets 0001/0011 from 2 each to 85 each; Keys 0008/0018 stayed at 81.
All **49 original computation sources** matched, and all **35 unchanged initial
updates** reproduced control loss and gradient norm exactly.

| Metric | Original control | Treatment 900 | Treatment 1,000 |
|---|---:|---:|---:|
| Six training controls | 2/6 | **6/6** | **6/6** |
| Fixed first diagnostic set | 11/20 | **15/20** | **15/20** |
| Fixed separate screening set | 10/20 | 10/20 | 10/20 |
| Combined training diagnostics | 21/40 | **25/40** | **25/40** |
| Original development recovery | 0/100 at both | **0/100** | **0/100** |
| Original development contrasts | 100/100 at both | 100/100 | 100/100 |

Exactly the four targeted cases improve. All 21 baseline diagnostic recoveries
remain exact, with zero losses across the fixed 40 cases. Diagnostic baseline
is the old final 1,000-update control; development compares the corresponding
900/1,000 checkpoints. Both >=95%/90% development gates still fail. This is
evidence that targeted exposure allocation contributes to these training
failures, not qualification or a unique explanation of generalization failure.
Most low-exposure views were not boosted; neither broader retention nor a fully
balanced schedule was tested. Do not claim catastrophic forgetting: the old
broad control restarted from the common initial state, not the six-case probe.

**Three metadata tests passed in 0.001 s**, syntax checks passed, and the full
metadata/serialized-byte audit passed. All **1,396 initial tensors**, optimizer
and RNG bytes match the control. All eleven checkpoints retain **1,338 unchanged
non-pooler tensors**, including **483 T/codec state tensors**. All **320 frozen
base tensors/tying** pass; decoder/live identities are unchanged; all gradients
finish None. Existing numeric matrices were reused with unchanged computation
sources. No new source data/reviews, agents, external model calls, capability
updates, locked-label fitting, promotion or service changes occurred.

Training: **3,376 W feature prefills / 131,255 positions**, CUDA cache;
**246,970 charged positions / 238,970 teacher-forced forward input positions**.
Run time after metadata preparation **3,236.2155652950023 s**, median update
**2.550225070999659 s**, peak CUDA **4,193,888,256 bytes**. No arbitrary time cap.
Runner exit **0/COMPLETED**, `reference_qualified=false`. Raw evidence:
`data/reference-exposure-20261106/run/`, plus `audit.py`, `audit.json`, `fit.log`
and `command.json` in its parent. Final unqualified reference fingerprint:
`d741b65f85ef4d381c74bf48b3bf2330de3009f8e561e20d9e3bdc389ffdeb42`.

Next unfinished slice: **reference recovery/generalization under broad training
coverage**, preserving retention and original paired qualification. A broader
exposure schedule is supported as a next experiment to discuss; no additional
fitting, data generation or capability training is queued. Preserve all W,
RTX 3090, latent/cache, precision and original-tolerance constraints.

## Previous completed block search

**The arbitrary time cap is removed, and all 32 candidates plus all four
screening evaluations are complete. No finalist met the original improvement/
retention rule. No job remains active.** Implementation `0d6b1e9` adds result
resume and removes the deadline. `AGENTS.md` records the user's instruction to
keep local machinery simple, finish agreed finite work, reuse valid measurements
and avoid unrequested agent-chosen time/token/cost cutoffs.

Read the [completion report](reference-block-search-completion-results-2026-11-06.md),
[portable receipt](evidence/reference-block-search-completion-v1.json) and
[resume protocol](reference-block-search-resume-v1.md). The old timeout report
below remains historical evidence and no longer describes current coverage.

All **21 complete reports were reused byte-for-byte**, and mask 20 reused its
**nine completed greedy recoveries**. Its nine sensitivity forwards had to be
reconstructed because the old partial log omitted foil log probabilities; these
are now recorded. The remaining 11 candidate reports completed. Ranking over
all 32 selects **20, 1, 4**, and all received the original separate screening.

| Mask | Components | First set | Screening |
|---|---|---:|---:|
| 0 | Baseline | 11/20 | 10/20 |
| 20 | First attention block + lift | 12/20 | 7/20 |
| 1 | Queries | 11/20 | 10/20 |
| 4 | First attention block | 11/20 | 9/20 |

Mask 20 gains two first-set examples while losing one; it gains none on screening
and loses three. Mask 1 preserves but does not improve screening; mask 4 loses
one. None meets strict screening gain with preservation of every baseline
recovery in both sets. **NO_FINALIST** is a completed experimental outcome.
No development qualification was warranted, and no new development score,
locked-label use or live promotion occurred. Both diagnostic sets are historical
training data; they are not unseen during prior model training.

New work: **40 W prefills / 1,514 annotation positions**, **300 sensitivity
prefills**, **291 recovery episodes**, **7,807 decoder positions**. Two-run physical
totals: **729 sensitivity prefills**, **720 recovery episodes**, **19,346 decoder
positions**; logical complete coverage is **720 cases per metric / 19,252 decoder
positions**. The 94-position difference is the nine necessary sensitivity
reconstructions. No greedy recovery was repeated. Runtime **511.3701073270022 s**;
peak CUDA **3,622,856,704 bytes**; exit **2/NO_FINALIST**, `search_complete=true`.

**Seven metadata tests passed in 0.016 s** for resume/corruption/case identity and
existing selection rules. Unchanged prior GPU block checks were reused. All
320 frozen base tensors/tying, 483 T/codec tensors, live identity, source artifacts
and release manifests remain unchanged. Baseline restored; all gradients None;
zero agents, gradient/optimizer/capability updates or new source records. The
metadata audit verifies reuse, complete coverage, actual work and selection.
Raw evidence: `data/reference-block-search-completion-20261106/run/`; older run
and all evidence retained unchanged.

This block-replacement family and its screening are finished. Next unresolved
work is reference recovery/generalization and retention, including a causal
exposure-controlled comparison for the documented imbalance. No further search,
data generation, gradient fitting or capability training is queued. Preserve
W freezing, the RTX 3090-only rule, original >=95%/90% paired qualification and
latent/cache contracts. Runtime is an observation, not a stopping condition.

## Previous interrupted block search (historical; completed above)

**The diagnostic runner is implemented; the bounded block search stopped
INCOMPLETE at its fixed 600-second search limit. No job remains active.**
Implementation/protocol `87b01c0` preceded measurements. Read the
[measured report](reference-block-search-results-2026-11-06.md),
[portable receipt](evidence/reference-block-search-v1.json) and
[fixed protocol](reference-block-search-v1.md).

The runner tests five-component replacements from the saved 6/6 pooler into
original checkpoint 1,000, always restoring from immutable source tensors.
There are 32 masks including baseline/full donor; **21 completed**, mask 20
completed **9/20 cases**, and masks 21–30 were never started. No completed
configuration exceeded baseline **11/20** on the fixed training search set.
Full donor reproduces 6/6 controls but only **7/20** overall. Masks 18/19 gain
one Target while losing three other recoveries, finishing **9/20**. No shortlist,
second-set evaluation, finalist, new development qualification or live promotion
occurred. Do not call this complete search coverage or a recovery fix.

The two prepared 20-case sets cover all ten roles with disjoint annotation hashes
and families; both are historical training data. Prior exposures are a concrete
finding: control Questions **1 each** in the broad run vs **86 each** in the donor;
Targets **2 each** vs **85 each**; Keys **81 each** vs **85 each**. The broad run
restarted from the original parent, so this is not evidence of forgetting the
6/6 checkpoint. Baseline correct packets improve the desired token probability
in **15/20** paired checks; the first differing token is predicted correctly in
**14/20**. This supports partial packet use, not semantic competence.

**26 checks passed** (5 metadata, 2 CUDA exhaustive apply/restore, 19 related
regressions). All **320 base tensors/tying**, live identity, **483 T/codec tensors**,
parent artifacts and release manifests remain unchanged; original baseline was
restored. All owner gradients are None. Independent byte audit reconstructs all
**22 applied fingerprints**, verifies **55 source/protocol/test files**, case
separation and exact counters: **40 W prefills / 1,514 annotation positions**,
**429 sensitivity prefills**, **429 recovery episodes**, **11,539 decoder positions**.
Zero gradient/optimizer/capability updates, agents, external calls or new records.
Development and locked record labels were not loaded. Selection on labels is
explicitly data-driven weight optimization despite having no gradients.

Total runtime **691.9866396940015 s** including setup/final integrity checks;
median completed configuration **27.0861807830006 s**; peak CUDA **3,622,856,704
bytes**; exit **2/INCOMPLETE**. Raw evidence is
`data/reference-block-search-20261106/run/`, including all complete and partial
measurements and exact reproducible edit recipes. No further run is queued.
Next unfinished machinery is a cheaper initial screen, reserving full recovery
for shortlisted candidates, with a measured budget before another bounded run.
Partial mask 20, ten unstarted masks and second-set checks remain unmeasured.
Reference recovery/generalization, semantic binding and the full learned flow
remain unresolved. Preserve original >=95%/90% paired qualification, W freezing,
RTX 3090-only workloads and all existing scope restrictions.

## Previous single weight repair (historical)

**The one authorized weight-space repair completed and failed. No job remains
active.** Implementation/protocol `54a171a` preceded application and all semantic
feedback. Read the [measured report](reference-weight-repair-results-2026-11-06.md),
[portable receipt](evidence/reference-weight-repair-v1.json) and
[fixed protocol](reference-weight-repair-v1.md). One common-mode projection rule
was applied to both existing final reference checkpoints, with no strength/search
retry. Both were exported before development feedback. Calibration reused 71
reviewed training views, 71 W prefills and 142 pooler forwards, plus four checks
on cached states. No new records/reviews, agents, gradients, optimizer steps,
fine-tuning, capability updates or external model calls occurred.

Both repaired 900/1,000 exports remain **0/100 exact recovery and 100/100
contrasts**, matching the hash-verified parents; the production loader rejects
both. The original >=95%/90% paired criterion is unchanged. Only four lift
weight/bias tensors changed per artifact; all other **537/541 tensors**, including
**483 T/codec tensors**, are byte-identical. All **320 base tensors/tying**, live
identity, parent exports and three release manifests are unchanged. Locked labels
were not loaded. Projection residuals pass; recovery does not. **2 metadata tests,
2 RTX 3090 synthetic tests and 19 related regressions pass**; all 54 archived
source/protocol/test hashes and export hashes verify independently. Runtime
284.00155400899894 s; peak CUDA 3,622,856,704 bytes; exit 2/FAILED.

Raw evidence/exports: `data/reference-weight-repair-20261106/run/`. Repaired
fingerprints: 900 `41bb9e035d9036cb182d16f965039d64a720c9cc85e0f6dbd7cf61235a1c0f29`;
1,000 `e40c025b621a4c587f93811f4e22aa66c25a1fb9b0b06874212a7d2ca6c66ada`.
These are unqualified and never promoted. This candidate is retired; no follow-up
edit/fit is queued. Next unfinished is full-mixture reference recovery and
retention; semantic binding, observed correction materialization and full
participating-trainer memory remain separate. Keep W frozen and all existing
scope restrictions. Numeric/cache/runtime code and live services were unchanged.

## Previous encoder checkpoint (historical)

**The Target-private encoder block finished and failed qualification. No fit is
running. The user's requested recovery fix remains unfinished.** Implementation
`fb920fd` (after `7b5aaed`) is committed/pushed. Read the
[full measured report](reference-encoder-results-2026-11-06.md),
[portable receipt](evidence/reference-encoder-v2.json) and
[preregistered Target-private v2](reference-encoder-target-v2.md).

Target-only learning recovered both Target controls while regressing other
roles. The minimal private Target branch then passed **6/6 focused training
recoveries**, exact initial migration for all ten roles, 30 total queries,
private gradients and export/load. **Six GPU regressions passed in 15.560 s;
68 model-free checks passed in 2.811 s.** Pooler parameters are 4,213,760 versus
2,110,720 originally. Other nine roles remain shared; W/T/codec are not duplicated.
Five earlier candidate probes failed and are retired; seven probes used exactly
448 reference updates / 3,584 exposures and 96 separate latent-variable steps.
All raw sources/results remain under `data/reference-encoder-20261106/`.

The accepted architecture's full run reset to the original parent, not probe
weights. It completed **1,000 reference updates / 8,000 eligible exposures** over
all **3,376** existing reviewed views, exact sampler/owners, no failed exposures.
Every gate 100–1,000 remained **0/100 exact recovery, 100/100 contrasts**; both
900/1,000 are rejected by the real qualification loader. Initial checkpoint
save/restore succeeded. All 483 T/codec tensors are byte-identical to the parent
at all 11 checkpoints, live weights unchanged, 320 frozen tensors/tying pass,
all checkpoint/export/source hashes verify and all three release hashes are unchanged.
Mean first/last-100 CE is 0.5453864361013985 / 0.5070646067904454; median window
2.4458106085003237 s; run 3913.555469761999 s; peak 4193888256 bytes; exit 2/BLOCKED.
This is semantic failure, not unavailable hardware. Capability updates are zero.

Final export: `run/reference-1000/`, fingerprint
`6b90e7b706239374a088462777eb072cd166a055e353135b0058c583b9da1dd8`.
GPU readback verifies it and connected frozen-decoder input gradients, but the
full-data artifact recovers only **2/6 training controls**. The focused 6/6 gain
does not transfer to the full recipe. Do not certify the encoder/learned flow.
The lineage has 3,000 reference updates; physical reference work is 3,464 including
448 probes and the original discarded 16-update setup. Latent-variable steps
are separate (96 in current probes, 384 in earlier isolation).

Before the bounded repair above, the user asked an opinion on direct weight-space
programming. Primary research was checked and a **read-only** RTX 3090 weight audit performed;
no edit or external model/agent call in that read-only audit. Projection spectra do not show obvious
rank collapse or dominant bias. This does not rule out a targeted repair, nor
demonstrate one. W remains frozen. No additional fitting, generation, agents,
capability training, numeric matrix rerun or service changes are queued/performed.
Next unfinished is full-mixture reference recovery/generalization and retention;
keep the unchanged >=95%/90% final-two gate and locked-data prohibition.

## Prior recovery revision checkpoint

**The earlier recovery-v1 revision finished and failed.**
The user's requested recovery fix is **not proved**. Implementation `908d8b5`
reuses 3,376 existing reviewed role views with source hashes/field addresses,
exact coverage/resume and the Reason exception. A separately documented
[revision](reference-recovery-revision-v1.md) warm-started the old reference,
reset reference AdamW, and ablated preceding-target vectors in half the exposures.
See the [final measured report](reference-recovery-results-2026-11-06.md) and
[portable receipt](evidence/reference-recovery-v1.json).

Exactly **1,000 reference-only updates / 8,000 eligible exposures** completed on
the prescribed RTX 3090. All 3,376 views were used, one to 82 times; 4,000
exposures used each history mode. Every gate 100 through 1,000 remained
**0/100 exact recovery, 100/100 contrasts**. Both final reports are rejected by
the actual qualification loader. No promotion, live initialization or capability
update occurred. Live hashes match, all 320 frozen tensors/tying pass, all
checkpoint/export hashes and 49 archived source files verify, and all three
release hashes remain unchanged. Mean first/last-100 losses were
4.7712684374183665 / 1.1359519423067104; median window 3.4190447954897536 s;
complete runtime 4916.259019798017 s; peak CUDA allocation 4276769280 bytes;
runner exit **2/BLOCKED**. This is failed recovery, not unavailable hardware.

**68 model-free checks passed in 2.658 s; four GPU objective checks passed in
13.270 s; three focused catalogue checks passed in 0.044 s.** The final reference
fingerprint is `d0d4aef44b4415beb19311350eb24533190e1a27406c5d6f59a15a5a57402dec`.
Its lineage has 2,000 reference updates; total physical reference work is 2,016
including the previous discarded 16-update setup attempt. Do not confuse those
counts with this revision's 1,000 updates. Raw evidence is under
`data/reference-recovery-20261106/`, final export `run/reference-1000/`.

Read-only diagnosis supports conditional/template overfitting: the old reference
recovered 31/50 unique training meanings versus 0/50 development. Revised actual
reconstructions recover unrelated reservation/criterion/key meanings, often the
same text for different inputs. These are substantive errors, not paraphrases.
A 20-case attention probe does not support chat-header-only pooling. A read-only
persistent-latent conditioning intervention worsened CE and was not installed;
no extra fitting block followed it. Sparse Key/Midpoint training views still
contain only five meanings each. These findings do not prove recovery impossible
or establish that new data is necessary. Read the final report before selecting
any further revision; retain the unchanged >=95%/90% final-two gate.

The [bounded training-only isolation probe](reference-recovery-isolation-v1.md)
completed: original pooled vectors recover **1/6**, disposable optimized payloads
recover **6/6** by step 16 and at step 64, with W/pooler/codec/T/live weights
unchanged. There were **384 disposable latent-variable steps and zero model
optimizer updates**; do not add these to model-fitting counts. These are fitted
positive controls, not a trained encoder or held-out recoveries. For these six
training cases, the decoder/wire can express the annotations but the produced
representations fail. Next unfinished is a causal encoder/pooling learning
revision; held-out decoder generalization also remains unproved.
No source generation, agents, capability training, numeric matrix rerun or
service change was performed. Actual semantic qualification, predicted-packet
binding, observed correction materialization and full-trainer memory remain
unfinished.

## Prior CPU performance checkpoint

The user authorized fixing CPU dispatch overhead with **no generation, agents
or extra training**. Implemented the separately versioned
[graph-v1 recurrence](codec-recurrence-graph-v1.md), now the expanded backbone
default. It replays the original token-ordered CUDA operations and PyTorch VJP;
it preserves v3 BF16 storage / FP32 transformer / compensated BF16 vocabulary
arithmetic. Eager remains explicit and supplies the genuine reference. See the
[measured report](recurrence-graph-results-2026-11-06.md) and
[portable receipt](evidence/recurrence-graph-v1.json).

**45/45 focused full-model traces passed in 944.7234295289963 s**:
128/512/1024 positions × Noul/Choice/Score × eager/graph functional/graph
checkpoint/graph serving/genuine FP32. Graph states, logits, scalars, budgets,
discrete decisions and functional/checkpoint gradients were exact against eager.
All thirteen status cells occur. Genuine FP32 comparison worst logit/gradient
RMS errors were 4.169371270284753e-6 / 8.316676446703025e-6, within existing
tolerances. Live weights were unchanged; all 320 frozen tensors and physical
tying passed. This focused protocol does not claim a repeat of the entire
original 432-case matrix, full-trainer memory or learned semantic qualification.

**Eight operator tests passed in 3.745 s**, including 32 output/all-input-gradient
comparisons, 512/1024 connected compositions, cold checkpoint capture, LRU
eviction, stream isolation, three repeats, nonfinite guards and four small GPU
FP64 directional checks. **65 model-free regressions passed in 2.801 s.**
**Eight affected integration checks passed in 42.783 s**, covering teacher-forcing
parity, frozen projection/vocabulary derivatives, scoped T checkpoint replay,
export/tying and frozen-reference input gradients without live-owner gradients.

Measured existing trained-reference forward/VJP median: **1.4445 → 0.3463 s
(4.17×)**. Host `cudaLaunchKernel` calls fell **52,243 → 9,007**; graph replays
are separate submissions. Device kernels increased **52,497 → 58,905**.
The graph path added 72 replay submissions (9,079 total launch API calls).
The fix reduces host dispatch, not device kernel count. Two actual scheduled
eight-exposure windows with real finite guards/accumulation improved
**18.7282 → 5.8341 s (3.21×)** warm median. Buffers were discarded; optimizer
updates stayed zero and reference weights were unchanged. Cold/capture times
and all samples are retained. No full optimizer-step throughput is claimed.

An initial fused Triton candidate passed local tests but failed 512-position
serial-feedback FP32 tolerances. It was rejected; its source/logs remain under
`data/recurrence-fused-20261106/`. Accepted graph evidence is under
`data/recurrence-graph-20261106/`. No tolerances were widened, extra fitting was
performed, existing data changed or live service touched.

**Next unfinished remains reference generalization:** the previous fitted
reference still failed recovery. Performance is improved; no semantic promotion
was granted. Investigate the retained fit/existing data and document a development
revision before another fitting run. Full participating-trainer memory, actual
correction bindings and later semantic/promotion checks remain outstanding.

## Prior reference fit checkpoint

The user subsequently authorized **reference fitting and qualification with the
existing data on the RTX 3090**, no generation, agents or capability training.
The fixed fit is complete, but **qualification failed**. See the
[fit/performance report](reference-fit-results-2026-11-06.md) and
[portable receipt](evidence/reference-fit-v1.json).

`scripts/fit_reference.py` completed **1,000 reference-only updates / 8,000
eligible exposures**, every one of 500 IDs exactly 16 times, with zero failed
exposures. Those rows contain 50 unique training meanings. Both final checkpoints
900/1,000 achieved **0/100 exact recovery (100 unresolved), 100/100 contrast**;
all ten scheduled gates had those counts. The actual qualification loader rejects
both final reports. No reference promotion or live initialization occurred.
The blocker is now **failed held-out recovery, not a missing trained artifact**.
Mean loss over the first/last 100 windows was 2.658715932392515 /
0.046998847847844444. Live parameters remained byte-identical; frozen W passed
320-tensor equality and physical tying. Peak CUDA allocation was 3,627,365,888
bytes; elapsed 10,568.742410472012 s; runner exit 2/BLOCKED.

The causal teacher-forcing optimization passed **2/2 RTX 3090 checks in 12.574 s**
for serial/parallel loss and connected input/T gradients plus suffix causality.
Existing tolerances/arithmetic are unchanged. **63/63 model-free checks passed
in 2.799 s.** Checkpoint 0000 saved/restored before the corrected fit; all periodic
full checkpoints and frozen reference-only exports are retained. Final export
readback verified fingerprint
`fb38a31359a4ff367780bb8729695894e663810efd124c016bc2d5ba3fdbf7de`.
The first attempt's 16 updates were discarded after a checkpoint owner-name bug;
its log/evidence remains. They are additional physical work, not part of the
final artifact's 1,000-update lineage. No automatic extra epoch followed failure.

The user's suspected CPU bottleneck is supported by measurement: **99.39% of one
CPU core / 16% mean GPU use** during the fit. A subsequent derivative-only trace
of one existing episode counted **52,497 CUDA kernels**, 172.78 ms kernel union
within a 3.07 s instrumented episode (profiler overhead applies). Weights,
features and gradients were CUDA; no CPU model fallback occurred. Two warm VJPs
and one traced VJP added **zero optimizer updates**. Eager recurrent/backward
dispatch is the first performance target; accumulation/optimizer costs were not
included in that trace. No performance kernel/arithmetic change was made.

Raw evidence: `data/reference-fit-20261106/`; final export
`run/reference-1000/`; checkpoints/reports/source snapshot in `run/`.
Data hashes and live services remain unchanged. No capability training, new
data, agents or completed numeric matrix reruns occurred.

**Next unfinished:** investigate failed reference generalization from the retained
run/existing data, document a development revision before another fitting run,
and obtain a real passing final-two-checkpoint gate. Independently, reduce CPU
kernel-launch overhead only with appropriate numeric/gradient qualification.
Do not extend epochs, relax gates, tune on locked tests or start capability
training automatically. Full-trainer memory and later semantic/promotion checks
remain outstanding. Earlier missing-artifact/no-fitting statements below record
their historical checkpoint, not the current state.

## Previous checkpoints (historical)

The user requested the remaining reference qualification/binding work with
existing data, **no generation, agents or training**. The read-only runner,
qualification-evidence gate and conservative actual-content binding are now
implemented. See the [reference/binding report](reference-binding-results-2026-11-06.md)
and [portable receipt](evidence/reference-binding-v1.json).

**Actual learned reference qualification is BLOCKED.** No compatible trained
reference artifact was found among 107 local safetensors headers. The explicitly
untrained frozen reference recovered **0/100 exact canonical annotations** (all
100 unresolved, ten per role). The same tensor fingerprint separated **100/100
role contrast groups**, reused from 40 existing reviewed intervention pairs
across 39 families. No new contrasts/scenarios were authored or reviewed. The
missing standalone contrast manifest was resolved by materializing the existing
reviewed transformations and identical same-family positives. Contrast separation
alone cannot satisfy the >=95% recovery / >=90% contrast gate. No qualification
was granted and no fitting was performed.

**63/63 model-free checks passed in 2.647 s; nine distinct focused GPU checks
passed** (5/5 in 37.414 s, 3/3 in 53.287 s, and 3/3 in 18.618 s with two relevant
comparison reruns). Alignment/reconstruction retain input gradients through frozen
T without live-adapter/reference gradients in the isolated engineering fixture.
Binding checks actual role contents, source identity/states/coordinates, rubric
metadata and stable candidate identities, including reorderings. Outcome labels
require a measured qualified reference and current admitted snapshot; actual
return compatibility additionally gates continuations and later guided calls.
Unresolved learned deviations remain masked; a decoded gloss alone cannot approve
them. Explicit diagnostic reference episodes reject optimizer accumulation.

`scripts/qualify_reference.py` is read-only. Supply the pinned development release
and a frozen reference bundle to qualify; `--diagnostic-untrained` is explicitly
nonqualifying. `--contrast-only` avoids repeating completed RF measurements and
cannot grant qualification. Actual diagnostic/preflight exits were 2/BLOCKED.
The original setup rejections and exact recovery/contrast source snapshots are
retained. Existing annotation release hashes, services and prior numeric evidence
remain unchanged. Model-service calls, agents and optimizer updates were zero.

**Next unfinished:** a genuinely trained frozen reference artifact and its passing
qualification. That requires later fitting authorization; do not create one under
the current no-training instruction. Existing data already supplies the recovery
and applicable contrast cases. After qualification, enable supported producer
bindings and perform the outstanding full-trainer memory and semantic/promotion
checks. Observed correction materialization remains distinct. No further data
generation is queued. Earlier missing-runner/binding statements below are history.

The user authorized episode production and proof of latent Qev invocation using
existing data, **no generation, agents or training**. Implemented:
`training/producer.py` supplies native complete-episode callbacks;
`training/component_producer.py` materializes clean receiver/reader/decision replay
and selects reference-stage annotations. `Session.evaluate` rejects textual and
dictionary requests before any forward. Clean packets can execute their actual
preparation positions in the same Session. See the
[producer report](episode-producer-results-2026-11-05.md) and
[source/log-bound receipt](evidence/episode-producer-v1.json).

**57/57 model-free checks passed in 2.217 s; 11 distinct focused RTX 3090 GPU tests
passed.** Eight actual checks passed in a 77.136 s run that also had one incorrect
test-class selector (overall exit 1); the corrected selector plus a new free-window
guard then passed 2/2 in 15.833 s. A final tied-choice reader binding regression
passed 1/1 in 16.793 s. This runner error and the initial genuine
untrained Reason failure are retained. No numeric matrix was repeated.

Transport proofs use existing train-writer-0004/0000/0001: Noul/Choice/Score request
tensors have 103/143/110 positions, width 1024, FP32 CUDA. Embedding lookup and
tokenization are disabled during each call. Actual Qev outputs are delivered;
native losses have connected packet gradients, with one prefill/cache lineage,
exact accounting and no remaining holds. Optional calls have no textual decision
prompt. Initial human-text W+D prefill/continuous effort selection is preserved.
Positive fixtures prescribe writer controls and Reason child-count zero; clean
component tests use an untrained frozen reference. **This is transport/construction
evidence, not learned invocation, qualified reference semantics or training readiness.**
The unmodified Reason decoder still fails honestly with zero losses and no gold
repair. No optimizer step, reference fitting, data-service call, agent or service
change occurred; existing annotation release hashes remain unchanged.

Next unfinished: qualified reference recovery/contrast and verified binding of
actual predicted request meanings to supervision. Joint semantic/outcome and
post-return continuation losses stay masked; later annotated guidance stops at
the first unverified return. Writer semantic-loss integration, compatible serial
continuations and actual correction bindings remain unfinished. Generic unknown
candidate feasibility is masked rather than inventing a four-class label. Full
trainer-memory qualification still precedes training. Do not launch training or
regenerate data. Older missing-producer statements below are historical.

The user authorized the remaining data collections with cheap batched generation,
reuse, model content review and **no training**. Complete annotation releases
now exist under `data/training-collections-20261105/`:

- Training: **8,000** records; manifest
  `c70d4f44135a34c00e7fa0348833b1ec56613def15f5753231589492ed35478f`.
- Development: **1,000** base cases plus 40 secondary mechanical executions;
  manifest `5b8f4cffdc4f33e255d3a34306cac03a68c50dcd9dab905ae8d8adcf89c81e77`.
- Locked: **2,000** cases; manifest
  `d13594f18738056ee179b3498d54d8261f7f7654aad633fabb776810681aa1cf`.

See the [collection report](training-collections-results-2026-11-05.md) and
[portable receipt](evidence/training-collections-v1-summary.json).
`catalogue.json` identifies usable releases; `*-candidate-*` folders are history.
The pilot remains development-only. Content review is explicitly compositional:
actual source-state/canonical-definition reviews plus exact reviewed view recipes,
with per-record hash and transformation checks. It does not claim 11,000 separate
literal-row model judgments or independent scenarios.

Actual usage: **31** capped Muse calls, 965,889 input / 254,566 output tokens;
no agents, automatic retries, local model runs or training. **64/64 model-free
tests passed in 2.198s**. Independent audit accepts all releases, exact allocations
and family/status/domain minima, disjoint split families/input payloads, five
reference meanings per role, 30,302 exact source literals and 35 hidden-fact pairs.
Production sampler ID windows work in all five phases. One unsupported development
effort preference is masked and omitted from effort-scored free-job selection.

The 400 correction records contain prescribed intervention specifications;
correction losses stay masked until a producer binds an actual runtime packet/
snapshot. Replay is a generated preservation proxy, not recovered original data.
The producer slice above now implements token/source and episode materialization;
qualified reference artifacts, semantic binding and
the documented full-trainer memory qualification remain prerequisites. No
training, fitting, deployment, service changes or further generation is queued.
Earlier no-training-release statements below are historical.

The user explicitly authorized reviewing the existing 24 drafts and generating
the remaining 96 for the pilot, with no humans, agents or training. **Complete:**
`data/pilot-120-20261105/release/` loads as an approved 120-record development pilot
under `automated-content-v1`, with 20 per domain and all 13 operation/status cells.
See the [pilot report](pilot-120-results-2026-11-05.md) and
[portable receipt](evidence/pilot-120-v1-summary.json). Manifest SHA256:
`177147b46629f7ee2ff39b567fd90543621fba07fed6960bca123b6aa32b4c9d`.
All 120 final records bind actual separate outcome-blind model review; no human
review/certification is claimed. Existing 24 inputs and 125 original run files
remain unchanged. 14 capped Muse calls used 257,768 input/76,344 output tokens;
six generation, six initial reviews, one 21-case correction review and one
replacement review after a duplicate was caught. No automated retry loop.
Independent checks bind final inputs/outcomes to actual review bytes; 55/55
model-free tests passed in 1.334s. Draft validation now correctly allows unknown
nonselected candidates in assessed any-workable Choice while requiring selected
candidates to be known feasible. No numeric/model/GPU tests were rerun. No
training, service changes or deployment occurred. Earlier no-pilot statements
below are historical; do not regenerate this pilot.

The user explicitly removed mandatory human review for this experiment.
Current policy is mechanical checks plus one separate, outcome-blind
model-assisted content review per record (batched calls permitted), followed by
automated release approval. This applies to pilot, training, development, locked,
subjective Score/effort and new rubrics. No reviewer hiring, mandatory second pass
or human sign-off remains. `kind:model` review evidence binds actual model/session
and prompt/response assets; `kind:automated` approval binds policy and release.
Old human-review evidence is still readable but is not required. Unreviewed,
rejected, stale or quarantined records still reject. Do not automatically retry
generation/review or mislabel model review as human certification. This policy
change performed no model calls and approved no real dataset. Historical review
requirements below describe earlier checks, not the current policy. See the
[validation receipt](evidence/automated-content-review-v1.json) and
[current review procedure](data-operations.md#5-independent-semantic-and-outcome-review).

The user authorized small fixes for confirmed overengineering, excluding generation
orchestration/cost controls; no model calls are permitted for this cleanup.
Draft inspection now lives in `scripts/validate_drafts.py`, outside the training
package. It preserves source/hash, role/Reason and unreviewed-draft checks without
fixed counts, IDs, domain quotas, producer-name heuristics or run/release identity
equality. Run `.venv/bin/python -m scripts.validate_drafts <scenarios.jsonl>`.
Acceptance: arbitrary-size/ID/run batches validate, duplicate IDs reject and draft
records cannot pass release. `PYTHONPATH=tests .venv/bin/python -m unittest
test_drafts test_training_inputs -q`: **25/25 passed in 0.012 s**. Both saved
12-record batches also pass draft checking; **24/24 reject release**. No data
bytes, annotations or historical evidence receipts changed.

Pilot policy now accepts authored or model-proposed fictional roots without
changing the 120-family/20-per-domain allocation, two independent human reviews,
source/review/approval byte binding or split protections. Model origin/author and
assistance metadata remain intact; no real release was approved. Acceptance:
schema-only reviewed model roots load, missing provenance/second review rejects.
`PYTHONPATH=tests .venv/bin/python -m unittest test_training_release
test_training_inputs test_drafts -q`: **39/39 passed in 0.934 s**.

Release/sampler binding is now checked once against immutable catalogue/pool
contents. Records are snapshotted individually; windows decode only the eight
selected exposures, with fresh copies even for repeated IDs. Acceptance: mutation
of sampler contents rejects, forged initial catalogues still reject, resume is
unchanged, and two simulated windows decode exactly 16 records without rebuilding
the catalogue. `PYTHONPATH=tests .venv/bin/python -m unittest test_training_driver
test_training_release test_training_schedule -q`: **28/28 passed in 1.179 s**.
These are CPU metadata/counter simulations, not model or throughput measurements.

Cleanup code commits: `1e86c08` (draft tool), `3cfa11c` (pilot provenance),
`fa94776` (release windows). README/topic headers now defer current status to
this handoff; obsolete status/authorization banners were removed and older
checklists/audits labeled historical. Acceptance: all **610 local links across
58 Markdown files resolve**, and `git diff --check` passes. Historical measured
reports/receipts and normative numeric/cache/gradient contracts are preserved.
No model calls or GPU reruns were made. Optimizer rollback and finite guards
remain unchanged; their full-trainer memory/performance costs remain unmeasured.
Next unfinished work is still genuinely reviewed records and concrete episode
producers, followed by the already documented trainer-memory qualification.

The user then explicitly requested deletion of the old Muse framework. The
active local `muse-implementation` skill has been replaced with the cheap
single-call default; its four coordinator/runtime/tmux reference files and
reference symlink are deleted. The existing skill name/link still resolves.
Repository `AGENTS.md` now makes the same default persistent. See the
[removal receipt](evidence/muse-default-migration.json). Both generated batches,
raw responses, saved run scripts and pre-existing evidence were hash-checked
unchanged. No model call was made for this change. Historical launch scripts
must not be reused as the workflow. Further generation is not queued.

The user rejected the first batch's disproportionate generation overhead and
requested a fresh batch with cost saving. The replacement is complete in
`data/muse-economy-12/drafts/`: **12 drafts in one Muse completion**, no retries,
tools, children or reminders; 29.969 seconds, 16,293 input / 4,637 output tokens.
See [bounded-generation evidence](muse-economy-12.md). Fifteen model-free tests
pass; all twelve records pass draft validation and reject approved release.
Original response and explicit Codex syntax/content repairs are retained.
This batch is also unreviewed model-proposed development data, not a training
release. Prefer this bounded single-call workflow for small draft requests;
do not restart the rejected coordinator/worker/observer workflow. The old batch
and its evidence remain historical, unchanged. No further generation is queued.

Earlier, the user explicitly authorized Muse setup repair and **12 draft
scenarios**. This bounded batch is complete: two per domain, 17 source files,
all development/model_proposal, zero human reviews and no approvals. See the
[Muse draft report](muse-draft-pilot-2026-11-05.md) and
[portable evidence](evidence/muse-draft-pilot-v1-summary.json). Local data:
`data/muse-pilot-20261105T191615Z/drafts/`; open `review-inputs.md` before the separate
`outcome-proposals.md` for outcome-blind review. This user authorization does not
start bulk generation, capability fitting or deployment. Muse has exited cleanly
and all run-owned workers/commands/tmux resources are released.

Backend implementation slices are committed and pushed directly to `main`.
The [slice report](backend-slices-2026-11-05.md) records acceptance conditions,
results and limitations. No capability/reference fitting, bulk semantic data,
frontend integration or deployment was performed. The Operating Table and other
live services remain unchanged.

Historical release-boundary implementation commit: `4d96af1`.
The prior numeric/finite-guard evidence remains at `abd8dcf`, followed by the
original final handoff at `2da8e43`. The new
[release-boundary slices](backend-release-slices-2026-11-05.md) implement actual
source/annotation/review/approval byte checks, immutable release snapshots,
pilot/driver/sampler binding, exact resumed exposure cursors and a read-only
loader for the existing numeric evidence. No real reviewed pilot was created.

The original **324/324** v3 cases at 128/256/512 remain passed and were not rerun.
The separate **108/108** `v3-1024-v1` matrix passed in **5598.540981531143 s**:
one prefill/cache lineage, connected gradients, exact accounting, unchanged live
parameters and frozen/tied storage. Worst state RMS was 0, logit RMS
`3.12525587881255e-06`, and gradient RMS `3.848441887921371e-06`;
peak allocation was 13,093,139,968 bytes. See the
[1024 measured report](codec-1024-results-2026-11-05.md).

The matrix retains its exact source snapshot from before the subsequent finite
validation guards. Those guards reject overflow before clamp/argmax or dispatch;
finite arithmetic, weights, fixtures and tolerances are unchanged. The separate
post-guard 44-test GPU run passed. Neither result silently substitutes for the
other's source provenance.

The complete learned decision capability remains untrained/unpromoted. Prescribed
operations/statuses, diagnostic weights, fixed continuations and engineering
reference fixtures do not establish invocation, semantic understanding or answer
quality. The old BF16 transformer arithmetic still fails; do not relabel it.

## Preserve these requirements

- One human prompt, one prefill, one continuous hybrid cache and a normal answer
  from that same job. D for initial effort, R for native work, D for internally
  requested Noul/Choice/Score; no keyword/domain routing or second decision job.
- Textless typed requests/results, immutable bindings and selected contents.
  No derivative through argmax; task scalars and selected continuous contents
  retain connected gradients. Startup budget feedback alone is detached.
- Initial effort is continuous scalar regression in its own real LM-head row.
  Preserve the ordered CUDA FP64 S-curve `floor((3*s*s-2*s*s*s)*131072+0.5)`;
  context 262144, separate answer/thinking allowances, cancellation and closure.
- Qualified v3: BF16 frozen storage, FP32 transformer/LoRA computation and causal
  state, compensated BF16 vocabulary projection, genuine FP32 reference.
  Original BF16 transformer arithmetic remains a preserved failed profile.
- Model workloads only on RTX 3090 UUID
  `GPU-225509c3-d702-6c27-de03-76b5e99b33b1`. Never CPU/RTX3060 fallback. Driver
  access required execution outside the sandbox here; diagnose access/device
  selection before declaring GPU unavailable. Do not kill unrelated services.
- Keep The Operating Table's snapshot read-only and its services/artifacts intact.
  Diagnostic checkpoints must never replace live capability weights.
- Commit authorized work directly to `main` and push, per AGENTS.md. No PR/branch
  or repeated permission request is needed. Deployment needs separate authority.

## Completed slices and evidence

1. **Contract/code review and versioned qualification.** Direct implementing-agent
   review of codec/v3 and measured reports; no independent Astra approval claimed.
   The runner now distinguishes the 324-case `v3-512` protocol from the 108-case
   `v3-1024-v1` extension and rejects incomplete/duplicate/foreign/failed coverage.
   Direct review additionally found nonfinite Score clamping, gate acceptance
   and norm overflow. Pre-fix regressions reproduced the defects; finite guards
   now reject them, including terminating invalid gates without dispatch.
   No finite codec arithmetic, fixture statuses, reference or tolerances changed.
2. **F7 inputs/provenance.** Strict per-role/per-branch whitelists, transitive
   dependency checks, byte/hash/version/source coordinates, exposure flags,
   family/split and review-manifest gates. Current labels stay outside request
   encoders; non-assessed Reason has its own role/status exception. Mechanical
   checks cannot establish honest annotation or authenticate human reviews.
3. **F3 clean construction.** Fresh W+D derived source contexts; frozen W/reference
   semantic contexts; independent registries/local identities; complete clean
   packets through production rendering/admission; grounded exact kinds and
   absence/empty distinction. Parent hashes/coordinates remain external audit.
   Hidden-fact twins have identical complete request/return tensors. Unverified
   overlapping token offsets reject before encoding rather than being invented.
4. **Losses and optimization.** A4 subtype losses/masks, actual-snapshot binding,
   per-role semantic normalization, per-component CUDA VJPs/eligible-example
   denominators, eight scheduled exposures, one combined clip/step, unused
   `grad=None`, transactional rollback and named unique owners. Scalar rows are
   isolated. Tiny synthetic optimizer tests ran; real-model optimizer steps=0.
5. **Reference/live and scheduling.** Independent pooler/codec/T sharing frozen W;
   reference objective and frozen decoder preserve input gradients. Reference
   checkpoint replay reinstalls the scoped T mapping; reconstruction shares one
   frozen vocabulary slice per episode. Deterministic A4 sampler, replay recipes,
   prefix guidance, complete-session collection and guarded window driver exist.
6. **Export/resume.** Versioned small-weight bundles, actual reference tensor
   fingerprints, hashes, private effective rows, optimizer moments, phase/sampler/
   RNG state, strict v3/tied-W restore and exact next-update restoration tests.

Final focused GPU suite: **44/44 passed in 22.329 s**, comprising 18 existing
codec/precision, 6 finite-guard and 20 backend tests. Model-free: 10 input/provenance, 4 schedule, 2 matrix coverage and 6
existing exact codec tests pass. Exact commands/timings/log hashes are in the
[slice report](backend-slices-2026-11-05.md) and
[portable validation](evidence/backend-validation-2026-11-05.json).
Two concurrent reference-objective attempts hit GPU OOM; sharing the identical
vocabulary slice resolved the auxiliary retention defect and the affected GPU
suite passed. No numeric matrix restart, tolerance relaxation or CPU fallback.

Continuation validation: **42/42 model-free tests passed in 0.950 s**; after the
final numeric Boolean-marker guard, the **14/14 affected numeric/driver tests
passed in 0.312 s**. The new evidence loader separately validated the existing
108-case aggregate, three child reports and 28 archived source/test files.
Release/driver/qualification preflight imports no torch. These are metadata/schema
checks, not new GPU/model runs. Tests use explicit temporary schema fixtures;
their human/pass markers are never actual review or measured-model evidence.
Exact log/test source hashes are in the
[portable continuation validation](evidence/backend-release-validation-2026-11-05.json).

## Next unfinished slice / blockers

**Reference recovery still fails after the Target-private v2 block.** Use
`data/reference-encoder-20261106/run/` and the current result above. A focused
6/6 training gain fell to 2/6 controls under the separately reset full-data run,
while all 100 held-out cases still fail. Full-mixture generalization/retention,
encoder/decoder behavior and any mechanistic weight repair remain unfinished.
No further fit is running. Do not rerun/extend failed blocks, relax gates, repair
from locked labels, invent semantic approvals or initialize live weights from
these unqualified artifacts. The weight-space question did not authorize a
change to the frozen base; no programmed weight repair has been attempted.

The dispatch performance slice is now implemented and qualified through the
separate graph-v1 protocol above. Keep its source-bound evidence distinct from
the historical eager 324/108 matrices. Capture overhead, dynamic shapes, full
optimizer-step throughput and full participating-trainer memory are not waived
by the reported warm speedups. Preserve all original tolerances and exact
cache/accounting/gradient invariants in subsequent changes.

The read-only qualification runner, reference formation objectives, actual-content
and compatible-return gates, and native/clean complete-episode producers exist.
Qualified reference evidence is still required to initialize live weights and
supervise actual learned deviations. Supported exact fixtures are construction
proofs, not learned semantic flow. Arbitrary predicted slots, observed correction
materialization, full-trainer memory and semantic/promotion evidence remain
unfinished. No capability training is authorized by the reference-fitting request.

Before capability training, qualify the full A4 mix of participating component
buffers, optimizer moments and auxiliary graphs at applicable lengths. The 1024
numeric matrix's conservative memory envelope is not that qualification.
Later phases must retain live optimizer moments, preserve final-two-checkpoint
promotion, and use actual returns without gold repairs. Deployment and live
service changes require separate authorization. No data generation is queued.

## Entry points and artifacts

| Location | Purpose |
|---|---|
| `codec/` | Existing typed runtime, session, v3 arithmetic and qualification |
| `training/inputs.py`, `clean.py`, `clean_packets.py` | F7 builders/provenance and fresh F3 artifacts/packets |
| `training/producer.py`, `component_producer.py` | Native/clean complete-episode callbacks and existing annotation materialization; no auto-launch |
| `training/reference_qualification.py`, `binding.py`, `scripts/qualify_reference.py` | Read-only recovery/contrast gate, qualified-reference evidence and actual-content/return binding |
| `training/release.py`, `qualification.py` | Content-checked release/pilot and existing numeric evidence ingestion; no model execution |
| `training/losses.py`, `episodes.py` | Losses/masks and actual snapshot/session guards |
| `training/engine.py`, `profile.py`, `sampler.py`, `driver.py` | CUDA component buffers/owners, scheduling and producer orchestration |
| `training/reference.py`, `checkpoint.py`, `manifest.py` | Independent reference/T, small-weight export/resume and identity pinning |
| Legacy `train.py`, `train_feedback.py`, `curriculum.py` | Prototype material; not the complete trainer or reviewed data |
| Legacy `server.py`, `thinking.py`, `static/app.js` | Unchanged migration targets; no deployment occurred |

Local inputs, never committed:

- Base revision `2fc06364715b967f1860aea9cf38778875588b17`, read-only snapshot at
  `/home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17`.
  Weights SHA256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
- Qev revision `1d2a6d7f891daab7483a49c3abe507c0c9d7ea62`, local `data/qev/`.
- Frozen diagnostic child `data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors`,
  SHA256 `1dd2e8527e14caa9002d672b4187253cdc174fb3c204ab56e80f257f23943640`.
- Prior 324-case raw bundle: `data/codec-qualification/feedback-v3-20261105/`;
  [portable evidence](evidence/codec-feedback-v3-summary.json). Do not rerun for
  documentation-only or unrelated training-package changes.
- New numeric raw bundle: `data/codec-qualification/1024-v1-20261105/`.
- Backend logs: `data/backend-slices-20261105/`; continuation logs:
  `data/backend-release-20261105/`. Current source/contract identities:
  [continuation backend manifest](evidence/backend-profile-release-v1.json).
  The [prior backend manifest](evidence/backend-profile-v1.json) remains unchanged
  for the earlier slice evidence.

Missing pinned artifacts block model execution; do not download another model,
refit the diagnostic checkpoint or use CPU. The 1024 command is in its
[protocol](codec-1024-protocol-v1.md); always use a new output directory for an
affected rerun. No completed numeric matrix needs repeating for these docs.
