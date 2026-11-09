# RD-08 format recovery and matched comparison — 2026-11-08

The explicitly authorized v2 format now represents **all1,500 existing training
examples**, with complete roles, unchanged data and no cropping. Both arms have
full shared training support. Development coverage is E127/130 and X180/180;
three oversized E requests remain explicit failures under request256. This
removes the training coverage blocker; it does not establish useful learning.
The user reopened GPU work after the historical interruption. Performance and
all19 affected GPU checks now pass and were committed/pushed before recovery.
Control is complete:750 retained updates, all1500×4/6000 presentations and full
initial/final1500 observations, with W/R/startup/reference frozen. Candidate
passed375 of750 updates with no rejected exposures; shared initialization,
Adam moments and CUDA/Python RNG match exactly. Its full final panel and both
development evaluations remain due. Useful improvement is not yet evaluated.
RD09 remains unstarted. See the [performance/recovery update](receiver-performance-v1.md)
and [current ledger](redesign-progress.md). Historical599 logged/500 recoverable
control updates and99 lost/replayed windows remain separately recorded in the
[pause evidence](evidence/receiver-matched-v2-pause-2026-11-09.json).

Contract committed/pushed before implementation: `c1021a3`,
[compact-role-states-v2.md](compact-role-states-v2.md).
The full BPE audit, field-mapping parity, common-support proof, qualification and
finite protocol are under `data/redesign-format-v2-20261108/`.
Portable [coverage](evidence/compact-role-layout-coverage-v2.json) and
[GPU qualification](evidence/compact-role-qualification-v2.json) pin exact bytes.

Both matched arms use compact bound returns: metadata versions3/4 bind the
original immutable question/target/bank/rubric and selected candidate identities
without replaying their vectors. Assessed returns cost3; maximum Reason return39.
Requests still carry every original role vector and all source evidence. The
candidate's explicit field capacity is63 states, covering observed maxima49 in
training/50 in E; the control retains the original pooled roles. Request256,
return64, source-span32/count8, writer events, handles/records, context/budget,
semantic grading and numeric tolerances do not increase. Old profile bytes and
limits remain available. The native33-way length head is unchanged/frozen;
native v2 writer/reader learning is unsupported and is not part of this task.

The new audit checks all1,810 training/E/X records with the actual Rust BPE/Jinja
framing and no model imports. Every original field input hash, mapping hash,
length, record/family/operation identity matches the historical v1 audit exactly.
Pooled/direct maximum training requests are124/256, respectively. All1,500 rows
remain in each arm's identical ordered catalogue; no common-support subset was
substituted. The three E requests of270/276/295 remain in evaluation denominators.

Technical qualification: **pass**,18 IDs including the cache-restore check below.
The original17 core IDs remain valid. The ten predecessor GPU checks are
refreshed and seven v2 checks cover actual reviewed Noul/Choice/Score inputs,
joint VJPs/eligible accumulation, all13 operation/status cells for both formats,
exact256 accepted/257 rejected requests, maximum39-position returns, eight
cancellation scenarios, FP64 projection reference, complete-window owner/moment/
RNG/sampler resume and24 long traces through1024. Eighteen paired comparisons
cover functional/checkpoint/serving/explicit-FP32. All320 W tensors, physical
tying and reference identity pass at every class cleanup; optimizer steps are
blocked during qualification and actual updates are0. Engineering forcing is
disclosed; no learned native capability is certified.

The initial qualification attempt has15 passes and one real pooled-Score binding
failure. The admission check normalized rubric fields separately while the
constructor normalizes their concatenation. Reproducing that exact reduction
shape repairs byte-exact binding without relaxing tolerances. The targeted
final-source rerun has3 passes: actual inputs/endpoint delivery, v2 resume and
both formats' exact eight-example accumulation. The24 passing long traces and
all-status/cancellation/projection methods are AST-identical and reused; the
qualification manifest lists the precise source migration and retains failures.
Unchanged324/108 kernel matrices remain historical evidence, not a new-interface
certificate. No W/reference or optimizer/numeric kernel was changed.

Final model-free suite: **103 tests pass**,137.464 s. Earlier failures from a
literal expected position count and stale historical-plan expectations were
corrected; initial logs remain. Historical source-bound v1 plans reject after
source changes; old evidence was not relabeled current. Compilation and staged
whitespace checks precede execution. The existing receiver probe loop now consumes
the exact prepared order for the finite comparison; no copied training loop or
worker framework was introduced. Initial/final training measurements record
actual decoded/delivered behavior as well as heads using the same Session.

The committed finite protocol uses the same frozen-W source policy, exact parent
checkpoint0188 shared masters/Adam moments/RNG/schedule, seed, joint coefficients,
owner names and6000 ordered presentations per arm: four full passes,750
eight-example windows crossing pass boundaries without padding. New candidate
projection parameters are FP32 zero-delta identity with fresh absent moments;
writer length parameters stay frozen. W/R/startup row/reference remain frozen.
Both fixed fits must finish before either development evaluation. There is no
development checkpoint selection, extra pass, new data/review, locked access or
promotion/deployment. Initial/final1500-case measurements are observations and
never enter gradient exposure counts. Complete-window checkpoints use the
existing transactional save/resume implementation.

Useful-improvement acceptance remains the original predeclared familiar-cohort
Noul/baseline and Score/interior requirements, operation/status retention, and
all previously passing Choice cases in the fixed50-case panel. The existing
historical probe records47 such passing Choice IDs; their case/raw hashes are
pinned before fitting. Full case-level wins/losses, both-sides-correct X pairs,
all E/X cohorts, unsupported Reason semantics, failed/oversized cases and startup
retention will be reported. A completed failed comparison is an execution result,
not authorization to retry or proceed to RD-09.

Execution and useful-improvement outcome are tracked separately. The original
pre-execution implementation commit preceded all updates. Current actual progress
and its user-requested pause are recorded below; this is not a completed comparison.

### First-update recovery

Control baseline completed all1500 actual head/delivery observations. Its first
update and exact model/Adam/RNG round-trip passed, with checkpoint0189/cursor8
saved. The next input then failed a stale reference version-counter guard:
restoring identical frozen tensors advances PyTorch counters. No reference bytes
changed. The guard now accepts that metadata change only after full checksum
validation against the originally bound immutable artifact and frozen ownership;
changed content/artifact/ownership still rejects. Payload/tokenizer/codec/model/
loss/optimizer arithmetic is unchanged by AST/source proof. A targeted real3090
checkpoint/cache test passes (32.961 s): cached heads/losses exact, cache hits2,
W320/tying/reference pass,0 optimizer steps. Three model-free compact tests pass.
Qualification now has18 resolved IDs; original17 core results remain valid.

The first update, eight exposures, full1500-case baseline and original failed log
are preserved. The explicit metadata-only migration in
`data/redesign-format-v2-20261108/control-migration.json` copies weights, optimizer,
effective rows and RNG byte-for-byte; only qualified source/protocol bindings
change. All original files, owner shapes, objectives, schedule, full6000 order and
sampler receipts remain. New plans are `plans-cache/{control,candidate}/plan.json`.
The resumed control starts at cursor8 with749 windows remaining; it does not
repeat the completed update or baseline. New output is `control-resumed/`; its
initial panel is an unchanged copy of the original observation file, retaining
original source provenance. Candidate remains unstarted until control finishes.
No additional training exposure or new data is introduced by this recovery.

### User GPU hold and preserved progress — 2026-11-09

At01:01:37 UTC, the verified control PID618710 was paused with SIGSTOP after the
user reported that the GPU must be turned off now. Its process state is T
(stopped); no more model work or device probes are scheduled. Existing services
and devices were not changed. Pausing retains process/VRAM state and does not
power off the card or create a new checkpoint. No automatic resume.

Actual control progress is599 completed windows/updates and4792 presentations
across all1500 records, with zero rejected exposures. This includes the preserved
first update and598 resumed updates. All1500 original initial head/delivery
observations remain byte-identical. Initial shared receiver/moments/CUDA/Python
RNG receipts match the original prefix. Final measurements and the candidate
fit have not started; neither E/X evaluation nor RD09 has run. There is no
useful-improvement result or accepted receiver. RD-08 remains in progress.

The latest complete disk checkpoint is `control-resumed/checkpoint-0688`, manifest
SHA `2eaea42f7bbbbc0a125c51f0b06016afbda2fa1011c913a05c912202269d91aa`.
All four weight/effective-row/optimizer/RNG file hashes verify without model
imports. Historical counter688 corresponds to500 new RD-08 updates and cursor4000.
If the process or GPU memory is destroyed,99 completed updates/792 presentations
plus any partial window600 are not recoverable from those files. The99 completed
windows took3848.013 seconds (64.13 minutes). Their telemetry remains, but logs
cannot reconstruct their exact missing optimizer/RNG state. If the stopped
process survives, its live state can continue after GPU permission returns.
Preserve both paths and never claim that checkpoint0688 contains599 updates.

The user's utilization concern prompted a read-only process/source/log check.
Trainer CPU was122% (about1.2 cores), while the12-logical-CPU machine was9.54%
busy over a two-second sample. Three per-process GPU samples were10%,1%,17%;
these are sparse observations, not a training-wide average. The last125 windows
averaged38.817 seconds and14.272 separate component gradient passes; their
runtime/count correlation was0.826. The source still performs separate component
backwards and per-example receiver fingerprints with GPU-to-host readbacks.
No live stack/timeline profiler was attached, so no cost share or speedup is
claimed. No numeric setting, optimizer, batching, source or schedule was changed.

Portable [pause evidence](evidence/receiver-matched-v2-pause-2026-11-09.json) pins
the paused report, training rows, baseline, log, migration and performance
observation. The raw report retains RUNNING as its last emitted state; the
separate pause receipt records the actual stopped process. Resume requires the
user to reopen GPU work, with no new data, duplicate fit, extra pass or RD09.
