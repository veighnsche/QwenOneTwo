# Next-agent handoff — 2026-11-07

Read this first for current status, then the
[implementation specification](architecture-and-training-plan.md) and its linked
topic contracts. Historical reports describe their recorded sources and runs.
This handoff records measured mechanisms, not learned capability.

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

## Current checkpoint — architecture slices 1–2 complete, no training

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
