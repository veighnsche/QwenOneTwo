# Shared CUDA context and frozen weights — diagnostic contract

Scope: existing eight-case CPU-profile selection on the RTX 3090, frozen
checkpoint-0188, no training/data generation, no service changes. This tests host
cost and isolation. It cannot certify learned invocation or reference semantics.

`scripts/benchmark_shared_context.py --execute-gpu --snapshot SNAPSHOT
--output data/RUN --mode serial|replicas|shared --repetitions 3` is an explicit,
opt-in diagnostic. Normal training, serving and two-GPU defaults do not change.
It never targets the 3060. A missing device, failed receipt or unwarmed graph
stops the run; there is no fallback, automatic retry or elapsed-time cutoff.

## Ownership and capture

- `serial`: one process, one lane. `replicas`: two processes, one lane each.
  `shared`: one process, two Python threads and two CUDA streams.
- Shared lanes have separate module objects, parameter/buffer dictionaries,
  bound forward methods, precision owners, tokenizer state, evaluation tapes,
  reference memo caches, Sessions, registries and causal caches. They share the
  exact frozen Parameter/buffer objects, including the existing embedding tie.
  Reference and live parameters remain disjoint. These views have no optimizer
  ownership and must never be used for training/export or mutable serving.
- Reference `functional_call` replaces entries in a lane's own module dictionaries;
  sharing a backbone module directly would race reference and live execution.
  Dynamic RoPE is rejected; the current default RoPE reads shared buffers.
- Initialize all lane streams first. Under a per-process lock, each lane warms
  all eleven distinct existing cases (three anchors plus eight main cases).
  The original recurrence cache retains 32 entries per stream/lane, keyed by the
  unchanged device/stream/shape/dtype key, with its original replay lock.
- Then forbid new captures process-wide with `recurrence.replay_only()`. Clear
  each lane's reference memo cache and concurrently repeat all eleven cases.
  This stress phase exercises actual reference calls and replenishes the memo
  caches before timing. Missing forward/backward captures fail explicitly.
  CUDA graph equations, precision, finite guards and derivatives are unchanged.
- Main work is the existing seven whole jobs, including the intact two-case X
  pair, repeated three times: 24 presentations, eight distinct main cases.
  Completion-driven workers consume one shared queue. A case remains within
  one lane, preserving one prefill, cache lineage, bindings and accounting.
- Synchronize only the owning stream at case/job completion. Memory statistics
  are process-wide and collected outside lane activity. All workers finish the
  main queue before any final integrity check starts.

## Acceptance and measurements

Every warm, stress and main receipt must pass the existing independent
`validate_case` checks. Every repeat of an ID must match exactly on this GPU.
Re-audit saved receipts against the CPU-04 3090 baseline; preserve all outcomes,
including existing semantic failures. Require exact job/presentation coverage,
initial/final base hashing and tying, receiver/reference fingerprints, frozen
parameters and `grad=None`, zero optimizer updates, successful child exits.
The shared view validator requires identical tensor objects/storage and private
module containers both before execution and after all lanes join.

Three alternating replica/shared pairs measure the same queue and per-lane
warmup/stress workload. A serial control reports the cost of one lane. Main wall
time starts at parent release and ends after validation/journaling of the last
main result. Main CPU sums each whole process's main window plus parent main
CPU; this includes Python/native helper threads without double-counting
overlapping lane CPU. Per-job thread CPU is diagnostic only, never added to the
process total. Setup-inclusive CPU/wall are separate. CPU time between spawn
and the child's entrypoint is not included; reports are not whole-system energy
measurements. IPC result handling and scheduling are included in the main window.
Parent end-to-end timing begins after plan creation, before loading the release
and starting children, and ends after child joins, before report encoding/write.
The child main CPU cutoff is after its lane threads join; asynchronous Queue
feeder work may trail that cutoff. Parent main CPU ends after admitting the last
case. Child total CPU ends before queuing its final `done` message. These are
declared process intervals, not an exact attribution of every IPC instruction.

Process allocator peaks are not synchronized physical-device peaks. Report
aggregate allocation/reservation with that limitation, and do not multiply gains
by previous experiments. A slower or CPU-heavier result is retained as a result;
shared execution is not promoted merely because its weight sharing is correct.

Reuse the existing 324-case and 108-case numeric qualification for unchanged
math. Focused 3090 checks cover the new replay-only barrier, exact forward and
connected gradients, checkpoint rematerialization, stream isolation/eviction and
nonfinite rejection. Passing this bounded diagnostic does not qualify the full
310-case suite, arbitrary shape capture, gradient training or long-context flow.

Raw plans bind all backend source hashes plus the benchmark script hash. Keep
JSONL events, reports, failed receipts, source snapshots and exact commands under
`data/cpu-07-20261107/`; portable results belong in the measured report/evidence.
