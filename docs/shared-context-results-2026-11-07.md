# Shared context and frozen weights — 2026-11-07

**Shared execution works with exact receipts, but does not reduce CPU work per
case.** Three alternating 3090 comparisons increase main CPU/case by **8.2258%**
at the medians and reduce main throughput by **15.4537%**. It reduces resident
allocation **46.7664%** and setup-inclusive process CPU **21.6540%**. Retain the
runner as an opt-in diagnostic; keep normal execution defaults unchanged.

[Contract](shared-context-diagnostic-v1.md),
[portable evidence](evidence/shared-context-diagnostic-v1.json).
No agents, training, new semantic data/reviews or service changes. Base W,
receiver and reference remain frozen. Only the explicitly selected RTX 3090
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1` executes model work. QwenOneTwo server
PID 2094 and The Operating Table remain untouched.

## Implemented slice and acceptance

`training/shared_frozen.py` constructs private module topologies with the exact
same **1,719 registered frozen tensors, 1,656,951,700 bytes**, rather than another
weight copy. Parameter/buffer dictionaries, bound methods, precision owners,
tokenizers, evaluation tapes and reference memo caches are private. Embedding
tying is preserved; reference/live parameters remain disjoint. Dynamic RoPE is
rejected. Each case still owns its Session, registry and causal cache.

`scripts/benchmark_shared_context.py` compares one serial lane, two process
replicas, and two thread/stream lanes in one process/CUDA context. It reuses the
existing pinned plan, loaders, evaluator and admission checks. There is no new
service, generalized scheduler, retry loop or elapsed-time cap.

The existing recurrence lock, stream/device/shape/dtype keys and equations stay
intact. Each lane retains 32 cache entries; shared mode therefore has a bounded
64-entry process cache. All streams initialize before sequential warmup. A new
process-wide `replay_only()` barrier rejects unwarmed forward/backward captures
while lanes run concurrently. Main work cannot silently capture a graph or use
alternate arithmetic. All workers finish before final weight checks begin.

Acceptance requires exact frozen tensor sharing with private mutable controls,
actual concurrent cold-reference execution, independent receipt admission and
exact baseline agreement, unchanged numeric/gradient checks, frozen identities
and zero updates. Those conditions pass. Performance acceptance as a reduction
in CPU/case fails; correctness alone does not establish the requested speedup.

## Controlled measurements

Each run warms all eleven existing cases per lane, then clears private reference
memo caches and concurrently repeats those eleven cases under the capture
barrier. This exercises actual reference `functional_call`, including temporarily
zeroed live-adapter mappings. Main timing follows with populated reference
caches and the same eight-case queue repeated three times: **24 presentations,
eight distinct main cases**. Whole X pairs remain atomic.

Run order: replicas/shared, shared/replicas, replicas/shared, then one serial
control. All seven exits are zero. The following are separate mode medians;
serial has one sample. They are not directly comparable with the previous pool
report's shorter warmup, two-card timings or early final-check overlap.

| Metric | Two replicas | Shared context, two lanes | Serial control |
|---|---:|---:|---:|
| Main wall, seconds | 15.933525 | 18.845921 | 26.031849 |
| Main cases/second | 1.506258 | 1.273485 | 0.921948 |
| Parent + worker main CPU seconds/case | 1.273707 | 1.378479 | 1.079273 |
| Setup-inclusive process CPU seconds | 173.318088 | 135.787794 | 87.108459 |
| End-to-end wall, seconds | 73.647668 | 93.516611 | 76.602646 |
| Ready allocated bytes, summed across processes | 3,587,922,944 | 1,909,981,696 | 1,793,961,472 |

Shared CPU/case rises in **every pair: 8.4832%, 5.8451%, 10.7591%**. Main wall
also worsens in every pair. Shared end-to-end wall rises **26.9784%** at the
medians despite loading/hashing only one model. Warmups must serialize graph
capture within the shared context. The serial control uses the least CPU/case
in this bounded comparison, with lower queue throughput.

Main CPU sums process intervals plus parent admission/journaling CPU, rather
than adding overlapping per-case process clocks. Per-job thread CPU is retained
only as diagnostic detail. Child main CPU ends when lane threads join; a Queue
feeder tail can extend beyond that cutoff. Total child CPU ends before queuing
the final message; spawn bootstrap CPU is excluded. These are declared intervals,
not whole-system energy or perfectly attributed IPC cost. Parent end-to-end
starts after plan creation and ends after child joins, before report writing.
See the contract.

Allocator peaks are process-local, not sampled simultaneous device peaks. At
the medians, summed main process peaks are 4,043,232,256 versus 2,281,259,008
allocated bytes. The live server's allocation is outside these figures. No
training-memory or full-context claim follows from these read-only results.

## Validation and evidence

- **454 formal presentations** independently admitted: 168 main, 143 serial
  warmup, 143 concurrent cold-reference stress. Eleven distinct existing cases.
- Every occurrence of an ID matches exactly across all seven runs. **376**
  presentations with IDs in the CPU-04 eight-case pack match that saved 3090
  receipt byte-for-byte as decoded JSON. Existing semantic failures remain.
- Every initial/final base check passes all **320 W tensors** and tied storage;
  receiver/reference fingerprints match, all parameters stay frozen with
  `grad=None`, and optimizer updates are zero. Shared tensor/container ownership
  is checked again after lanes join. Graph entry counts remain **30 per replica
  / 60 shared**, with no new capture during stress/main.
- **Four recurrence GPU tests pass in 0.946 s**: exact forward/connected-gradient
  replay with cold-capture rejection, checkpoint replay, stream/eviction behavior
  and nonfinite rejection. **Two sharing GPU tests pass in 0.324 s**, including
  a deliberately paused `functional_call` while the other view computes with
  its original parameter, and rejection of trainable/duplicated tensors.
- **30 CPU tests pass in 10.890 s** with model runtimes blocked. The first
  CPU run had one AST-fixture namespace error after adding `_SEALED`; the
  fixture was corrected and its full suite rerun. Both logs are retained.
- Original recurrence definitions are AST-identical to starting commit
  `ef78a4f` after removing only cold-capture guards. Transformer, precision,
  codec, reference, clean encoder and evaluator sources are byte-identical.
  Reuse the completed 324/108-case numeric matrices; no tolerances change and
  the previously failing original BF16 transformer is not restored.

An earlier exploratory shared run admitted another **46 presentations**, all
matching formal receipts. Its 19.095220 s main time is excluded from medians:
that earlier runner lacked the cold-reference stress/final-check barrier.
Its original plan hashes and reports remain; the archived source snapshot is
the final measured revision. Total retained case observations are 500.

Raw plans, exact commands, source snapshots, journals, individual timing samples,
tests, initial failures and offline audit: `data/cpu-07-20261107/`. That directory
is a chronological experiment name; it does not complete the separate CPU-07
host-synchronization task. Portable evidence includes every formal sample and
report/source hash. No accelerator/reference/learned-flow qualification is granted.

## Remaining target

CPU-06 remains unfinished: reduce host dispatches per case inside a larger pure
GPU region, retaining exact operations and every required finite, accounting,
cancellation and control boundary. Shared weights solve duplication and reduce
setup CPU; they do not remove the per-case Python/operator dispatch work. The
measurements do not separately attribute the slowdown to the GIL, replay lock
or GPU scheduling. Do not select another kernel merely on isolated throughput,
relax the failed RMS tolerances, grow the replica pool or launch training.
