# Larger recurrent attention replay — 2026-11-07

**The opt-in larger pure GPU block reduces complete-case CPU cost by 3.5924%
and improves resident throughput by 3.5649%** on the existing eight-case 3090
subset. All three alternating pairs improve CPU/case. This is a modest measured
gain; most host work remains. Default execution stays unchanged.

[Contract](linear-attention-block-v1.md),
[portable evidence](evidence/linear-attention-block-v1.json).
Only RTX 3090 UUID `GPU-225509c3-d702-6c27-de03-76b5e99b33b1` executes model work.
No agents, training, data generation/reviews, deployment or service changes.
W and the reference remain frozen. QwenOneTwo PID 2094 and The Operating Table
remain untouched. No elapsed-time or cost cutoff was introduced.

## Implementation and acceptance

`codec/attention_block.py` captures the original 32-position token-ordered
recurrence chunks, concatenation, RMS normalization and SiLU gating as one pure
sequence operation, bounded to 1..1024 supplied positions. Graphs consume actual
q/k/v, decay, beta, incoming memory, gate activation and norm-vector values.
They contain no dense projections, promoted frozen matrices or module lookups.
Storage stays BF16, transformer computation FP32, vocabulary projection
compensated BF16. Original failed BF16 transformer arithmetic stays rejected.

The independent dense gate projection moves before recurrence to provide a
graph input. Its CUDA/frozen-weight admission happens earlier, outside capture.
No finite check or state publication is crossed. The existing codec/Session
guards, control, cancellation/version checks, S-curve and accounting remain.
Actual-input saves and replayed VJPs preserve connected first derivatives;
independent output clones protect prior causal state from scratch reuse.

Keys bind device/stream, all shapes/dtypes and epsilon, with contiguous inputs
and LRU retention. Default retention is eight; this explicit read-only benchmark
selects 32. Unknown backends/shapes/dtypes, nested capture and higher derivatives
have no alternate execution path. `Backbone.attention_block_backend='eager'`
remains the default; `'graph-v1'` explicitly selects this diagnostic profile.

Acceptance requires strict original full-state/gradient checks, exact receipt
and frozen-weight agreement, and measured full-case CPU/launch reductions.
These pass within the recorded scope; no learned/semantic or training grant is
produced. Full-context and unseen-shape deployment remain unqualified.

## Numeric and frozen-reference evidence

- **Six GPU operator tests pass in 8.218 s**: exact outputs/all first derivatives,
  seeds 17/29/43, temporal boundaries, production dimensions, 512/1024 connected
  positions, zero/near-zero/strided inputs, single-output derivatives, FP64
  directional checks, checkpoint/reuse/eviction/stream behavior, independent
  outputs and nonfinite visibility. The strengthened post-eviction saved-input
  gradient test passes separately **1/1 in 0.682 s** after its assertion was added.
  The intentional cross-stream test emits a PyTorch AccumulateGrad stream warning;
  its exact-gradient assertions pass. It does not qualify distributed training.
- **36 full-model numeric traces pass in 410.912892 s**: 128/512/1024 ×
  Noul/Choice/Score × oracle/functional/checkpoint/serving, seed 17. All **27**
  candidate comparisons have exact forward state values. All **18** gradient
  trace comparisons have zero RMS difference/cosine 1 for R, D, wire and emitter
  groups. Strict FP32 pointwise/RMS and original gradient limits are unchanged.
  Logits, scalars, budgets, decisions and accounting also match. Existing 324/108
  matrices are reused, rather than repeated.
- Initial/final W checks pass all **320 tensors** and tied storage; diagnostic
  weights remain identical with `.grad=None`. No optimizer is constructed or
  stepped. The prescribed fixtures remain numeric diagnostics.
- On existing `train-reference-0000`, the actual checkpoint-0188 frozen reference
  has identical teacher-forced loss **0.10903861373662949** and exact latent-input
  VJP (8,192 values, gradient RMS difference zero/cosine 1). Eight latent prefix
  positions and forty targets; differentiate only the latent input. W/reference/
  receiver hashes stay pinned and every parameter stays frozen with `.grad=None`.
  This is parity, not a renewed exact-reconstruction or semantic gate.
- The data benchmark explicitly clears/recomputes reference payloads under the
  candidate in a full eight-case parity pack before timing. Exact audit hashes
  and receipts prevent oracle memoization from hiding a changed reference path.
- **34 CPU tests pass in 13.223 s** with model runtimes blocked. Four plan tests
  rerun after the reference-parity addition (2.026 s) and after the timing-order
  correction (**1.959 s**). They bind all 147 presentations and prohibit any
  profiler from starting before the last timed pack.

Largest qualifier allocation is 10,909,249,536 bytes for the oracle and
10,915,595,264 / 6,352,775,680 / 3,421,905,408 bytes for candidate functional /
checkpoint / serving. Candidate comparisons retain the oracle's snapshots;
these are qualification-process peaks, not standalone training/inference memory
comparisons. No full-trainer memory claim follows.

## Resident measurements

One worker/model/CUDA context reuses the existing development subset, receiver
checkpoint and frozen reference. Three paired orders are original/candidate,
candidate/original, original/candidate. Each trial resets workspace, warms all
eight cases, then times eight cases with counters and profilers off. Only after
all six timed packs finish do separately warmed observations collect counters
and traces. Whole X pairs remain intact; source states are fresh per case.

| Median metric | Existing recurrence graphs | Larger block |
|---|---:|---:|
| Worker-case + parent-phase CPU seconds/case | 1.109460 | 1.069604 |
| Summed eight-case worker wall, seconds | 8.921580 | 8.600628 |
| Parent cases/second | 0.889370 | 0.921075 |
| Timed peak allocated bytes | 1,991,849,472 | 2,063,090,688 |
| Timed peak reserved bytes | 2,199,912,448 | 2,334,130,176 |

Paired CPU/case reductions: **3.7546%, 3.4705%, 3.4305%**. Worker wall falls
**3.5975%** at the medians. Extra allocated memory is **71,241,216 bytes
(67.94 MiB)**. Counter-instrumented cold workspace warmup pack CPU medians are
9.177257 versus 9.156095 s: essentially equal. These include inference and counter
overhead, not isolated capture cost. Every setup sample remains in raw receipts.

Separate warm observations contain **1,998 old recurrence requests versus
1,548 larger-block requests** across eight cases, all hits, zero recaptures or
evictions. Cold trials create 22 old versus 29 larger shape graphs, fitting the
selected 32-entry capacity. This does not establish an arbitrary-workload cache
policy, long-context memory bound or cold single-request speedup.

The same existing first Noul case has these independently profiled counts:

| Trace observation | Existing | Larger block |
|---|---:|---:|
| Direct CUDA kernel launch API calls | 43,324 | 41,722 |
| Graph launches | 198 | 180 |
| Device kernel events | 72,682 | 72,502 |
| Stream synchronizations | 792 | 792 |

Direct kernel launches fall **3.6977%**; required synchronization counts stay
unchanged. Replay reduces host dispatch rather than fusing arithmetic. API and
device durations overlap; do not sum them into GPU compute/idle time. Traces and
their overhead are excluded from timing. The traced Noul case takes 1.701746 /
1.671348 s versus unprofiled medians 0.936693 / 0.900407 s (existing/candidate):
combined profiler/counter overhead is 81.68% / 85.62%. Worker case CPU excludes some IPC/
Queue-feeder work outside its case intervals; parent phase CPU includes result
admission/journaling. Allocator peaks exclude the live server and are not sampled
physical-device peaks. Do not multiply these gains by previous reports.

## Preserved first run and provenance

The first 131-presentation run passed every receipt but put profiled observations
between timed trials. Its oracle CPU/case shifted **1.098 → 1.427/1.443 s** after
profiling, invalidating that performance comparison. Its entire timing set is
excluded, not selectively trimmed. Moving all profiling after timing in a fresh
process yields stable oracle CPU/case and the three positive pairs above. This
supports correcting the experiment order; exact profiler-internal attribution
was not separately measured.

The corrected run admits **147 presentations** (48 timed, 96 other main-phase
observations, three startup anchors). Both runs together admit **278**, all
matching prior CPU-04 3090 receipts exactly, across eleven distinct existing
cases. Existing semantic failures remain; no new independent training cases exist.
Both process exits are zero, and initial/final base/receiver/reference/frozen
checks pass with zero updates.

Numeric sources are identical between qualification and final measurements.
The only later changes are benchmark phase/reference-cache preparation in
`two_gpu_worker.py` and `benchmark_attention_block.py`; both measured source
versions are preserved. Removing the disabled new branch/class setting leaves
the existing backbone AST identical to starting commit `777d411`. No unchanged
numeric matrix is rerun for benchmark scheduling changes.

Raw source snapshots, commands, qualification, reference VJP, both benchmarks,
all timing samples, flushed journals, tests, traces and offline audit:
`data/cpu-08-20261107/`. This chronological name does not qualify the new profile
on two GPUs. Next: use the saved traces to select the next pure region among
remaining projection/attention dispatches, respecting BF16 storage and immediate
guards. Full 310-case, >1024-position, secondary-card, training-memory and learned
invocation/answer behavior remain unfinished.
