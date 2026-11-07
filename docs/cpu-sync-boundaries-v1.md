# CPU-02: synchronization map and first capture-region specification

Static source review, 2026-11-07. **No GPU timing, device queries or model
execution.** This is a map of the existing receiver path, not a measured ranking
of bottlenecks. CPU-01 instrumentation and optional future CUDA traces are needed
to rank costs. The qualified execution profile remains unchanged.

## Boundaries on the diagnostic path

| Site / symbol | Purpose and host involvement | Rule for capture | Measurement coverage |
|---|---|---|---|
| `g1_device.require_visibility/require_cuda`, `neural.cuda_only` | UUID/local-device admission; CUDA availability/name queries and tensor metadata checks. Repeated queries are not assumed to synchronize GPU work every time. | Validate outside a captured region. Preserve default 3090 admission and explicit read-only secondary scope. | Codec call-site counts/CPU intervals; device initialization in setup. |
| `neural.finite`, `finite_linear_output` | `bool(t.isfinite().all())` reads a device-derived decision. Reject before later head selection/nonlinear use. | Mandatory boundary. No hook removal or deferred error in this slice. | `codec_finite_guard` includes nested device-check cost; trace needed to separate kernels/waits. |
| `neural.norm` | Square/mean produces magnitude; `finite(m)` observes it before guarded division. | Two pure regions separated by the existing finite check; proposed first region below. | Guard plus containing codec/pooler phases; no new per-operation synchronization. |
| `finite.all_finite` through `Session.step` | Aggregate hidden/cache validity into one host bool, then check prior versions, ledger position and lineage. | Keep rejection before returning from the step; captured scratch cannot become public cache state. | `session_finite_guard`, `session_step`, `backbone_forward`. |
| `thinking_budget`, `Session.start_budget/startup` | Check raw FP32 score, compute ordered FP64 mapping, convert result to host integer for budget admission. | Keep exact operations and budget integer; no float reassociation or capture spanning admission. | Session/codec scopes; scalar-event details require optional CUDA trace. |
| `records.tensor`, registry scalar validation | Finite/device/shape/type checks and device scalar range predicates protect immutable typed records. Many other registry checks are ordinary host metadata, not CUDA barriers. | Registry publication, binding and sealing stay outside capture. | Not an exhaustive guard counter; tensor ops appear in the optional trace. |
| `g1_backbone.EffectiveTable.embed/project`, `Backbone.embedding` | Device-derived forbidden-private-token predicates prevent text use of private scalar rows. Dtype `.float()` methods elsewhere are GPU casts, not Python scalar readbacks. | Keep the admission checks; do not classify every Python enum/int or GPU cast as a synchronization. | Containing execution phase and optional trace. |
| `writer.Decoder` pointer/attachment/typed-value helpers, `decode_result` | `int(argmax)`, scalar `bool`, byte `.tolist()` determine real candidate IDs, lengths, types, exact values and Reason structure. | These are decisions, not redundant audit work. Preserve current masks/bindings and immediate rejection. | `result_decode`; detailed device/readback events in optional trace. |
| `behavior_qualification.evaluate_case` | Host status/value readouts are recorded, actual latent return is decoded/delivered, then labels are graded. | Grading stays after actual evaluation; no gold substitution or text invocation. | Basic per-case intervals, codec evaluation/decode/Session scopes. |
| `CleanEncoder.encode` source/reference tensor hashes | Original `contiguous -> cpu -> numpy -> tobytes` exposes actual permitted source/payload bytes for audit. | Keep byte order and audit coverage. No D-source caching or skipped hashes. | Explicit audit readback calls/requested bytes, `audit_readback` and `audit_numpy_bytes`. |
| `reference.state_fingerprint` | Ordered headers plus detached contiguous byte views are copied/hash-checked for model identity. | Do not cache from tensor versions alone; `.data` mutation can bypass them. | `weight_fingerprint` and explicit audit readback sites, including setup/final checks. |
| `g1_evidence.frozen_check` | GPU `torch.equal` returns host booleans for pinned W verification; tying checks use pointer metadata. | Keep initial/final integrity checks. No claim that they are cheap or asynchronous. | Basic setup/final-integrity intervals; not included in explicit audit-copy byte counts. |
| `recurrence._graphs/_Graphs` | Shape/dtype/device/stream keys and LRU bookkeeping are host metadata. Creation warms/captures work; copies/replay enqueue device work. Stream waits are not automatically host waits. | Existing graph-v1 equations and eight-entry policy remain. Independent outputs/saved inputs stay mandatory. | Lookup/hit/miss/eviction/key counts, capture attempts/completions, capture/copy CPU intervals. |
| `Runtime`/registry and `Session.step` cancellation/version checks | Host state machine owns reservations, charges, switches, failure/cancellation and publication. | Never capture or replay these side effects as tensor work. | Per-case unchanged runtime receipts; profiler counters are external. |
| Parent IPC, receipt validation, journal/report encoding and writes | CPU serialization, validation, process scheduling and file I/O. | Separate from model-facing computation. Preserve flushed partial receipts and canonical case order. | Parent CPU total, dispatch sends/receives, nested reporting/validation phases, final-write journal receipt. |

`Session.native_step` and native writer loops also convert learned gates/tokens
to host choices. They are **outside this forced receiver diagnostic workload**.
Similarly, component VJPs and activation checkpoint recomputation are training/
derivative-path costs, not current no-gradient inference measurements. Do not
attribute their costs to this runner without an appropriate separate workload.

The static inventory deliberately distinguishes a known device-derived read
from metadata such as `shape`, `numel`, Python tuple truth, enum conversion and
source-coordinate checks. It is not an automatic type analysis or an exhaustive
CUDA synchronization counter. No checks have moved or been removed.

## First candidate: guarded normalization tail

**Proposed profile:** `v3-norm-tail-replay-v1`, disabled and unimplemented here.
It would extend the current recurrence-only graph exception under its own
execution specification; CPU-01/02 do not register or enable it.

In `codec.neural.norm`, leave these operations and their existing failure
boundary intact:

```python
x = x.float()
m = x.square().mean(-1, keepdim=True)
finite(m, 'codec normalization magnitude')
```

The candidate region is only the subsequent existing expression:

```python
x / torch.sqrt(torch.where(m > 1e-12, m, torch.full_like(m, 1e-12)))
```

- **Inputs:** the actual invocation's CUDA FP32 `x` and its already checked FP32
  magnitude `m`; preserve broadcasting dimensions and the original epsilon.
- **Output:** a distinct normalized tensor with the original shape/dtype, with
  no alias to reusable static output buffers. Never mutate `x`, `m` or prior
  causal state. No model/registry/cache/ledger side effects occur in the region.
- **Key/workspace:** process-local selected GPU/local device, current stream,
  both input shapes/strides/dtypes and region version. Copy actual values for
  every replay. The region captures no weights or adapter lookup, so R/D/T mode
  cannot supply stale captured parameters. Invalidate workspace on key/version
  changes and explicit cleanup. Retained workspace must be measured.
- **Boundary:** `finite(m)` stays before any tail scheduling. NaN/Inf magnitude
  must fail at the same guard, with no tail replay, decoding or later admission.
  Existing Session pre/post cancellation checks and charging remain outside.
- **Oracle:** the exact eager expression above, under current v3 flags. Preserve
  operation/reduction order; do not enable compile/fusion/fast-math as part of
  the replay comparison. It must not retain permanent FP32 promoted base weights.
- **Derivatives:** a future connected implementation must preserve the paths
  through both `x` and `m`, including how `m` was formed. Never detach them or
  reuse another invocation's inputs. Derivative tests take zero optimizer steps.
- **Acceptance fixtures, deferred to GPU access:** existing observed shapes,
  zeros and magnitudes below/at/above `1e-12`, finite extremes/near-zero cases,
  strided inputs, repeated/alternating inputs, reuse/eviction/stream isolation,
  both-input derivatives, and NaN/Inf rejection at the unchanged outer guard.
  Then applicable model traces check all state families, exact decisions/budgets,
  one prefill/lineage, P+B+A+H, cancellation and frozen W/reference values under
  the original numeric limits. Reuse unchanged qualification evidence.
- **Performance decision:** compare CPU time, warm/cold latency and memory on
  the same GPU/cases. Buffer copies/replay/clones can erase the benefit of this
  small region. This is a concrete low-risk integration candidate, not a claim
  that it dominates or will improve speed. Profiling may select a better region.

Capturing a frozen dense projection is not automatically equivalent: graph
workspace can retain promoted FP32 weights, conflicting with the current storage
strategy. Capturing an entire codec block would cross required finite checks.
Neither is silently included in this candidate. GPU-side first-error flags and
predicated downstream work remain CPU-07's separate contract discussion.

## Evidence and next step

CPU-only byte/call-order, recorder, cache-metadata, aggregation and selection
fixtures are in `tests/test_host_metrics.py`. They do not execute this candidate
or prove GPU numerical agreement. The source audit verifies unchanged tensor
operations after expanding the readback helper and removing instrumentation
statements. Actual kernel/stream/memcpy tracing and attribution await CPU-03.

See [CPU-01/02 implementation evidence](cpu-01-02-results-2026-11-07.md),
[task list](cpu-involvement-tasks.md), [numeric owner](numeric-profile-v1.md),
and [current recurrence exception](codec-recurrence-graph-v1.md).
