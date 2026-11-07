# Token-ordered recurrence: CUDA graph profile v1

This is a separately versioned execution profile for the expanded codec's
FP32 gated-delta recurrence. It retains the [v3 precision zones](codec-feedback-precision-v3.md)
and [A3 tolerances](numeric-profile-v1.md). It supersedes A3's disabled-CUDA-graphs
setting **only inside this pure recurrence operation**. It enables neither
`torch.compile` nor fused/reassociated arithmetic. The original eager profile
and its measured 324/108-case matrices remain historical evidence for their
recorded source snapshots.

The later [diagnostic retention v1](recurrence-cache-retention-v1.md) permits an
explicit 32-entry cache in dedicated read-only workers, with measured evidence
in [CPU-04](cpu-04-results-2026-11-07.md). Ordinary model/training retention stays
eight; no recurrence equation, precision or tolerance is changed.

The later, separately selected [linear-attention block v1](linear-attention-block-v1.md)
extends the graph exception to recurrence/concatenation/normalization/gating only
when explicitly enabled. It has its own [numeric and throughput evidence](linear-attention-block-results-2026-11-07.md).
The default remains this recurrence-only path; that larger region is not certified
by the historical recurrence measurements alone.

`Backbone.recurrence_backend='graph-v1'` is the expanded backend default;
`'eager'` explicitly selects the retained oracle. Unknown backends reject.
Legacy G1 remains unchanged. The numeric qualification CLI defaults to eager
for its historical protocols; selecting `--recurrence-backend graph-v1` adds
`-recurrence-graph-v1` to its profile identity. Genuine FP32 reference cases
always execute ordinary eager PyTorch recurrence.

## Operation and isolation

- Keep the original token-ordered equations, PyTorch reductions, exponential,
  FP32 state, and dense/vocabulary arithmetic. Capture at most 32 positions.
- Forward replays the original CUDA operations. Return independent output and
  final-state tensors; never expose reusable static graph buffers as cache state.
- Save each invocation's actual inputs. Backward recopies those inputs and its
  incoming output/final-state adjoints, then replays forward plus the ordinary
  PyTorch VJP. Never reuse activations from another layer, packet or cache version.
- Custom autograd supplies connected first derivatives to all six inputs and
  both outputs. Cache transitions remain functional and retain incoming-state
  derivatives. There is no registry/accounting/prefill inside graph capture.
- Key graph buffers by device, current CUDA stream, shapes and dtype; serialize
  host access and retain at most eight entries with LRU eviction. Eviction may
  require recapture in backward; it must not change outputs or gradients.
- FP32 CUDA is the model path. GPU FP64 is only for small derivative checks.
  Batch size one, local cuda:0 under the prescribed RTX 3090 UUID, and head
  dimensions 1..128 are mandatory. Unsupported dtype/shape, nested capture or
  higher derivatives reject; no CPU or alternate-arithmetic fallback exists.
- Cold graph creation warms/captures CUDA forward/VJP operations. These are
  derivative/setup executions, never optimizer updates or extra logical model
  positions. Graph buffers are local execution workspace, not checkpoint weights
  or another model/cache lineage.

## Acceptance and measured scope

The GPU operator checks cover seeds 17/29/43, lengths 1/2/8/16/31/32/33/65,
connected 512/1024-position compositions, near-zero/padded shapes, both-output
and single-output derivatives, three identical checkpoint repeats, cold capture,
eviction/reuse, stream isolation and nonfinite guards. Require original FP32
pointwise/RMS/gradient limits and additionally exact graph/eager operator values
and first derivatives. Small GPU FP64 directional checks use 1/2/8/16 steps,
epsilon 1e-6 and the original atol 1e-5 / rtol 1e-3.

`scripts.qualify_recurrence` defines a focused, separate protocol: 128/512/1024
positions × Noul/Choice/Score × eager oracle / graph functional / graph
checkpoint / graph serving / genuine eager FP32 = **45 prescribed traces**.
Seed 17 fixes this full-model fixture; the operator tests separately cover the
three required seeds and repeatability. Preserve exact discrete choices, budgets,
position accounting, one prefill/cache lineage, all thirteen status cells,
all state families and connected gradient probes. Apply the stricter FP32
forward limits in addition to existing whole-codec checks. Compare the saved
trained reference's loss and VJP, without stepping any optimizer.

Measure warm timings and actual host launch calls separately from device kernel
counts. CUDA graph replay reduces host dispatch; it does not promise fewer
device kernels. Report capture overhead, profiler overhead, retained workspace
and any exclusions. A speedup for one reference episode is not a full-trainer
throughput or memory qualification. This focused protocol does not claim that
the entire original 432-case Cartesian matrix has been repeated, nor qualify
learned invocation, semantic competence, failed reference recovery or full context.

Actual results and evidence are recorded separately in the handoff and measured
report after execution. No training, data generation or services belong to this
profile change.
