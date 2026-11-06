# Recurrence dispatch overhead — 2026-11-06

Implementation and measurements for the user-authorized CPU bottleneck fix,
without data generation, agents or additional fitting. The new
[graph-v1 execution contract](codec-recurrence-graph-v1.md) retains v3 BF16 frozen
storage, FP32 transformer computation and compensated BF16 vocabulary projection.
No numeric tolerance or semantic promotion gate changes.

## Implementation

The expanded `Backbone` defaults to `recurrence_backend='graph-v1'`. The original
token-ordered PyTorch equations remain available through `'eager'`. A CUDA graph
replays each <=32-position recurrence without Python dispatch for every multiply,
reduction and backward node. This reduces **host launch overhead**, rather than
changing the operations or promising fewer device kernels.

The forward returns cloned outputs and final memory. Its custom first-order
autograd operation saves the actual inputs of each call. Backward recopies them
and the incoming adjoints, then replays the original forward plus PyTorch VJP.
It cannot accidentally use a later layer's static activations. Graphs are
device/stream/shape/dtype-specific, protected against concurrent host reuse and
bounded to eight LRU entries. The cache contains execution workspace, no model
weights, registry state, positions or alternative causal lineage. Evicted calls
can recapture when their backward is needed.

Only the pure recurrence is captured. Dense products, vocabulary projection,
adapters, scalar rows, budget mapping, packet rendering, session transitions and
accounting retain their original implementations. There is no `torch.compile`,
changed reduction order, new FMA contraction, TF32 enablement or CPU fallback.
Nested graph capture, unsupported inputs and higher derivatives have no alternate
execution path. Graph creation does GPU warm-up/capture work; warm benchmarks
exclude that setup and report cold behavior separately.

The numeric CLI retains eager as the historical matrix default and records
`-recurrence-graph-v1` when explicitly selected. Genuine FP32 reference cases use
the eager recurrence, not the candidate graph implementation. Reference recovery
evidence now pins the graph profile and recurrence source hash. Frozen flags or
old source receipts cannot grant current semantic qualification.

## Rejected fused-kernel attempt

An initial Triton implementation reduced the number of kernels. Its six local
GPU tests passed in 15.074 s, including input/cache gradients. It then failed
the 512-position Score serial-feedback comparison: state normalized RMS reached
8.471535152303649e-5 and strict FP32 gradient errors reached
4.3080439594012614e-4 (wire). Both exceed their unchanged 1e-5 / 1e-4 limits.
Tiny local reduction-order differences were amplified by feedback.

That candidate was rejected and removed from production code. Its logs, source
snapshot and failed report remain under `data/recurrence-fused-20261106/`.
The accepted approach replays the original reductions. Local success alone was
not used to approve a full-model path, and no tolerance was widened.

## Validation protocol

All model and derivative work uses RTX 3090 UUID
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1`; ordinary metadata/static checks use
the host. No optimizer step is part of this qualification.

**Eight operator tests passed in 3.745 s.** They cover production head shapes,
seeds 17/29/43, temporal boundaries 1/2/8/16/31/32/33/65, connected 512/1024
compositions, near-zero/padded dimensions, both and individual outputs,
checkpoint replay and three repeats, cold capture, LRU eviction, stream isolation
and visible nonfinite-state rejection. Compared output/state values and all six
input gradients are exact against eager; original FP32 limits also pass.
Small GPU FP64 directional derivatives use 1/2/8/16 steps and the original
epsilon 1e-6 / atol 1e-5 / rtol 1e-3.
All 32 operator comparisons have zero forward and gradient error; the largest
FP64 finite-difference absolute error is 4.248507101678456e-11.

**Eight affected integration checks passed in 42.783 s**: serial/parallel
teacher forcing and suffix causality, three frozen-projection/vocabulary/gradient
checks, scoped T checkpoint replay, full-bundle export/row tying, and frozen
auxiliary reconstruction's connected input gradient with no live-owner gradient.
The latter reproduced input gradient norm 0.25784578919410706. These selected
tests do not step an optimizer.

**65 model-free checks passed in 2.801 s**:

```sh
PYTHONPATH=tests .venv/bin/python -m unittest test_reference_qualification test_episode_producer test_training_inputs test_training_driver test_training_release test_training_schedule test_data_production test_qualification_protocol -q
```

The separate focused full-model protocol is 128/512/1024 × Noul/Choice/Score ×
eager / graph functional / graph checkpoint / graph serving / genuine FP32:
45 prescribed traces. It checks every captured hidden/cache state family,
logits, scalars, discrete decisions, integer budgets, one prefill/cache lineage,
all thirteen status cells and connected native/latent gradient paths. It applies
strict FP32 forward limits in addition to the existing whole-codec checks.
This is affected-path qualification, not a claim that the original 432-case
Cartesian matrix was repeated or that the entire learned flow is certified.

**45/45 traces passed in 944.7234295289963 s**, including 36 comparisons against
nine eager baselines. Hidden/cache errors were zero at every captured boundary;
returned scalar and startup-score differences were zero, and discrete choices,
predicted/granted budgets, accounting and all thirteen operation/status cells
matched. Graph serving carries no derivative claim.

| Comparison | Maximum state RMS error | Maximum gradient RMS error | Minimum gradient cosine |
|---|---:|---:|---:|
| Graph functional / eager | 0 | 0 | 1 |
| Graph checkpoint / eager | 0 | 0 | 1 |
| Graph serving / eager | 0 | N/A | N/A |
| Genuine eager FP32 / candidate precision | 0 | 8.316676446703025e-6 | 0.9999999999901952 |

Graph/eager vocabulary logits were exact. Genuine FP32 vocabulary differences
reached normalized RMS 4.169371270284753e-6, inside the unchanged 0.01 limit;
the candidate continues to use compensated BF16 products. Live parameters stayed
byte-identical. All 320 frozen base tensors passed equality and physical tying
remained intact. The report and runner exit both indicate pass.

## Measured launch overhead and throughput

The saved 1,000-update reference, fingerprint
`fb38a31359a4ff367780bb8729695894e663810efd124c016bc2d5ba3fdbf7de`,
was loaded without fitting. Both backends used the same frozen-W annotation
states and existing `train-reference-0000` input. Three timed forward/VJP passes
followed a warm-up; every graph loss and all 7,899,904 active gradient elements
were exact against eager. Reference weights remained unchanged.

| Measurement | Eager | graph-v1 |
|---|---:|---:|
| Median warm forward/VJP | 1.4445074120012578 s | 0.3462741900002584 s |
| Timed samples, seconds | 1.4445074120, 1.6177119560, 1.4400201710 | 0.9126432990, 0.3462741900, 0.3461874470 |
| Profiled `cudaLaunchKernel` host calls | 52,243 | 9,007 |
| Profiled device kernels | 52,497 | 58,905 |

The measured warm median speedup is **4.1715711240279845×**. Host kernel-launch
calls fell by about 82.8%; CUDA graph replay calls are separate API submissions.
There were 72 `cudaGraphLaunch` calls, versus zero eager. Counting both launch
APIs gives 52,243 → 9,079 host submissions (about 82.6% fewer).
Device kernel count increased with replay/rematerialization. This is a reduction
in CPU dispatch overhead, not a fusion/kernel-count claim. The first timed graph
sample was slower than the following two and is retained above. Profiling was a
separate pass, outside the timing samples. This single-episode measurement
excludes accumulation guards and optimizer execution.

An additional timing uses the **first two actual scheduled eight-exposure
reference windows**, with 16 existing IDs and conservative lengths 21..49. It
calls the real `Window.add_episode` eligibility, per-component VJP, finite checks
and accumulation path. After each eight exposures, it verifies eligibility and
discards the buffers with `abort()`. It never calls `step()`; AdamW moments,
scheduled optimizer windows and optimizer updates stay at zero.

Three post-warm-up samples for the two-window sequence were
18.7174303570 / 18.7281852870 / 18.7382580950 s eager, and
5.8465706710 / 5.8218986650 / 5.8340748200 s graph. Warm medians are
**18.728185287007364 → 5.83407481998438 s (3.210138002148129×)**, including
actual guard/accumulation costs and dynamic-shape recapture behavior. Cold first
samples were 18.84284500402282 / 5.99536470399471 s respectively. All mean losses
were exactly 0.042331128042860655; final reference fingerprints matched.
Maximum process allocator peaks were 3,554,264,064 / 3,602,701,312 bytes allocated
and 4,324,327,424 / 4,328,521,728 bytes reserved (eager/graph). These are local
allocator measurements, not combined-device peaks or full A4 trainer memory.

## Scope and retained evidence

The fixed reference's earlier failed held-out recovery remains failed. This
performance change does not qualify it, add epochs or authorize capability
training. Training/development/locked data, fitted tensors and live services
remain unchanged. The Operating Table's snapshot is read-only.

Raw graph evidence is under `data/recurrence-graph-20261106/`; raw rejected-fusion
evidence is retained separately as described above. Binary tensors, profiler
traces and local benchmark helpers remain outside git. Portable source/log hashes
and measured results are recorded in the
[portable receipt](evidence/recurrence-graph-v1.json).

The checked entrypoint is:

```sh
CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 \
.venv/bin/python -m scripts.qualify_recurrence \
  --snapshot /home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
  --output data/recurrence-graph-20261106/model --benchmark
```

It requires a new output directory and existing pinned local artifacts; no
download or CPU fallback is provided. The raw source snapshot includes this
runner and the actual graph implementation. Use its retained evidence for
inspection; repeat model work only for relevant changes. The mixed-window
helper and exact integration invocation are separately retained/hash-bound.

The next unfinished semantic slice is still investigating failed reference
generalization and documenting a development revision before another authorized
fit. Dynamic-shape/capture costs and full optimizer-step/trainer-memory evidence
remain distinct from the measured two-window accumulation speedup.
