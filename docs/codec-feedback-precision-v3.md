# Full-codec feedback precision v3

This revision fixes arithmetic for the expanded serial-feedback path. It changes
precision zones, not weights, fixtures, the FP32 reference or numeric tolerances.
It is **BF16 frozen storage with FP32 transformer dense arithmetic**; it is not
a passing claim for the original BF16 transformer arithmetic. Native vocabulary
projection still uses real compensated BF16 GEMMs. The legacy G1 sources and
checkpoints remain unchanged.

## Diagnosis

The original 512-position Noul trace reaches state normalized RMS 0.234006513
and gradient RMS errors around 0.71. Same-profile checkpoint and serving replay
are exact, so this is not a checkpoint/cache-reset discrepancy.

Controlled offline replays use identical frozen weights, prompt, packet schedule,
FP32 input and incoming cache. At the first request (position 55), the local
hidden difference is already 0.000104394. In the serial trace, returned Reason
vectors become subsequent inputs; the first state error above 0.01 appears at
the return ending at 181. By the request ending at 206, the hidden error is
0.234006513. This separates local projection arithmetic from its cumulative
amplification through differentiable feedback.

Keeping BF16 weight storage, replace only the specified projection arithmetic
with genuine FP32 products:

| Ablation | Worst state normalized RMS |
|---|---:|
| Original two-part BF16 | 0.234006513 |
| FP32 frozen-base products only | 0.186274616 |
| FP32 LoRA products only | 0.282908639 |
| FP32 all transformer base and LoRA products | **0** |
| FP32 linear-attention projections only | 0.236745590 |
| FP32 MLP projections only | 0.335497303 |
| FP32 full-attention projections only | 0.148516585 |
| FP32 layer 13 projections only | 0.107310671 |
| FP32 everywhere except BF16 delta gates | 0.015329211 |
| FP32 everywhere except BF16 QKV | 0.303880413 |
| FP32 everywhere except BF16 MLP | 0.234922348 |
| FP32 everywhere except BF16 full attention | 0.272252418 |

No tested narrower arithmetic island meets the 0.01 state limit. Three-part
BF16 input/weight expansion also failed, at 0.262063155. Extra representation
bits do not restore FP32 reduction order: compensated GEMMs still round
separate products/reductions before combining them. The evidence supports a
consistent FP32 feedback path rather than blaming a single gate, cache reset,
adapter switch or merely missing low input bits. This is a qualification of
these frozen diagnostic trajectories, not a general conditioning guarantee for
future trained weights.

## Exact precision zones

| Component | v3 |
|---|---|
| Frozen pretrained matrix storage | Original BF16 values, one registered backbone |
| Transformer base projections | CUDA IEEE FP32 `F.linear`, transient weight promotion |
| R/D adapter masters and projections | FP32, original scale and algebra |
| Residuals, nonlinearities and causal state | Existing FP32 profile |
| Native tied vocabulary projection | Existing two-part BF16 products, FP32 outputs |
| Private scalar rows | Existing isolated masters/effective rows and FP32 readout |
| Wire encoder/reader and accounting | Unchanged |
| Genuine reference | Ordinary FP32 PyTorch backbone and vocabulary products |

The frozen projection's custom autograd operation saves the original BF16
weight only, promotes it transiently again for its FP32 input derivative, and
never produces a frozen-weight gradient. Adapter gradients use ordinary
PyTorch FP32 autograd. There is no detach, alternate cache, model prefill,
recomputed request or permanent duplicate FP32 backbone. Kernel/backend flags
remain those in [A3](numeric-profile-v1.md), including IEEE FP32, no TF32 and
deterministic math SDPA.

The separately versioned [recurrence graph profile v1](codec-recurrence-graph-v1.md)
retains these arithmetic zones and reductions while replaying the pure recurrence
through CUDA graphs. Its execution evidence is separate from the eager v3 matrix.

The actual forward profile is explicit in every qualification report. Arithmetic
counters require both FP32 feedback products and BF16 vocabulary products in
each candidate case, and check frozen projection storage. GPU regressions
compare projection outputs/input gradients to PyTorch exactly, inspect saved
tensor identity/dtype, verify checkpoint/adapter gradients, and reject trainable
weights at the frozen-only operation.

The earlier two-/three-part transformer paths remain selectable using
`--feedback-precision compensated --precision-parts 2` (or 3). They are failed
diagnostic profiles, not alternative qualified settings. v3 may change speed
relative to BF16 tensor-core arithmetic; no throughput improvement is claimed.

## Qualification and scope

See the [v3 results](codec-feedback-results-2026-11-05.md) for the measured
matrix, provenance, resource accounting and memory. The original [failure
report](codec-qualification-results-2026-11-05.md) remains historical evidence.
No training, optimizer update, reviewed-data expansion or frontend work belongs
to this revision. Passing prescribed numeric traces cannot establish learned
invocation, request meaning or answer quality. The subsequent separate
[1024 extension](codec-1024-results-2026-11-05.md) passed 108/108 prescribed cases.
That report distinguishes the measured source snapshot from later finite guards.
Maximum context and full A4 trainer memory still require separate qualification.
