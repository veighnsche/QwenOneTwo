# G1 compensated precision profile v2

Scope: user-authorized BF16/FP32 diagnosis and G1 rerun on RTX3090 only.
This revises numerical implementation, not codec, data, capability training or
frontend. Tolerances, fixtures, dimensions, diagnostic weights, discrete controls
and position accounting stay unchanged. Preserve the [original failure](g1-core-results-2026-11-05.md).

## Root cause

The old backend repeatedly rounded normalized activations, dense outputs, LoRA
additions and residual sums to BF16. Casts also rounded intermediate gradients.
Error appears in the FIRST prefilling layer, before any adapter switch, and
propagates through24 layers, recurrent state and writer/return feedback. It was
not a second-prefill, wrong-GPU, cache-reset or disconnected-gradient problem.

Controlled whole-flow ablations used the same frozen diagnostic checkpoint:

| Comparison to FP32 | Noul worst state RMS error | Score worst state RMS error |
|---|---:|---:|
| Original BF16 | .0366066 | .0867262 |
| Original BF16, matching reference LoRA/wire rounding | .0436267 | .0897197 |
| FP32 residual/state islands, ordinary BF16 products | .0345991 | .0346785 |
| Compensated BF16 products and FP32 islands | .0000417819 | .0000298173 |

Matching reference rounding did not fix the issue. FP32 islands reduced Score
readout error from .01068449 to .000105798, but state checks still failed.
Compensation reduced readout error to .000000149012. GPU reproducer:
x=[1,1+2^-10],w=[1,-1] gives zero with ordinary BF16 input rounding, whereas
FP32 and compensation give -2^-10. BF16 residual1+2^-10 likewise rounds to1.

## Arithmetic

Split x into BF16 x_hi and x_lo=BF16(x-FP32(x_hi)); split FP32 adapter W similarly:

    y = x_hi W_hi^T + x_lo W_hi^T
        + x_hi W_lo^T + x_lo W_lo^T

All products have CUDA BF16 operands and FP32 output/accumulation. Frozen BF16
weights have no low part: two products suffice; FP32 adapter masters need four.
Keep the result and residual addition in FP32. This retains low-order terms
without silently running the candidate's dense forward GEMMs in FP32.

Backward computes the underlying linear-map derivative in CUDA FP32:
grad_x=grad_y W;grad_W=grad_y^T x. BF16 rounding is not a smooth derivative
target and remains excluded from finite differences. This is an explicit
master-gradient convention, not a derivative through hard routing decisions.

See [precision math](../qwen_one_two/g1_precision_math.py) and
[backbone](../qwen_one_two/g1_backbone.py). Four
[GPU regressions](../tests/test_g1_precision_gpu.py) passed: cancellation,
residual rounding, actual BF16 operands and FP32 analytical gradients.

## Precision zones

| Zone | Candidate v2 |
|---|---|
| Frozen pretrained storage | Original BF16 bytes/dtypes, preserved |
| R/D masters and optimizer buffers | FP32 |
| Dense base/LoRA/native-output forward | Compensated BF16 products; FP32 output/accumulation |
| Wire payloads/residual stream/LoRA additions | FP32 |
| Norms/nonlinear gates/depthwise convolution | FP32 |
| Attention arithmetic and K/V state | CUDA SDPA MATH/FP32 |
| Convolution and recurrent state | FP32, same functional causal order |
| Dense input/master gradients | CUDA FP32 |
| Protocol modules | FP32, unchanged |
| Scalar rows | Same BF16-effective rows/FP32 accumulation, unchanged |
| Metrics/local smooth references | CUDA FP64, unchanged |

FP32 reference keeps genuine FP32 dense arithmetic and unrounded adapter masters.
Rounding-matched modes are diagnostic ablations only, not gate exceptions.
TF32/reduction/split-K settings, cuBLASLt, zero dropout, no fused recurrence/
compile/hub kernels and deterministic settings remain A3's. No CPU fallback.

This is a versioned mixed profile, not a claim that ordinary single-pass BF16
became FP32-equivalent. Extra products and FP32 state/gradients cost memory/time.
High/low parts add FLOPs, not model positions, slots, prefills or cache lineages.
No production service or full-context qualification is implied.

## Evidence and reproduction

Use `python -m qwen_one_two.g1_proof --stage certify --precision-policy compensated`
with pinned snapshot, read-only original `--checkpoint` and a new output directory.
`legacy` reproduces the old backend; `residual` isolates the island-only ablation.
There is no new bootstrap or fitting to change the comparison checkpoint.

Original checkpoint: `data/g1-proof/run-20261105/diagnostic.safetensors`.
Ablations: `data/g1-proof/precision-20261105/`.
Full rerun: `data/g1-proof/precision-fixed-20261105/`.

The complete 72-case matrix passed UNCHANGED state<=.01, scalar<=.002,
gradient<=.05 and cosine>=.995 limits. Worst slot-gradient error was .000131086;
cosines were >.99999999. Cache/gradient, negative-control, accounting and optimizer
checks passed. At that stage, released Noul still selected unsupported Choice,
so the [precision-stage rerun](g1-precision-results-2026-11-05.md) FAILED overall.
The later [gate-head repair/rerun](g1-gate-results-2026-11-05.md) PASSED bounded
G1-core while preserving this numerical profile and its tolerances.
Numeric success does not establish learned invocation or lift the user's
explicit no-expansion instruction.
