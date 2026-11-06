# A3: numeric profile v1

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

Owner: exact A3 near-zero rules, budget evaluation, backend flags and fixtures.
The [numeric verification plan](numeric-verification.md) owns the broader
invariants, pointwise tolerances, gradient paths and export checks. Follow the
[model profile](model-configuration.md), [budget accounting](position-limits-and-budget-accounting.md)
and [implementation specification](architecture-and-training-plan.md).

G1-only implementation revision: [compensated precision v2](g1-compensated-precision-v2.md)
records the BF16 truncation diagnosis and compensated-forward/FP32-state/backward
fix. It changes arithmetic zones, not the tolerances below. The original
single-pass BF16 profile remains selectable for historical reproduction.

Full-codec arithmetic revision: [feedback precision v3](codec-feedback-precision-v3.md)
promotes transformer dense computation to FP32 while retaining BF16 frozen
storage and compensated BF16 vocabulary products. This supersedes the G1
arithmetic zones for that scoped codec profile only. All limits, floors,
budget formulas and genuine reference rules below remain unchanged. See its
[measured results](codec-feedback-results-2026-11-05.md) separately from the
original failed expanded BF16 profile.

## 1. Near-zero comparisons

Compute verification norms/metrics in GPU FP64, separately for every layer,
state family and meaningful parameter group. This does not change model-forward
normalization or scalar accumulation precision.

    E = RMS(candidate-reference) / max(RMS(reference), 1e-6)

Keep the numeric owner's pointwise tests AND require:

| Comparison | Maximum E | Absolute RMS error for reference RMS <=1e-6 |
|---|---:|---:|
| FP32 reference/functional | 1e-5 | 1e-11 |
| BF16 train/serve states | 1e-2 | 1e-8 |
| Allowed vocabulary logits | 1e-2 | 1e-8 |

The denominator floor provides one continuous rule through zero. Scalar checks
remain separate: identical-hidden-input row/export atol=1e-6/rtol=1e-5;
whole-path Noul/task Score absolute difference<=0.002; startup budget score
absolute difference<=1e-5 PLUS exact predicted/granted integers.

For unscaled/unclipped gradients:

    E_g = RMS(candidate-reference) / max(RMS(reference), 1e-10)

| Comparison | Maximum E_g | Minimum cosine |
|---|---:|---:|
| FP32 reference/functional | 1e-4 | 0.99999 |
| FP32 checkpoint off/on | 1e-4 | 0.99999 |
| BF16 training/FP32 reference | 0.05 | 0.995 |

Evaluate cosine only when BOTH gradient RMS values exceed 1e-10. Below that,
use the implied absolute RMS bound (1e-14 FP32; 5e-12 BF16). A near-zero match
does not establish connectivity: each intended path also needs a nonsaturated
fixture producing a measurable nonzero derivative. Frozen parameters retain
grad=None; prohibited paths remain disconnected. Distinguish legitimate initial
LoRA zero gradients from missing connections using post-bootstrap fixtures.

Unexpected NaN/Inf fails immediately, including intermediates/gradients/moments.
Only the exact intended masked logit positions may contain negative infinity.

## 2. Canonical budget arithmetic

The raw readout is FP32. Reject nonfinite values BEFORE clamping and promote
the exact FP32 value to FP64. Execute these separate operations in order:

    s = clamp(float64(raw_float32_score), 0, 1)
    u = s * s
    v = 3.0 * u
    w = 2.0 * u
    z = w * s
    curve = v - z
    scaled = curve * 131072.0
    B_pred = int64(floor(scaled + 0.5))

Every floating operation uses binary64 round-to-nearest-even. Do not use FMA,
reassociation, another polynomial, decimal-string round trips or generic round().
The final nonnegative rounding is half-up. Values 0/.25/.5/.75/1 map exactly to
0/20480/65536/110592/131072 positions.

All position arithmetic is checked nonnegative int64:

    available = C - P - A - H
    B_granted = min(B_pred, available)

Reject P+A+H>C; do not conceal the error by clamping available to zero. Preserve
the established manual-override bounds and separate override telemetry.
Preparation is charged as it occurs. Before dispatch reserve actual request
length + maximum legal return length under [control-flow v1](control-flow-v1.md),
which removes the former 32-position consumption hold. Debit actual
use and release unused reservation, preserving remaining A/H. The three-position
startup return is charged to H; starting the answer does not reset B.

## 3. Shared GPU kernel/precision profile

Select RTX 3090 UUID GPU-225509c3-d702-6c27-de03-76b5e99b33b1 explicitly. After
CUDA_VISIBLE_DEVICES selects it, use its local cuda:0 identity. Record the actual
hardware, driver, Torch/CUDA/cuDNN and relevant library/kernel versions and hashes.

| Setting | v1 |
|---|---|
| Forward precision | A2 BF16 base/adapters; FP32 protocol projections, normalization and losses |
| Metrics/small derivative references | GPU FP64 |
| Attention | CUDA SDPA MATH only, explicitly selected |
| Gated-delta recurrence | Same functional token-ordered CUDA recurrence in train/serve; FP32 recurrent accumulation |
| Causal convolution | Explicit differentiable CUDA implementation, identical causal ordering |
| Fused recurrent/automatic hub kernels | Disabled |
| torch.compile/CUDA graphs | Disabled |
| Dropout | Zero everywhere |
| Numeric generation | Greedy, batch size one |

Freeze these flags:

- torch.backends.fp32_precision = "ieee"
- torch.backends.cuda.matmul.fp32_precision = "ieee"
- torch.backends.cudnn.conv.fp32_precision = "ieee"
- torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
- torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (False, False)
- torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
- torch.use_deterministic_algorithms(True, warn_only=False)
- torch.backends.cudnn.benchmark = False
- torch.backends.cudnn.deterministic = True
- CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization.

Use new fp32_precision controls consistently, without mixing deprecated TF32
controls. The reduction tuples disable reduced-precision accumulation and split-K.
Treat unsupported deterministic operations as a blocker requiring a declared
GPU implementation/profile revision, not silent backend substitution.

Training retains functional transitions without mutating autograd-saved tensors.
Serving may replace cache-container entries using the same equations. Faster
kernels require a separately versioned profile passing all applicable checks.

The subsequent [recurrence graph profile v1](codec-recurrence-graph-v1.md)
replays the same eager operations inside bounded pure recurrence CUDA graphs.
It is an explicit, separately measured exception to the disabled-graphs row
above; it does not change precision zones, tolerances or historical results.

## 4. Fixture manifest

Use seeds 17/29/43 and immutable manifests recording inputs, weight/config
hashes, expected structural outcomes and execution settings.

| Group | Required cases |
|---|---|
| Near-zero | Zero reference; below/at/above RMS floors; sign changes; cancellation; injected NaN/Inf |
| Budget mapper | Endpoints, finite out-of-range values, signed zero, nonfinite rejection; neighboring FP32 values around all 131072 rounding transitions |
| Ledger | Context exact-fit/one over; context-capped grant; B=0; reservation exact-fit/one short; release unused capacity; answer-phase calls |
| Complete traces | Direct answer; Noul/Choice/Score; successive R-D-R calls; answer-phase call; every supported semantic status; cancellation/resource failure |
| Packet bounds | Exact typed round trips; min/max bodies; empty/absent; 1/2/16 candidates; request 256 accepted/257 rejected; maximum legal returns; oversized values rejected |
| Gradients | Isolated intended paths; blocked budget feedback; inactive rows/frozen weights unchanged; checkpoint replay without registry/accounting side effects |
| Export | Effective scalar-row bytes, tying, private-row masks, raw scores, integer budgets and continuation after save/load |

The budget oracle independently implements the specified binary64 rounding;
it must not call the production mapper to generate its expected outputs. These
scalar fixtures are not hundreds of thousands of model executions.

Complete trace fixtures use 128/256/512 TOTAL positions when the full packet
fits. Larger packet-bound fixtures run separately at their actual lengths;
never truncate to fit a nominal length. A4 requires the same applicable checks
at 1024 total positions before joint training; passing initial 512-position
fixtures does not waive that length qualification. Small GPU FP64 smooth derivative checks
use epsilon=1e-6, atol=1e-5, rtol=1e-3 and local compositions of 1/2/8/16 steps.
Hold discrete choices fixed; do not finite-difference argmax, byte selection,
BF16 rounding or clamp boundaries.

## 5. Promotion and scope

Prescribed traces compare states/gradients with identical supplied operation
schedules. Free greedy traces require identical learned decisions, refs, statuses,
integer budgets and generated tokens between train-compatible and serving paths.
These are separate reports; forced traces cannot prove learned invocation.

Changed discrete traces fail parity even if tensor tolerances pass. Report
near-tie margins; no routing exception. For each seed, three identical reruns
check same-profile repeatability. Cross-precision FP32/BF16 checks use the stated
tolerances, not a bitwise-equivalence claim. All exact invariants, forward/gradient
tests, discrete parity and export checks must pass.

These are starting acceptance settings. Calibration/revisions require versioned
development evidence before locked testing. Numeric agreement establishes neither
semantic understanding, autonomous invocation nor maximum-context reliability.

References: [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html)
and [SDPA backend selection](https://docs.pytorch.org/docs/2.15/generated/torch.nn.attention.sdpa_kernel.html).
These support the controls/methodology, not the empirical adequacy of these limits.
