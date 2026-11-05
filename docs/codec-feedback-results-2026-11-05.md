# Full-codec feedback v3 qualification — 2026-11-05

**Passed: 324/324 prescribed numeric qualification traces on RTX 3090.**
The [v3 precision revision](codec-feedback-precision-v3.md) uses BF16 frozen
weight storage, FP32 transformer base/LoRA computation and compensated BF16
native vocabulary products. It is an explicit arithmetic change; the original
BF16 transformer profile still fails. No tolerance, weight, fixture or genuine
FP32 reference was changed to obtain this pass.

## Root cause and fix

Dense projection rounding differences compound through successive latent
Reason feedback. Common-input/cache replays already differ locally; full serial
traces amplify those differences. The original 512-position state RMS error was
0.234006513, with gradient errors around 0.71. Neither extra BF16 split parts nor
the tested narrower projection islands resolved it. FP32 arithmetic for both
base and LoRA transformer products reproduces the reference states exactly.
See the [controlled ablations](codec-feedback-precision-v3.md) and preserved
[original failure report](codec-qualification-results-2026-11-05.md).

The frozen projection saves its original BF16 weight for backward and transiently
promotes it for FP32 products. It does not save an FP32 weight copy per call,
create another backbone, detach the cache or repeat a logical prefill. Real
BF16 vocabulary products are checked with per-case arithmetic counters.

## Matrix and measured agreement

The matrix is 128/256/512 positions × Noul/Choice/Score × functional/checkpoint/
serving/FP32 × seeds 17/29/43 × three repetitions: **324 distinct traces**.
There are 315 comparisons to nine baseline traces, including 81 genuine FP32
reference comparisons. Every case and every comparison passes. Seeds reset RNG
on a fixed frozen checkpoint; they are reproducibility checks, not independently
trained models or capability evaluations. Inputs and weights remain fixed across
seed/repeat axes; those axes check determinism. All thirteen status cells occur in
the 512-position coverage, including Choice's no-feasible-selection result.

| Whole-path measurement | Worst observed | Unchanged limit |
|---|---:|---:|
| Hidden/cache normalized RMS | 0 | 0.01 |
| Native vocabulary logit normalized RMS | 5.825623065e-06 | 0.01 |
| Returned scalar absolute difference | 0 | 0.002 |
| Startup raw score absolute difference | 0 | 0.00001 |

State/logit pointwise tests also pass; no denominator floor was changed.
Predicted/granted budget integers, discrete selections, statuses, boundaries and
accounting match exactly. Same-profile checkpoint state/gradient comparisons
are exact. Cross-precision gradients satisfy RMS <=0.05 and cosine >=0.995:

| Parameter group | Maximum normalized RMS | Minimum cosine |
|---|---:|---:|
| R | 8.149575206e-06 | 0.9999999999931394 |
| D | 7.784975292e-06 | 0.9999999999934059 |
| wire | 7.517741501e-06 | 0.9999999999901952 |
| emitter | 8.316676447e-06 | 0.9999999999915314 |

Native-loss VJPs through returned Noul/Score values, request packets, question
slots, writer R and evaluating D are finite and nonzero. Choice has connected
selected-content gradients and a separate connected pointer objective. No
argmax derivative or semantic correctness is claimed.

## Cache and accounting

Every trace uses one prefill and one logical causal-cache lineage. Incoming
state versions remain unchanged. Requests/results stay bound to immutable
snapshots, each dispatched result is consumed exactly once, and every completed
trace leaves zero position holds. P+B+A+H equals its exact 128/256/512 length.
Serial calls occur before and during answering, without resetting B.

| 512-position operation | P | B | A | H | Total | Dispatch/consume | Released holds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Noul | 27 | 188 | 291 | 6 | 512 | 4/4 | 132 |
| Choice | 27 | 345 | 134 | 6 | 512 | 5/5 | 155 |
| Score | 27 | 291 | 188 | 6 | 512 | 4/4 | 132 |

## Memory, arithmetic and regression checks

Maximum process CUDA allocation: 9.2546 GiB;
maximum process reserved memory: 9.3457 GiB.
These include 446489460 bytes of resident envelopes: two FP32 moment copies and
three full-parameter component-buffer copies. No optimizer is created or
stepped. This is a conservative memory envelope, not actual A4 eligibility or
joint-training capacity. Two bounded workers share the explicitly selected
RTX 3090 UUID `GPU-225509c3-d702-6c27-de03-76b5e99b33b1`; per-process peaks do
not claim a measured combined-device peak. The existing service was untouched.

Candidate arithmetic counter ranges (including backward rematerialization and
VJP probes; these are not logical position counts or complete FLOP totals).
Counters mark completed instrumentation sites; checkpoint early-stop can skip
an increment after already executed products. They verify exercised paths,
while exact state/gradient comparisons verify the numerical result:

| Counter | Minimum/case | Maximum/case |
|---|---:|---:|
| fp32_frozen_forward | 2046 | 16368 |
| fp32_frozen_backward | 0 | 10461 |
| fp32_adapter_forward | 4092 | 30000 |
| bf16_forward_products | 2 | 2 |
| fp32_backward_products | 0 | 2 |

All 320 frozen base tensors match their original values in each of the nine
completed groups, tied storage is preserved, and all live parameters remain
unchanged. Regression results: 6 model-free exact tests, 15 full-codec GPU
component tests, and 3 new GPU precision regressions pass. The new tests verify
exact FP32 projection output/input derivatives, saved BF16 tensor identity,
checkpoint/adapter gradients, real BF16 vocabulary products and rejection of
trainable weights by the frozen-only operation.

## Provenance and limits

The [machine-readable summary](evidence/codec-feedback-v3-summary.json) pins
source hashes, axes, per-operation accounting, precision, input hashes, environment,
all nine child reports and the aggregate raw report. Local raw bundle:
`data/codec-qualification/feedback-v3-20261105/`, containing reports, logs,
source/test snapshots, input identities and `manifest.json`. Model weights,
checkpoints and raw generated artifacts remain excluded from git.

Aggregate report SHA256: `157b1dcb196da3fb7f97295b59e29ca9f19a644b7623cf7795336e4c359307e4`.
Bundle manifest SHA256: `5ee126565981d1ac28db62df3bfc34fb6142b6b1ebdea842a5060289fb437cc4`.
Frozen diagnostic SHA256: `1dd2e8527e14caa9002d672b4187253cdc174fb3c204ab56e80f257f23943640`.
Environment: NVIDIA driver 615.71.10, Torch 2.15.0+cu132, CUDA 13.2,
Transformers 5.22.1, PEFT 0.21.0,
safetensors 0.8.0. No CPU model execution or fallback.

The original G1 sources/checkpoints and Astra review remain unchanged. The
prescribed fixtures use an explicit thinking-budget override, supplied operations/
statuses and native continuation; they are engineering traces, not unforced
invocation or semantic-answer successes. Newly introduced heads remain untrained.
Source states are contextual production-tensor transfers, not clean withheld-fact
semantic tests. No training, parameter fitting, reviewed-data expansion,
frontend change, deployment or production checkpoint promotion occurred.
**1024 positions, maximum context, semantic promotion and joint training remain
unqualified.** This pass does not authorize expansion beyond the user's scope.
