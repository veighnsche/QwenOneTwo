# Throughput kernel screening — 2026-11-07

The next CPU-06 candidates were tested on the selected RTX 3090 with existing
weights/fixtures. **None was adopted.** The final repository keeps the original
model arithmetic and finite predicates. The useful next measured direction was
resident application-worker concurrency, handled by the existing dispatcher.
No agents, training, new semantic data/reviews or live-service changes occurred.

## Screening results

| Candidate | Actual evidence | Decision |
|---|---|---|
| Replay full frozen Qwen RMS normalization | Four GPU tests pass with exact values/input derivatives, checkpoint/stream/composition checks. Isolated speedup 1.160–1.179× across five shapes. | Not selected: modest isolated benefit; no end-to-end benefit certified. |
| Retain PyTorch square/mean, fuse only pointwise RMS tail | Exact operator checks pass. Standard Triton JIT launch is slower; compiled launch improves the isolated operation only about 1.09–1.12× in the final pointer-dispatch screen. | Not selected. Prototype source/logs retained locally; no active backend or dependency added. |
| PyTorch native fused RMS normalization | Isolated 1.535–1.584× speedup and very small local FP32 errors. **First full-feedback comparison fails strict state and gradient limits.** | Rejected; no tolerance change or promotion. |
| Single-CTA finite predicate at the same boundary | Four GPU tests pass on FP16/BF16/FP32/FP64 extremes, NaN/Inf positions, readonly/version/gradient identity and original error labels. Isolated predicate speedup only 1.026–1.044×. | Not selected; existing CUDA predicates restored. |

These are local operator screens, not learned capability or full-workload
performance qualifications. Early launch variants and failed runs remain in
`data/cpu-06-20261107/`; no unsuccessful implementation is silently enabled.
The pointwise experiment retained the original PyTorch reduction and disabled
FMA fusion; reducing kernel count alone did not overcome host launch overhead.
Classification in the finite prototype used exponent bits and an integer AND
reduction, not a floating norm that could overflow on valid values.

## Why native RMS was rejected

`native-qualification/report.json` requests 128/512/1024 positions and all three
ports. It executes the eager/native **128-position Noul pair**, with connected
probes and checkpointing, then stops because that prerequisite fails. The other
eight pairs were **NOT RUN**. This is numeric rejection, not an elapsed-time cap.

Against the unchanged FP32 oracle:

- Maximum state normalized RMS: **7.987952127127693e-5**, exceeding **1e-5**.
- Maximum state absolute difference: **0.0019115805625915527**.
- Gradient normalized RMS: R **0.001983626456644852**, D
  **0.0033141260298677863**, wire **0.005544496300927236**, emitter
  **0.0009098620133089888**; all exceed the existing FP32 **1e-4** limit.
- Scalar difference is only **1.430511474609375e-6** and startup-score difference
  is zero. Matching these readouts alone would have missed the state/gradient
  failure. Original exact control/accounting checks remain in the comparison.

The independent script and failed report/log are retained. It constructs no
optimizer and performs zero updates. Because it exits on numeric rejection,
its final frozen-weight verification block is not executed; no final integrity
result is fabricated for this failed candidate. The original v3 profile and its
324/108-case evidence remain unchanged. Original BF16 transformer arithmetic
still fails; none of these screens offers a replacement for it.

## Current scope

The model, codec, budget mapping, reference encoder and actual case evaluator
remain byte-identical to commit `c8e9daf`. Temporary kernels/tests are archived
under `graph-screen-source/`, `pointwise-screen-source/` and
`finite-screen-source/` in the raw evidence directory. Native RMS uses an inline
instance override in its saved launcher, not an installed-library modification.

The archive is historical experiment code, not an entrypoint to run against an
arbitrary current checkout. Restore its recorded sources in an isolated copy
when reproducing a screen. The final pointwise source corresponds to the last
compiled-pointer variant; preliminary JIT timings are screening observations,
not an independently source-qualified backend release.

[Portable screening evidence](evidence/throughput-kernel-screens-v1.json) binds
raw scripts, logs, results and archived sources. Further fusion work must satisfy
the original FP32/gradient limits and every failure boundary before adoption.
