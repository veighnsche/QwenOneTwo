# G1 precision diagnosis and rerun — 2026-11-05

**Numeric gate: PASSED. Overall G1-core: FAILED on released Noul continuation.**
No expansion, capability training, live service change or deployment followed.
All model execution and tensor comparisons ran on RTX 3090 UUID
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1`; there was no CPU fallback.

## Root cause and correction

The old backend repeatedly discarded low-order activation/update information at
BF16 projections, LoRA additions, residual sums and gradient casts. Error was
already present in the first prefill layer, before adapter switching. Matching
the reference's adapter/input rounding did not fix it. FP32 residual/state
islands alone also failed the state tolerance. The controlled ablations and
arithmetic are recorded in [compensated precision v2](g1-compensated-precision-v2.md).

The corrected candidate uses compensated BF16 dense forward products with FP32
outputs/accumulation, FP32 residual/state/nonlinear islands and FP32 master/input
gradients. Frozen pretrained weights retain their original BF16 bytes. This
costs extra products, memory and runtime. It is an explicit revised numerical
profile, not ordinary single-product BF16. The genuine FP32 reference, original
diagnostic weights, numeric tolerances and position rules were retained.

## Numerical results

Errors below are relative to the genuine FP32 reference; RMS errors are fractions.

| Measurement | Noul before | Noul fixed | Score before | Score fixed | Unchanged limit |
|---|---:|---:|---:|---:|---:|
| Worst hidden/cache normalized RMS | .0366066 | .0000417819 | .0867262 | .0000298173 | .01 |
| Semantic-slot gradient normalized RMS | .177832 | .000131086 | .464958 | .000126932 | .05 |
| Writer-R gradient normalized RMS | .160356 | .000121973 | .635150 | .000126400 | .05 |
| D gradient normalized RMS | .119125 | .0000885185 | .448349 | .000114755 | .05 |
| Native-R gradient normalized RMS | .107472 | .0000665583 | .343601 | .000118618 | .05 |
| Returned-value absolute difference | .0000773668 | 5.96046e-8 | .0106845 | 1.49012e-7 | .002 |

All pointwise state/logit bounds and applicable gradient-cosine bounds passed.
Minimum cross-precision gradient cosine was .9999999930130797 (limit .995).
The reference returned values remain exactly .5058713555335999 and
.2949802279472351, matching the original failed run. They were not moved toward
the candidate. Candidate values are .5058714151382446 and .29498007893562317.

The complete 72-case matrix passed: two fixtures, four profiles (functional
candidate, checkpointed candidate, serving candidate, FP32 reference), seeds
17/29/43, three repetitions. Serving/checkpoint comparisons matched the candidate
exactly at sampled boundaries; checkpointed gradient errors were zero/cosine one.
The first candidate in each fixture is the comparison baseline, not an independent
FP32 comparison. Counters confirm BF16 forward products in candidate runs and
zero compensated products in reference runs. Physical checkpoint recomputation
does not add logical positions or prefills.

## Cache, gradients, accounting and optimizer

| Prescribed trace | P | Evidence m | B | A | H | Total | Prefills | Cache lineages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Noul during thinking | 33 | 9 | 44 | 8 | 6 | 91 | 1 | 1 |
| Score during answering | 52 | 9 | 67 | 8 | 6 | 133 | 1 | 1 |

Both execute D → R → D → R, consume the return once, release 36 unused held
positions and leave no held positions. Source-state version checks and causal
cache-length assertions passed. Finite/nonzero native-loss gradients reach the
returned value, request packet, semantic slots, writer R, evaluating D and reader R.

All four detachment negative controls, all 15 accounting controls (three each of
zero budget, exact fit, one short, duplicate delivery and cancellation), and four
GPU FP64 local derivative checks passed. Four focused precision GPU regressions
and five model-free bookkeeping tests passed.

The eight-example disposable optimizer verification matched an independent CUDA
FP64 component-VJP aggregation: normalized RMS 5.44187546e-8, cosine
.9999999999999987. Eligible denominators remained native 8/discrete 8/calibration 4.
R/D/Score changed; budget did not. Masters were restored after this check. There
was no new bootstrap to improve the frozen comparison checkpoint.

All 320 frozen base tensors matched the original model on GPU. Shared scalar-row
provider, inactive-row momentum isolation, vocabulary exclusion and export checks
passed; exported readout differences were zero.

Largest observed allocation across numeric profiles was 15.50 GiB; reserved peak
was 15.79 GiB. These are this proof process's allocations, excluding the existing
service, and establish no longer-context capacity or performance claim.

## Remaining failure and expansion gate

All 36 released-continuation attempts were collected. All 18 Noul attempts
selected Choice after consuming their initial result. Choice is outside the
bounded G1 codec, so each explicitly failed with:

    Released gate selected Choice: G1 subset cannot certify this continuation

All 18 Score attempts matched across functional/serving runs and produced the
same partial native text: `The user is asking for a fraction`, preceded by
newlines. They ended at the 16-position answer cap with repeated denied requests.
This is continuation parity, not a complete or semantically correct answer.
Prescribed initial calls are not proof of learned autonomous invocation.

Overall G1 therefore remains FAILED. Numeric success does not authorize codec,
data, training or frontend expansion. The continuation failure is preserved,
not hidden by forcing Continue or changing supported operations.

## Reproduction and preserved evidence

Read-only input checkpoint SHA256:
`50cc08803d20b70c71d75a6da3c325ef66b3c2c14d70f766b72cdf6d7ce8556c`.
Original [failed report](g1-core-results-2026-11-05.md) and raw artifacts remain
unchanged. New ignored artifacts are under `data/g1-proof/precision-fixed-20261105/`:
`report.json`, `manifest.json`, `fixtures.json`, row export and disposable optimizer
verification weights. `data/g1-precision-certify.log` preserves the full run.
Source hashes recorded before measurement match the final source files.

Final report SHA256:
`77f16d4b5ceb2ea8280ef6e75492a720149a3efebe32aee27c2a0f84f471e0cc`.

From the repository root, with GPU access (no sandbox driver restriction):

```bash
CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv/bin/python -m qwen_one_two.g1_proof \
  --snapshot /home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
  --checkpoint data/g1-proof/run-20261105/diagnostic.safetensors \
  --output data/g1-proof/precision-fixed-20261105 \
  --stage certify --precision-policy compensated
```

The process deliberately exits nonzero because overall certification failed.
Package versions, pinned artifact hashes, precision profile and per-case memory/
product counts are in the local manifest/report. These are diagnostic artifacts,
not production or capability checkpoints.
