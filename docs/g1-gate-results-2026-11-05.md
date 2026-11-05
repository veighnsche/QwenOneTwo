# G1 gate repair results — 2026-11-05

**Overall G1-core: PASSED**, with the compensated precision profile and the
isolated diagnostic gate-head repair. This is a bounded engineering proof, not
semantic promotion or general invocation readiness. No codec, curriculum,
capability-training pipeline, service or frontend expansion followed.
All model work, readout fitting, tensor comparisons and derivative checks ran on
RTX 3090 UUID `GPU-225509c3-d702-6c27-de03-76b5e99b33b1`. No CPU model workload ran.

## Root cause and repair

The original Noul return's four logits were Continue .381041, Noul .335342,
Choice 1.039501, Score -2.192952. Choice genuinely won, identically in training
and released execution. Two bootstrap optimizer updates left this random gate
undertrained, including incorrect raw pre-call Noul/Score decisions. The teacher
trace also omitted gate labels at later native-continuation boundaries.

Request supervision had a second bug: counters were incremented/reset before
the supervised gate read, while inference reads them before starting a request.
Supervision now reads that same pre-mutation boundary. Dispatch and model
position accounting are unchanged.

The [bounded repair protocol](g1-gate-repair-v1.md) records an explicit additional
component fit, separate from the original bootstrap: 19 detached CUDA features
from the existing two teacher trajectories, 256 fixed class-balanced four-way
CE/AdamW steps, fitting only the existing gate weight and bias. Seventeen rows
teach continuation; one each preserves Noul and Score invocation. Choice remains
in the softmax and may still win. Only the two gate tensors changed; independent
safetensors byte comparisons confirmed all other 802 diagnostic tensors identical.
The parent and old failure reports remain intact; the new child was frozen before
certification. These are engineering annotations, not a reviewed semantic dataset.

There is no runtime Continue substitution, Choice mask, one-call latch or new
operation. A regression test confirms raw Choice still raises the original
unsupported-operation error. The other regression checks pre-mutation timing.

## Complete rerun

| Check | Measured result |
|---|---|
| Numeric matrix | 72/72 passed: 2 fixtures × 4 profiles × 3 seeds × 3 repeats |
| Raw gate supervision | 144/144 correct, including Noul/Score calls AND post-return Continue |
| Released continuations | 36/36 passed: 2 fixtures × functional/serving × 3 seeds × 3 repeats |
| Released gate executions | 612 recorded boundaries; all raw argmax, 0 resource masks |
| Connected gradients | Returned scalar → Qev → request packet/slots → writer R, and reader R, finite/nonzero |
| Negative controls | 4/4 passed |
| Accounting controls | 15/15 passed |
| Local CUDA FP64 derivatives | 4/4 passed |
| Scalar rows/export/momentum | Passed; zero export differences |
| Optimizer verification | 8 examples; normalized error 5.64737209e-08, cosine 0.9999999999999982 |
| Frozen base | 320/320 original tensors unchanged |
| Focused regressions | 2 gate checks and 5 model-free bookkeeping checks passed |

Numeric thresholds were not relaxed. Noul/Score worst state normalized RMS
remained 4.17818842e-05 / 2.98172666e-05
(limit .01); scalar absolute differences remained
5.96046448e-08 / 1.49011612e-07
(limit .002). Checkpointed gradients matched exactly. Largest allocation across
numeric profiles was 15.50 GiB, reserved 15.79 GiB, excluding existing
service allocations. This is no full-context capacity or performance claim.

## Cache, budget and native continuation

Prescribed core traces remain one prefill/one cache lineage, D → R → D → R:
Noul P33+B44+A8+H6=91; Score P52+B67+A8+H6=133. They consume each return once,
release 36 unused held positions and leave no remaining hold.

The released gate is evaluated freely after the prescribed first call. Noul's
learned gate requests one additional existing Noul call, then continues native
text; it is not forced through a Continue-only path. Score uses its first call.
Functional/serving text, accounting and stopping match in all repeats:

| Released fixture | P | B | A | H | Total | Dispatched | Prefills / cache lineages |
|---|---:|---:|---:|---:|---:|---:|---:|
| Noul | 33 | 86 | 16 | 6 | 141 | 2 | 1 / 1 |
| Score | 52 | 67 | 16 | 6 | 141 | 1 | 1 / 1 |

Noul's decoded generated continuation includes `Yes, Jo may enter.`; Score's
includes `Fraction blocked: 0.`. Both end Limited at the original 16-position
answer cap. These are partial native continuations, not complete or semantically
validated answers. The prescribed first calls/manual effort override remain
explicit. This does not qualify general successive-call registry interoperability,
Choice/non-assessed codecs, 512/1024 positions, learned latent meaning, autonomous
initial invocation, optimal budget or A4 capability readiness.
The subset ledger's record/released-hold summary describes its latest request;
it is not a cumulative immutable-record registry audit across successive calls.

## Evidence and reproduction

New ignored bundle: `data/g1-proof/gate-fixed-20261105/` with `report.json`,
`manifest.json`, fixture definitions, row export and disposable optimizer check.
Report SHA256: `ec402e2a8cb9625cbc85be4e15886b64aac4fb133766902cc90847859ae2ece5`.

Frozen child: `data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors`;
SHA256 `1dd2e8527e14caa9002d672b4187253cdc174fb3c204ab56e80f257f23943640`.
Parent SHA256 `50cc08803d20b70c71d75a6da3c325ef66b3c2c14d70f766b72cdf6d7ce8556c`.
The repair directory also preserves annotations, CUDA feature data and fit/
preliminary validation results. The initial pre-alignment fit remains separately
recorded; its evidence is not pooled into this certification.

Run the existing G1 command with `--precision-policy compensated`,
`--checkpoint data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors` and
`--output data/g1-proof/gate-fixed-20261105`, explicitly setting
`CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. The pinned snapshot path and software/hardware
versions are in the manifest/report. The complete log is `data/g1-gate-certify.log`.
All ten measured G1 source hashes match the final files.

The [initial failure](g1-core-results-2026-11-05.md) and
[precision-only rerun](g1-precision-results-2026-11-05.md) are historical evidence.
This pass closes the bounded G1-core gates under these versioned profiles.
The user's current task remains limited to diagnosis, repair and rerun;
downstream expansion has not begun.
