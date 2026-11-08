# RD-07 technical qualification — 2026-11-08

All ten deferred RD-03..06 GPU checks and three new RD-07 checks pass on the
specified RTX3090. This qualifies the measured machinery. No training or
optimizer update occurred, and no learned semantic improvement is claimed.

Portable results, exact commands, executed source/profile/input hashes, failures,
counts and raw artifact hashes are in
[redesign-qualification-v1.json](evidence/redesign-qualification-v1.json).
Raw logs/manifests and executed source snapshot:
`data/redesign-rd07-20261108/`. Earlier failed attempts are retained unchanged.

## Recovery and execution

Started on clean main `42d6fa41fe554f2014526e83517cacddbc771e89`, matching the
authenticated remote. RD-01..06 deliveries, source/test/contract/raw identities,
recorded migrations and pushed ancestors were verified in order. Their staged
deliveries were complete; ten real-GPU checks were due here. Valid model-free
audits were reused. No later queue slice was executed.

Sandbox GPU access failed. Host access exposed the exact selected device:
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1`, RTX3090, PyTorch2.15.0+cu132,
CUDA13.2. `CUDA_VISIBLE_DEVICES` was set before model imports; the process saw
one CUDA device and checked its identity. No CPU/3060 model fallback occurred.
Existing services were inspected and left alone. The final inventory records
the original server PID2094 and no qualification process remaining.

The local diagnostic launchers execute existing unittest methods sequentially,
with one resident test backbone at a time. They block optimizer `step`, record
the actual PID and source hashes, instrument Session accounting, verify frozen
state at class cleanup, and reject source changes during execution. Temporary
checkpoint serialization is removed by the existing test. Historical parent
counters of 188 are restored state, **not** updates performed in this task.

## Measured results

Final deferred run: **10 tests pass**, 155.915 s, zero failures/errors/skips.
Final changed-path run: **3 tests pass**, 198.589 s, zero failures/errors/skips.
Every required ID is resolved in the portable manifest:

| IDs | Actual coverage |
|---|---|
| RD03-GPU-D-INDEPENDENCE, CACHE-PARITY, CONSUMER-GRADIENTS | Existing reviewed Noul input; frozen-W features ignore D changes; cached/uncached features and bindings agree; real consumer gradients and frozen W/reference |
| RD04-GPU-IDENTITY-PROJECTION, DIRECT-ROUNDTRIP, DIRECT-GRADIENTS | Exact identity initialization and legacy wire; variable role fields, exact attributes and immutable bindings; all 13 applicable operation/status cells; live D/projection gradients |
| RD05-GPU-GRADIENT-ACCUMULATION, RESUME | Eight eligible existing frozen-W/pooled exposures; actual joint status/Noul VJPs and exact weighted accumulation; parent/direct owners, nonzero historical moments, complete-window save/restore, sampler, Python/CUDA RNG and frozen owners |
| RD06-GPU-NATIVE-RETURN, CONNECTED-GRADIENT | Actual Noul/Choice/Score returns from original prefill states; native loss through actual Noul return, frozen receiver, continuous writer states and R; one prefill/cache and exact accounting |
| RD07-GPU-PROJECTION-FP64 | Nonzero deterministic projection delta; explicit FP64 residual-linear forward and input/weight/bias VJP reference on the GPU |
| RD07-GPU-LONG-PARITY | Twelve direct traces: three ports × functional/checkpoint/serving/explicit-FP32; all reach 1024; nine paired comparisons, including final hidden/cache state |
| RD07-GPU-CANCELLATION-ENVELOPES | RPC/pre-request/in-request/pre-return cancellation; exact cache/ledger positions, released holds, late-result rejection; request256 accepted, request257 atomically rejected, return64 accepted |

All five final class cleanup checks verify all **320 W tensors**, physical tying
and unchanged reference fingerprint
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.
Sources stayed unchanged during each final run; optimizer guard calls and actual
optimizer updates are both zero. Final changed-path instrumentation records
16 Sessions, 437 model steps and 12,496 causal positions across those Sessions.
The twelve long Sessions each have one prefill, lineage1 and zero held positions.

Projection FP32 versus FP64: maximum absolute forward error
`5.86734958396562e-08`, normalized RMS `2.2804070790904514e-08`.
Worst input/weight/bias VJP RMS is `5.531260724169399e-07`, minimum cosine
`0.999999999999848`. Long-trace checkpoint and serving comparisons are exact;
FP32 comparisons have worst logit RMS `2.5859035355531056e-06`, state RMS0,
worst D/projection VJP RMS `1.3087216238383911e-06`, minimum cosine
`0.999999999999294`. Existing pointwise/RMS/cosine thresholds are unchanged.

The historical 324-case 128/256/512 and 108-case 1024 matrices were hash-verified,
including all child reports. Reuse applies to unchanged arithmetic primitives
only: compensated GEMMs, feedback projection math, accounting and pinned cache
methods. Current interfaces are not certified by those older reports. Explicit
FP32/FP64 and twelve new direct traces cover the affected paths. The failed old
BF16 transformer arithmetic path was not reopened.

## Failures and repairs

`deferred-1`: nine passes, one error. Noul returned a separately computed `raw`
node that was a sibling of the node used in `value.sigmoid()`. The native
return-specific VJP could not differentiate against the reported node. Sharing
that exact subtraction node fixes provenance without changing forward arithmetic.
`deferred-2` then passed all ten checks; the final run repeats them after the
subsequent Session repair.

`changed-1`: projection and long parity passed, but two cancellation subcases
failed: before-request ledger58/cache50, before-return ledger67/cache58.
Session now checks cancellation before charging public boundaries and completes
already charged work before discarding a late output. The normal-path tensor
computation remains pinned by AST comparison. All four cancellation scenarios
pass in `changed-2`. That run also captures the final cache state at1024, extending
the initial run's final logits/VJPs with the explicit final state comparison.

The model-free regression command covered 89 tests in116.988 s; 88 passed and
one new source-contract assertion failed because it expected a literal space in
test source. Its whitespace-independent correction passed all three new contract
tests in0.061 s. Thus all **89 distinct model-free tests** pass on current code;
the unchanged 86 results are reused. Compilation and staged whitespace pass.
Four current-source CLI preparations (direct/repeat/frozen-pooled/legacy), with
model imports blocked, pass; direct plans are byte-identical and all execution
counters zero. Older source-bound preparations remain historical. Preparations
retain their conservative static `pending_RD07`/nonlegacy execution guard; this
report/manifest resolves qualification, without granting fit authorization or
removing admission guards.

## Limits and next work

Engineering fixtures force guided controls, Assessed status, Choice index0 where
applicable, budget512 and existing continuation tokens. Deterministic projection
deltas test arithmetic, not learning. No free learned invocation, corpus semantic
bridge, full-context/optimizer-memory qualification or final-answer improvement
is established. The connected native-loss VJP is specifically measured for Noul;
actual-return/accounting fixtures cover all three ports. New-profile joint Reason
and unsupported native supervision masks remain intact.

There are **no pending RD-07 technical checks**. Real-data limits remain:
direct receiver admission0/1500 train,0/130 E,0/180 X; complete direct formation0
across3220 writer/interleaved calls; native decision/post-call continuation and
direct supplied-reader support0. Typed native-final extraction and actual-native
X-pair labels are absent. Field32/request256/return64 limits are unchanged.
An explicit layout revision/re-audit and valid supervision are required before
affected fits. Historical profiling receipt/input drift remains unmodified and
is not reported as a passing check.

RD-08 is the earliest unfinished slice and was not started. This submission
authorizes no training. RD-09 also remains unstarted. No agents, new data/reviews,
locked access, deployment or service/Operating Table changes occurred.

Implementation/evidence commit `7da99c652fba8729df247c62fcb029fb931d5cea` is pushed
to main. Authenticated push exited0 (`42d6fa4..7da99c6 main -> main`); subsequent
`git ls-remote` returned the identical full SHA. Delivery finalization is committed
and pushed separately on top. Current source/test/raw hashes and190 local Markdown
targets verify; compilation and staged whitespace pass.
