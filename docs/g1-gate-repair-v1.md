# G1 gate diagnosis and bounded repair

Execution result: [G1-core PASSED on RTX 3090](g1-gate-results-2026-11-05.md).
All 72 numeric cases, 36 released continuations and supporting controls passed.
No downstream expansion followed.

Scope: the user's authorization to diagnose released Noul selecting Choice,
repair it without forcing Continue or expanding the codec, and rerun G1 on the
RTX 3090. The existing compensated numerical profile and tolerances remain fixed.
No service, production capability checkpoint, data curriculum or frontend change.

## Diagnosis

The original frozen diagnostic checkpoint's immediate Noul return has raw gate
logits `[.381041, .335342, 1.039501, -2.192952]` for
Continue/Noul/Choice/Score. Choice genuinely wins, with probability .487417;
Continue has probability .252310. The training and released boundary agree.
There is no tokenizer-ID interpretation, adapter-switch or cache discrepancy.

The two-window engineering bootstrap moved the gate weights by at most .00040065
from seeded initialization (relative RMS .00412739). It failed both pre-call
labels too: raw argmax was Continue at the prescribed Noul and Score requests.
Its immediate Noul Continue-label CE was 1.37710. Two small shared optimizer
updates did not establish a usable discrete readout. Mean discrete loss mixed
gate, writer and status objectives, and the original traces supervised only the
immediate return gate, omitting later continuation boundaries.

An additional supervision-timing bug affected request labels: `call()` reset
writer/record counters and incremented attempts BEFORE evaluating the gate loss.
Inference evaluates the eligible native boundary BEFORE those mutations. Move
the supervised gate evaluation before request mutation so training reads the
same inputs. This does not change the actual request or its position accounting.

Evidence is preserved in ignored `data/g1-proof/gate-diagnosis-20261105.json`
and `data/g1-gate-diagnosis.log`. The original parent checkpoint/report remain
unchanged. This is diagnosis of the disposable G1 head, not Qev's published
pretrained decision capability.

## Explicit diagnostic training amendment

The original <=16-episode bootstrap is not silently extended. Add one separate,
bounded readout-only fit to the existing diagnostic snapshot. This is training
of the proof's gate weights, not A4 capability training or curriculum expansion.

1. Collect detached CUDA control features from the same two prescribed G1
   teacher trajectories. The request boundary targets Noul/Score respectively;
   completed-return, fixed closure and subsequent native-answer boundaries
   target Continue. These are engineering control annotations, not semantic
   assessment labels or claims of optimal invocation.
2. There are 19 rows: 17 Continue, one Noul, one Score. Features are the existing
   256-wide `feature(hidden,counters)`. Target/fixture ID is not appended to
   them. Keep all four output classes, including Choice, in the softmax/CE.
3. Freeze the backbone, R/D adapters, shared control trunk, writer/evaluator and
   scalar rows. Fit only `codec.gate.weight` and `codec.gate.bias` with 256 fixed
   full-batch AdamW steps: lr .01, betas (.9,.95), eps 1e-8, wd 0, global clip 1.
   Average row CE within each observed class, then average the three class
   losses. No early stopping or schedule selection using certification results.
4. Require every training label to win the raw four-way logits. Save a new child
   checkpoint plus feature/annotation provenance. Assert only the two gate
   tensors differ from the original parent; never overwrite the parent/live
   pointer. The fitting math is CUDA-only; detached collection removes backbone
   gradients intentionally in this component-only diagnostic fit.
5. Run two preliminary continuations with ordinary learned argmax, then freeze
   the child before the full independent 72-case/36-continuation G1 collection.
   Repeated seeds/fixtures test deterministic execution, not generalization.

An initial isolated fit preceded the request-counter alignment. Its artifacts
remain in `data/g1-proof/gate-repair-20261105/`. The final aligned fit is separate
under `data/g1-proof/gate-repair-v2-20261105/`; do not pool their evidence.
Each fit collects two existing teacher episodes and validates two released
episodes, in addition to the unchanged full G1 matrix and optimizer verification.
The 256 optimizer steps reuse cached CUDA features, adding no model positions.
All attempts are recorded; neither fit is an extension to the general codec.

## Serving behavior and regression requirements

Release still uses all four learned logits and argmax. Choice is NOT masked for
being unimplemented: a winning Choice remains an explicit failure. Existing
exhausted-budget/attempt structural masks remain logged. There is no one-call
latch, unconditional Continue, semantic keyword router, extra prefill or cache.
Noul/Score remain callable after a return; their existing admission checks apply.

Add gate telemetry recording raw logits, selected actions, resource masks and
executed actions. Strengthen G1 acceptance: both prescribed call types AND
post-return Continue must win raw logits on all 144 supervised boundaries in
the 72-case matrix, preventing an always-Continue head from passing. Every
released executed action must equal raw argmax except the original logged
resource-limit masks. Do not relax any previous numeric or continuation gate.

Regression checks exercise pre-mutation supervision and prove a raw Choice is
still rejected rather than replaced. The complete G1 result must be measured
before claiming this repair passes; no downstream expansion follows this task.
