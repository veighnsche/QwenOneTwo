# A4: training run profile v1

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

Owner: A4 phase lengths, optimizer groups/schedule, exact loss subtypes,
correction allocation and development promotion fixtures. The
[loss configuration](loss-configuration-and-training-schedule.md) owns top-level
coefficients, eligibility/normalization and gradient boundaries; this profile
specializes them without replacing them. Follow [A2](model-configuration.md),
[A3](numeric-profile-v1.md), [data operations](data-operations.md),
[readiness](training-data-and-readiness.md) and the
[implementation specification](architecture-and-training-plan.md).

## 1. Prerequisites and scheduled phases

Short G1 GPU feasibility and reviewed pilot material precede capability training
and full data scale-up. All pretrained W parameters remain frozen throughout.
Each scheduled optimizer window contains eight scheduled examples/records,
microbatch one, with complete-episode backward and gradient accumulation. Step
only after the window. Normalize by component eligibility over the whole window,
not equal weighting of unequal microbatches.

| Phase | Updates | Trainable components | Guidance |
|---|---:|---|---|
| Reference codec | 1000 | Reference pooler/codec and reconstruction adapter T | Role-whitelisted semantic records; request targets exclude current outcomes; Reason may describe justified non-assessed status/support |
| Components | 2000 | Relevant R, D, live writer/reader/protocol heads and scalar rows | Reviewed writer actions; reference receiver packets; gold returns only in isolated reader exercises |
| Guided joint | 1000 | R, D and applicable live protocol/readout components together | Full compatible reviewed decision prefix; predicted slots; actual Qev returns |
| Transition | 2000 | Same joint components | Contiguous reviewed prefix decreases to zero |
| Free-control joint | 1000 | Same joint components | No forced decisions, substituted slots or gold returns |

Total: 7000 scheduled update windows/56000 presentations, including reference
records and repetitions, not 56000 independent authored scenarios. Reference
preparation is a separate semantic component task, not subject to the 25% replay
mixture for native/decision training. Freeze/version the reference encoder,
decoder and reference codec after passing its phase. Initialize compatible live
codec components from the snapshot with independent storage, as required by A2.
Do not reset live components at later phase boundaries.

For transition k=0..1999:

    q = 1 - k / 1999
    guided_events = floor(q * annotated_decision_events)

Use only a compatible contiguous prefix; no gold actions pasted back after
incompatible divergence. Joint runs always use predicted semantic slots and
actual Qev returns. Native answer-token teacher forcing remains allowed against
an approved continuation for the actual state; it is logged separately. Free
evaluation generates native text too, with no hints/forcing/overrides.

| Length/control | Limit |
|---|---|
| Reference annotation encoding | <=256 tokens |
| Reference reconstruction context | <=512 positions |
| Complete component episode | <=512 total positions |
| Complete joint episode | <=1024 total positions |
| Early component/guided B override | 512, explicitly logged |
| Early component/guided A | 128 |

Qualify A3 numerics/gradients at 1024 positions before joint training. Remove B
overrides throughout transition and free-control training. No override is
autonomous budget evidence. Length ceilings are training guards, not changes to
C=262144 or the predicted/granted budget. An overlength episode is a failed
scheduled exposure with no truncated-prefix backward or fabricated successful
continuation; do not silently replace it with an easier example. Report actual
eligible counts. Qualify fixtures under their declared allowances; do not enlarge
the allowance invisibly to complete a packet.

## 2. Sampling and data quantities

Retain 6000 core + 2000 replay training examples, 1000 development and 2000
locked tests. These are dataset quantities, not independent-scenario claims.

| Component window | Examples |
|---|---:|
| Writer exercises | 2 |
| Receiver exercises | 2 |
| Isolated reader exercises | 2 |
| Replay | 2 |

| Joint window | Examples |
|---|---:|
| Interleaved trajectories | 4 |
| Direct-answer trajectories | 2 |
| Replay | 2 |

Split replay into 1000 ordinary-language and 1000 reviewed decision examples;
take one of each per native/decision window. Balance operations over sampling
cycles, with component decision sampling 50% Assessed and the remainder balanced
over that operation's supported non-assessed statuses. Preserve domain coverage.
This is training sampling, never a runtime invocation quota. Review effort labels
independently; no operation-to-budget mapping.

Training seed 17; checkpoint sampler order/cursors and relevant RNG state. A3
seeds 17/29/43 are verification seeds, not an automatic three-full-run schedule.
All splits, variants, approvals and source provenance follow the data owners.

## 3. Optimizer groups and schedule

AdamW; all trainable masters/moments FP32 on CUDA. Register/deduplicate shared
parameters once. Frozen base/reference artifacts are excluded from optimizers.

| Group | Peak LR | Weight decay |
|---|---:|---:|
| Native R | 1e-4 | 0 |
| Qev-initialized D | 2e-5 | 0 |
| Inherited Qev Q/K matrices | 2e-5 | 0 |
| New writer/reader/protocol/reference-pooling projections | 2e-4 | 0.01 |
| New biases, embeddings, queries and Noul prototypes | 2e-4 | 0 |
| Reconstruction T | 1e-4 | 0 |
| Budget and Score row masters | 1e-4 | 0 |

Exclude biases, learned query/prototype vectors and embedding tables from decay.
The [exact module/optimizer contract](module-and-optimizer-contract-v1.md) owns
the complete inventory, sharing and per-component CUDA gradient buffers.
Normalize each component by its window's eligible EXAMPLE count, combine once,
then clip/step; never divide a partly eligible component by a fixed eight.
Frozen reference artifacts leave the optimizer after preparation; copied live
codec parameters start fresh optimizer state. Later live phases retain moments.
The row LR is the existing nominal/peak 1e-4 with the phase multiplier below;
row isolation and inactive grad=None remain unchanged.

    betas = (0.9, 0.95)
    epsilon = 1e-8
    global_gradient_norm_limit = 1.0
    foreach = false
    fused = false
    capturable = true

Optimizer tensor state/arithmetic remains on CUDA. capturable does not enable
CUDA graphs; A3 keeps graphs/compilation disabled. BF16 uses no gradient scaler.
Check A3 unscaled/unclipped gradients before clipping. Inactive parameters retain
grad=None, with no momentum/decay update. In particular, never create a zero
gradient to make an inactive scalar row participate in a step.

For each phase with U windows, set w=ceil(0.05*U). With update j=1..U, apply:

    multiplier = j/w                                  if j <= w
    multiplier = 0.1 + 0.45*(1+cos(pi*(j-w)/(U-w)))   otherwise
    learning_rate = group_peak_learning_rate * multiplier

Warmup is 50/100/50/100/50 updates; each phase ends at 10% of its peak LR.
Carry Adam moments/parameter step counters for continuing parameters; reset only
the phase LR schedule. Freeze the reference artifacts rather than stepping their
old state. Preserve schedule/counter/optimizer state when resuming a checkpoint.

## 4. Exact loss subtypes

Keep top-level coefficients/masks from the loss owner. Average eligible
subcomponents per field, fields per example and eligible examples per window.

| Target | Loss |
|---|---|
| Gate/Choice approved set | Negative log total probability of the approved set |
| Writer action/count/type/mode/status | Cross-entropy |
| Presence/constructed Boolean | BCE with logits |
| Constructed int64 | Mean CE across eight byte predictions |
| Constructed decimal | Mean of coefficient-byte loss and 37-way exponent CE |
| Constructed UTF-8 | Mean of length CE and mean active-byte CE; omit byte term when empty |
| Object/source/span/boundary references | Pointer CE |
| Semantic alignment | MSE over canonical normalized payload coordinates |
| Reconstruction | Token CE through frozen decoder |
| Candidate feasibility | Four-way per-entry CE: feasible/infeasible/insufficient/conflicting |
| Noul | Stable BCE on true/false logit difference |
| Task Score | Raw-readout MSE or squared distance to approved interval |
| Startup effort | Raw-readout MSE against independently reviewed effort target |

No floating regression for exact ints/decimals: preserve exact bytes and scale.
Copied/parsed values receive applicable mode/pointer supervision, no fabricated
constructed-byte targets. Charge reference-valued attributes under pointer loss
once, not again under exact-value loss. Continuous returns are not byte targets.
An approved point and an interval-only target are alternative regression modes,
not automatically two objectives. No label smoothing, automatic class weights,
extra call/avoidance reward or implicit token-minimization loss.

Contrastive temperature 0.1; only approved equivalent positives and reviewed
meaning-changing negatives. No valid contrast means no contrastive contribution.
Use multi-positive InfoNCE within the same semantic role/shape: flatten canonical
normalized payload slots, compute FP32 cosine similarity divided by 0.1, and
take -log(sum(exp(logits_positive))/sum(exp(logits_positive_and_negative))) with
stable log-sum-exp. Reference targets are detached. Exclude the anchor itself
and unreviewed relations; do not treat incidental same-batch examples as negatives.
Preserve per-role normalization and the reference-validity masks. Mechanical
failure/cancellation/exhaustion never acquire a fabricated semantic status/value.

## 5. Reviewed correction material

Reserve 400 EXISTING core examples: 100 within writer/reader material and 300
within complete trajectories. For each of five families, allocate 80 examples,
20 writer/reader and 60 complete trajectories:

| Family | Target behavior |
|---|---|
| Wrong question/rubric | Repair intended meaning; do not inherit the old outcome label |
| Missing/wrong alternatives/attributes | Repair bank; never select an absent entry |
| Wrong binding/lifecycle | Reject stale/foreign/consumed result |
| Non-assessed/contradictory result | Qualify, seek missing evidence or recognize conflict |
| Malformed output/resource failure | Revise where permitted, decline or terminate without a semantic value |

Record actual snapshot, observed problem, valid next actions, eligible losses and
acceptable continuation. These labels are offline supervision, not inference
prompts or new semantic result statuses. Unsealed revisions obey existing bounds;
a sealed replacement is a new attempt/snapshot. Cancel means stop/discard late
results; do not teach a generated apology after cancellation. Cancellation-only
fixtures have mechanical targets, not fabricated native answer labels.

Known reviewed/intervened packet semantics can support corrected targets. An
arbitrary generated latent packet cannot be relabelled from an unvalidated gloss.
Keep supported formation losses, mask unsupported outcome/continuation targets,
and count the failure. Do not restore the original happy-path answer blindly or
replace actual joint returns with gold values.

## 6. Development fixture allocation and promotion

The [promotion matrix v1](promotion-matrix-v1.md) specializes this section:
W/E=70/130 within200, reader's13-cell allocation, exact intervention groups,
mechanical secondary protocol, minimum family/status coverage, per-cell gates,
effort metrics and separate guided/free protocols. Preserve phase lengths and
final-two-checkpoint rule. No matrix fixture, baseline or run is instantiated.

Partition the existing 1000 development examples; related variants stay together:

| Collection | Examples |
|---|---:|
| Reference-codec semantic recovery | 100 |
| Writer/receiver components | 200 |
| Isolated result reading | 100 |
| Semantic interventions | 200 (100 contrast pairs) |
| Free complete jobs | 400 |

No original discussed prompts/close paraphrases or locked-test cases enter these
collections. Record slice/denominator/route judgements before running evaluation.

| Phase | Development promotion |
|---|---|
| Reference/receiver compatibility | [Latent semantic/behavioral protocol](reference-behavior-v1.md): existing component thresholds and complete applicable intervention pairs; canonical text recovery is diagnostic |
| Components | >=95% fully correct writer structure/typed fields; >=90% semantic success overall and >=85% per operation; >=95% status accuracy; >=90% reviewed Score agreement; >=95% isolated-reader success |
| Guided/transition | >=90% complete-flow success under DECLARED guidance and >=90% applicable intervention-group success; not autonomous evidence |
| Free control | Same semantic/readiness thresholds on unforced runs; >=50 valid unforced calls per operation across multiple domains; >=100 successful direct-answer jobs |
| Every phase | Applicable A3 checks; zero observed binding/cache/accounting/GPU-only violations; fabricated values <=1% of non-assessable cases |

Frozen candidate references may supply exactly bound experimental supervision
without a passing reconstruction or receiver certificate. This permits constructing
learning inputs; it does not authorize a run, assert semantic acceptance, or promote
weights. Diagnostic fixtures remain ineligible. Receiver acceptance binds the
actual tested reference and receiver and has the limited scope above. Learned
invocation, Reason support meaning and native result use require separate evidence.

Always count failed/rejected/overlength cases in the applicable denominators.
Missing/insufficient operation/status coverage is incomplete readiness, not a pass.
Free-control training may still teacher-force approved native text; promotion
evaluation has no forcing of either decisions or native text. Numeric parity is
reported separately from these semantic percentages and never inferred from them.

Native replay NLL may rise at most 5% from a fixed matched pre-training baseline.
Decision replay must reach >=90% accuracy after components, then remain within
two percentage points of that promoted component checkpoint in joint phases.
Fix the replay collection, baseline interface and masks; report these diagnostics
separately from task improvement or claims of complete original-Qev preservation.
Capture the native baseline after reference freeze and before the first component
update, with R still at its initialization. Use a fixed reviewed direct-answer
development slice for matched native-token NLL, scored as an explicitly guided
diagnostic separate from free-job evaluation. Use a fixed reviewed receiver
development slice for decision replay; its post-component reference is the
promoted final component checkpoint. Concrete slice IDs/hashes belong in the
actual manifest, without adding to or drawing from the locked test set.

Checkpoint/evaluate every 100 updates. Finish the prescribed block and pass at
its FINAL TWO checkpoints (U-100 and U) before advancing. Failure requires a
documented development revision, not automatic extra epochs, relaxed thresholds
or selection of an earlier favorable checkpoint. Preserve lineage on revision.

The 2000 locked tests (500 component, 500 interventions, 1000 free prompts) remain
untouched until development gates pass. Training/development promotion does not
waive final locked gates or G6 length expansion. Testing to 1024 positions does
not establish reliability at full configured context or maximum thinking budget.

## 7. Concrete run artifact

Before any authorized job, instantiate a manifest with immutable model/adapter/
codec/tokenizer/data/split/config hashes, actual approved fixture IDs and reviewer
records, GPU/library/kernel identities, seed/RNG/sampler state, phase lengths,
optimizer groups/schedule/state, forcing flags/overrides, mask eligibility,
length/position ledger, baseline versions and promotion reports. This document
sets defaults; it does not claim those artifacts or reviews already exist.
