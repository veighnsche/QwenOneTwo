# QwenOneTwo implementation specification

## Status, authority and implementation entry point

This is the consolidated target specification: requirements and versioned
starting designs. The [current handoff](next-agent-handoff.md) owns implementation
status, measured evidence and remaining work. The
[documentation map](README.md) assigns ownership to detailed topic contracts.
User instructions and AGENTS.md take precedence; within the docs this file
resolves cross-component conflicts. Examples do not override the specification.

The existing prototype demonstrates bounded routing and switching. It does not
establish general request formation, rubric/candidate generation, native result
understanding or a fully differentiable alternating-cache training path.

### Final decision status

The packet, writer, semantic-supervision, loss, scalar-row, numeric-verification
and data-operations decisions are now recorded. Their starting defaults are
summarized below and owned by the linked topic contracts. They are not validated
capabilities. The [final decision audit](final-decision-audit.md) identifies
recorded A1-A4 and [B1 neural input encoding](neural-input-encoding-v1.md), plus
reconciled interpretations and remaining contract/operational/empirical work.
The [Astra review](review-astra-2026-11-05.md) supersedes the earlier all-choices-
closed assessment. F1-F8 responses are now recorded in their topic owners;
independent re-review, actual artifacts, implementation and GPU evidence remain
unperformed. No G0/GPU gate passes by recording settings.

## 1. Requirements and explicit scope

| ID | Required behavior |
|---|---|
| R1 | One human submission starts automatic budget selection, optional latent decisions and a normal answer in one inference job |
| R2 | One prompt prefill and one causal hybrid-state history; no separate decision request, second button, prompt replay or second prefill |
| R3 | Qwen learns when to invoke Noul/Choice/Score and forms the question, alternatives or rubric itself |
| R4 | The initial thinking budget is determined by one continuous score and the retained S-curve, with maximum half the pinned context |
| R5 | Runtime call/return uses vectors, indices, scalars and typed metadata, without generated tool-question strings or JSON |
| R6 | Model workloads use explicitly selected RTX 3090 UUID GPU-225509c3-d702-6c27-de03-76b5e99b33b1; no CPU fallback |
| R7 | Runtime owns transport, validation and limits; no keyword routing, domain eligibility rules, gold-answer substitution or hidden semantic scorer |
| R8 | Calls may occur during thinking or answering within remaining allowance; no call and multiple serial calls can both be legitimate |
| R9 | Native Qwen is trained to interpret bound results and continue the user's task; bridge-only fitting is insufficient |
| R10 | Preserve cancellation, immutable checkpoint provenance and independence from The Operating Table |

Initial scope is text input/output, one pending request and no nested or
asynchronous decision calls. No refunds, calendar invitations, bookings or other
external actions are executed by these examples. New human input queues a new
job or cancels the old one; it does not mutate a pending request.

Textless describes the deployed internal interface. Runtime memory still holds
request-specific data; all such data is not stored in weights. Offline English
annotations and training-only semantic supervision are allowed. No second
translator LLM is required in deployed inference.

## 2. Shared backbone, trainable extensions and modes

Pinned identities, read from current repository constants:

| Item | Pinned value |
|---|---|
| Base | Qwen/Qwen3.5-0.8B |
| Base revision | 2fc06364715b967f1860aea9cf38778875588b17 |
| Qev source | twainsk/qev-0.8b |
| Qev revision | 1d2a6d7f891daab7483a49c3abe507c0c9d7ea62 |
| Hidden width / configured context | 1024 / 262144 |

One base W is resident. Keep pretrained parameters frozen. R is the trainable
native adapter for writing requests, reasoning, result reading and answering.
D initializes from Qev and is aligned as a versioned decision adapter. Preserve
the original Qev release unchanged. Shared protocol writers/readers/heads remain
resident in the same model bundle.

Native mode is W+R; decision mode is W+D. Do not silently stack W+R+D. Both
adapters remain registered as trainable during joint training; mode selection
controls contribution, not global requires_grad toggles. No merge/unmerge or
weight replacement inside an active training graph.

Budget and task Score have distinct scalar regression rows in the actual
LM-head matrix, outside the tokenizer's generatable vocabulary and masked from
text sampling. These are explicit learned extensions, not permission to unfreeze
the whole pretrained matrix. The [scalar-row design](scalar-row-handling.md)
reserves IDs 248077/248078 with two independent FP32 row masters over the frozen
tied table, BF16 effective row values and FP32 scalar accumulation. Native text
loss/sampling excludes private and unused IDs; load-time serving installation
preserves physical tying without changing cached base artifacts. Gradient and
export equivalence still need GPU verification. No categorical budget head or
weighted average over budget bins is used.

The [model profile](model-configuration.md) records A2: R rank/alpha 8/16,
D 16/32 initialized from Qev and T 8/16 for offline reconstruction, all with
zero dropout. D's source 0.05 dropout remains in the original immutable artifact;
the aligned zero-dropout profile is a versioned change. Protocol width is 256,
attention has eight 32-wide heads, reference pooling has two blocks and draft
reading one. Canonical N has RMS floor 1e-6/no affine gain; wire scale derives
once from frozen valid-token embedding RMS. Both deployed modes share one live
codec; frozen reference pooling/T/codec use independent immutable small weights
with the same W. The profile owns exact tables, projections/head shapes and
FP32/BF16 boundaries; no shape or performance test has run by documenting it.

## 3. Single-job state machine and dispatch timing

1. **Prefill / W+D:** process the human input once; read the fixed initial budget
   score. This startup port is not an optional operation-gate action.
2. **Native / W+R:** reason or answer. At eligible continuation boundaries the
   learned policy chooses continue, Noul, Choice or Score.
3. **Draft / W+R:** a non-continue action expresses operation intent and begins
   bounded latent argument formation. Existing immutable records may be reused.
   The writer produces fields/references and a learned completion signal.
4. **Seal/admit / runtime:** on completion, validate/freeze the packet and reserve
   its bounded completion cost. A valid completed packet is dispatched; there
   is no second semantic decision model or confirmation action.
5. **Evaluate / W+D:** append the packet to the same causal history and produce
   semantic status and the operation's value if valid.
6. **Return / runtime:** validate the pending binding and resolve selected memory
   contents. Never substitute a reference annotation for a model result.
7. **Consume / W+R:** append the bound return, interpret it and resume native
   execution. Repeat as selected by Qwen, then finish the normal answer.

The gate chooses operation intent; learned writer completion controls when a
packet is ready; runtime admission checks structural/resource conditions. The
three are distinct. A walkthrough's completed-request invocation point refers
to dispatch, not a second learned gate. No optional-operation gate runs while
D is evaluating or during return delivery in the first version.

The [writer schedule](latent-writer-schedule.md) uses a native WRITER_START
position, parallel question/target heads, then bounded WRITER_TICK steps that
emit evidence/candidate/rubric records or revisions. A learned action head
chooses FINISH/ABORT; draft summaries and direct bounded-memory access preserve
construction context. Baseline maximum preparation is 10 Noul, 26 Choice or 11
Score positions before revisions, within the 32-position ceiling. These are
structural event counts, not GPU feasibility measurements.

Attempt count increments when an optional-operation intent starts an attempt,
before its writer forward; abort/non-admission still consumes that attempt.
D-call count increments only on actual dispatch. Writer-position count includes
every executed preparation tick. Structurally invalid FINISH leaves a draft
unsealed and may be revised within the same bounds. Resource non-admission
ends that attempt; a retry is a new attempt/identity, never an edited snapshot.

One request lifecycle is draft -> sealed -> evaluating -> validated return ->
consumed, with explicit not-admitted, cancelled and failed exits. Preparation
and rejected attempts are charged. An interrupted state is never assumed to
support rollback; resume only if it is demonstrably usable.

The recorded [control-flow v1](control-flow-v1.md) specializes this lifecycle.
WRITER_START initializes arguments without a seven-action event; each next
WRITER_TICK executes one action, including real FINISH/ABORT positions. Return
delivery consumes the packet through RETURN_END and resumes the same phase;
extra consumption ticks=0. Roles 37/38/40 are not emitted. Thinking closes with
IDs [248069,271] (two H positions), chat ends with 248046 (one H position), and
input-only/protocol/private IDs are masked from native sampling. B=0 forces
closure; A=0 forces a Limited end. Cancellation adds no closing/notice tokens.
At most one RUNTIME_LIMIT notice per closed attempt; code12=WriterAborted.
Healthy H spend<=3 startup+2 closure+16 notices+1 chat end=22, within H=64.
The profile defines priorities, exact masks, EOS override, terminal outcomes and
training labels; it is not passing runtime evidence.

## 4. Request and memory contract

| Field | Content / owner |
|---|---|
| Request identity, frontier and bundle version | Runtime-assigned/pinned bookkeeping |
| Operation | Learned native policy |
| Question | Eight learned 1024-wide vectors expressing proposition/criterion |
| Target and evidence | Learned bindings to immutable objects/source spans |
| Choice bank | Ordered immutable entries; four semantic vectors per entry plus exact recoverable attributes |
| Score rubric | Five two-vector fields: dimension, conditions, zero/one anchors and optional midpoint; exact range/direction/presence/version |
| Completion | Learned writer output; runtime checks structural completeness |

Questions specify what to assess, not the expected verdict. Rubrics describe
a scale independently of desired scores. Candidate attributes preserve the
supplied evidence, not gold feasibility labels. Exact names, times, quantities
and identities must remain recoverable, not solely approximate vectors.

References are run-scoped, typed, versioned and never silently rebound. A learned
pointer chooses a catalogue entry; runtime resolves it to the exact reference.
Source spans refer to immutable original source-token offsets, not growing
cache positions. The store is append-only; edits create new versions. Snapshots
pin owned immutable values, bank order, rubric meaning and model/protocol bundle.
Immutability preserves gradient connections during training.

Choice pointer indices resolve immediately through the sealed bank to stable
entry identities. Deliver selected contents, not only an integer or opaque ID.
A score remains attached to its target and rubric version. Duplicate/late/foreign
returns cannot inject feedback twice or resume a cancelled job.

Owners: [latent layout](latent-representation-v1.md),
[packet codec](latent-packet-codec.md) and
[memory/snapshots](memory-and-request-snapshots.md). The codec records ordered
request/result framing, role inputs, source-state presentation, typed exact
values and continuous result encoding. The target uses two semantic slots and
non-assessed reasons use four as new starting defaults. Native returns echo the
original question and Score rubric; all positions are charged. The
[typed-value ABI](typed-value-abi.md) resolves A1 as a design: eight-byte atoms,
explicit type costs, eight attributes per record, 32-token source spans and
128-byte UTF-8 values. The [writer](latent-writer-schedule.md) produces bounded
parallel typed heads; actual implementation/interpretation remains unverified.

Return envelopes reserve 50 for Noul, 61 for Score and max(50,18+largest_attribute_
cost) for Choice. Every Choice entry has at most 46 attribute positions; extra
reason contents total at most 32 positions/eight attachments.
Admission must cover every possible selected-entry return and bounded reason,
not only the winner one currently expects.

## 5. Results, status and native interpretation

| Operation | Assessed payload | Native meaning |
|---|---|---|
| Noul | Probability that the bound proposition is true | Interpret polarity/criteria; true is not universally approve |
| Choice | Selected stable entry and recoverable contents | Use that entry's meaning/attributes, independent of array position |
| Score | One continuous value bound to target and rubric | Interpret dimension, conditions, anchors and direction |

Learned semantic statuses are Assessed, Insufficient evidence, Conflicting
evidence, Unresolved and Choice-only No feasible choice. Only Assessed carries
a decision value. Zero is a valid assessed value; absence is not tested through
numeric truthiness. Missing evidence is not false, zero, 0.5 or an arbitrary
winner. A maximum pointer score does not prove feasibility. Ties can admit
several valid selections; unknown availability is not known unavailability.

Structural errors, nonfinite/malformed payloads, resource exhaustion, execution
failures and cancellation are runtime outcomes, separate from semantic statuses.
The runtime validates structure/range/binding, not domain correctness. Status
heads and evidence sufficiency are learned and fallible.

Non-assessed returns include bounded latent reasons with appropriate bindings.
A probability, scalar or selected ID alone does not reveal supporting facts.
Native explanations must use supplied evidence or explicitly returned reasons;
isolated readers must not invent withheld facts. Preserve conflict recognition
rather than unconditional obedience or universal ignoring of results.

Owner: [typed statuses](result-statuses.md). The three
[training walkthroughs](complete-training-examples.md) explain result use.

## 6. Context and position accounting

C=262144. For finite raw startup readout, s=clamp(raw,0,1):

    B_pred = floor((3*s*s - 2*s*s*s) * 131072 + 0.5)
    B_granted = min(B_pred, C - P - A - H)

P includes all prompt/template positions. A reserves configured visible-answer
text; H initially reserves 64 restricted startup/phase/termination/failure-control
positions. Reject configuration/input if P+A+H>C. Show predicted, overridden
if applicable, and actually granted allowances separately.

B pays for reasoning text AND every subsequent latent request/receiver/return/
consumption position, including calls during answering. Starting the answer
does not reset B. Zero B still permits a normal answer but no additional Qev
calls; the fixed initial budget prefill still occurs. General Score never
implicitly updates B. Manual overrides must be explicit, bounded and excluded
from autonomous budget-prediction evidence.

One soft vector input is one position, not 1024. Storage and lookups alone do
not advance context; presenting contents does. Re-presented rubrics and separate
native/receiver processing passes are charged each time. Heads with no position
advance still have attempt/record bounds; a position budget is not a FLOP limit.

Starting engineering ceilings: 32 native preparation + 256 receiver packet +
64 return/content delivery + 0 extra consumption = at most 352
positions per attempt. At most 16 alternatives, 8 evidence refs, 32 new records
per attempt and 16 attempts/job; packet ceilings still apply. These are versioned
starting limits, not proven semantic capacities.

The 32-new-record limit counts newly allocated immutable Ref objects/versions
within an attempt, including bank/result/reason objects when created. Inline
attributes and vector slots belong to their parent record; they do not each
allocate an extra Ref. Revised versions still count. Packet lengths and head-
expansion limits bound inline data separately; summaries are not a loophole for
unbounded computation. A position receives one primary debit, not duplicate B
debits because it is both reasoning and associated with a call.

Reserve actual packet length plus bounded return/consumption before dispatch;
debit actual use and release unused reservation. Failed preparation still costs.
The full resource contract additionally holds Result+Reason (2 Refs) and sealing
objects from intent entry: snapshot (1), plus final bank (1) for Choice. Writer
allocation limits before sealing are 29 for Noul/Score and 28 for Choice; bank
draft edits allocate no immutable bank versions. After sealing allocated+2<=32.
Freeze compatible published return catalogue K: attachments reuse its objects or
contain inline values; no new Ref-bearing attachments beyond Reason. With q the
exact request length and m existing K identities outside its handle table, hold
q+return_max positions and 2+min(32,m) handles, protecting A/H and enforcing
total handles<=256 independently. A hold does not mint an object or assign a
handle. The resource owner fixes atomic admission, lifetimes and exact examples.
Never start a call that cannot complete. No hidden use of A or H for semantic
work; no silent dropping of fields/alternatives to fit. Exhausted H cannot fund
an unbounded failure loop. B exhaustion closes thinking on the same history
and continues within A; exhausted A stops/marks incomplete. Cancel discards late
returns. Unsafe execution failure terminates rather than pretending rollback.

Owner: [budget accounting](position-limits-and-budget-accounting.md).
The [packet codec](latent-packet-codec.md) fixes structural framing costs: 13
common request/result positions, 11 for a rubric and 5 per candidate before
attributes. The implicit startup budget return uses three H positions.

## 7. Hybrid state and end-to-end gradients

Maintain one causal history of attention keys/values, convolution buffers,
recurrent memory and positions. Switching modes changes future computation;
it never rewrites earlier state. This is a purpose-trained alternating model,
not automatically equivalent to stock Qev or an entirely native history.

Serving may mutate its working cache. Training uses differentiable functional
updates preserving earlier values needed by backward. Saved activations and
pure GPU checkpoint recomputation are training memory, not a second serving
prefill. Recompute with explicit captured modes and no repeated registry,
accounting or consumption side effects. Do not deep-copy the entire prefix at
every step, detach at call boundaries or blanket no-grad through frozen W.

The task-return continuity rule excludes two intentional boundaries: frozen
reference targets and startup budget control-feedback values. Those are
detached as specified; Noul/Score task returns and the shared history are not
accidentally detached. Component/reference exercises have separate labelled
contexts; they are never hidden extra prefills in the deployed job.

Continuous Noul/Score returns can connect continuation losses through the native
reader, decision readout and request writer. Hard action, status, pointer and
reference choices need explicit supervision; no ordinary derivative through
argmax or integer lookup is claimed. Native R must receive effective updates
for interpretation. Step optimizers only after complete episode backward.

Installed fused recurrent backward/state mutation issues make G1 a real
feasibility gate. Begin with short complete GPU episodes before full data/training.
Owner: [cache/gradient strategy](cache-and-gradient-strategy.md).

The [numeric verification plan](numeric-verification.md) owns proposed forward/
gradient tolerances and exact control checks. Tensor closeness alone cannot
excuse changed budgets, bindings or certified greedy traces. Compare functional,
checkpointed and serving paths, verify return-path gradients, frozen/row update
isolation and export. Begin with short GPU fixtures; no numeric check has passed
merely because its tolerance is documented.

Starting checks include FP32 forward atol/rtol 1e-6/1e-5, BF16 hidden/state
2e-2/2e-2 plus normalized RMS <=1e-2, task probability/Score absolute difference
<=0.002, and startup score <=1e-5 plus exact integer allocation. Scalar-row
round trips use 1e-6/1e-5 for identical hidden inputs. Exact control/binding and
frozen-content checks are separate. Gradient limits, all logit/state checks and
FP64 methodology stay in the numeric owner. The recorded
[A3 profile v1](numeric-profile-v1.md) fixes near-zero denominator floors,
ordered FP64 budget arithmetic, explicit common GPU kernels/flags and fixtures.
These settings have not been implemented or validated.

### 7.1 Consolidated A3 numeric contract

| Item | Recorded initial setting |
|---|---|
| Forward RMS error | RMS(diff)/max(RMS(reference),1e-6), in GPU FP64; <=1e-5 FP32 and <=1e-2 BF16/logits, alongside pointwise limits |
| Near-zero absolute forward bounds | 1e-11 FP32; 1e-8 BF16/logits |
| Gradient RMS error | RMS(diff)/max(RMS(reference),1e-10); <=1e-4 FP32 and <=0.05 BF16/FP32; cosine >=0.99999/0.995 only when both RMS>1e-10 |
| Near-zero absolute gradient bounds | 1e-14 FP32; 5e-12 BF16; a separate nonzero path fixture is still required |
| Budget | Finite raw FP32 -> exact promotion to FP64 -> ordered separate multiplications/subtraction in A3 -> floor(scaled+0.5); checked nonnegative int64 ledger |
| Kernels | CUDA SDPA MATH only; same functional token-ordered recurrence/FP32 recurrent accumulation and differentiable causal convolution in training/serving |
| Backend flags | IEEE FP32; reduced-precision GEMM accumulation/split-K disabled; math-SDPA reduced precision disabled; deterministic algorithms strict; cuDNN benchmark off/deterministic on; CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA init |
| Other execution settings | Zero dropout; no automatic fused/hub kernels, compilation or CUDA graphs; explicit RTX 3090; no CPU fallback |
| Fixtures | 17/29/43 seeds; budget endpoints/all 131072 rounding transitions; typed packet/ledger boundaries; isolated gradients, export and free/prescribed trace parity |

The A3 owner contains the exact flags, arithmetic order and oracle requirements.
Begin with complete 128/256/512-position fixtures. A4 additionally requires A3
qualification at 1024 positions before joint training. BF16/FP32 audits use
tolerances; exact discrete parity is between the paired deployed-precision
paths. Numerical closeness never excuses changed allocations/decisions/tokens.

## 8. Data, training and functional readiness

Start with about 120 distinct fictional scenarios across six domains and review
them before scaling. Construct facts/rules before natural wording; teachers may
propose phrasing/annotations but are not truth oracles. All model generation/
encoding runs on the selected GPU. English annotations stay out of inference.
Split scenario families before generating variants; related contrasts stay
within a split. Preserve ordinary direct-answer and original decision replay.

The [data-operations procedure](data-operations.md) fixes a 120-scenario pilot,
20 per domain. Authored templates need no model generator; model-proposed roots
may count under the same reviews/approval with their origin preserved, per that
procedure. Keep human input, semantics, gold outcomes, guidance and provenance
separate. Use semantic review with proposed outcomes hidden, then independent
outcome derivation in a separate model session. All records, including pilot,
locked, subjective and rubric cases, need one model-assisted content review
plus mechanical checks. No human reviewers or second-review quotas are required
for this experiment. Release approval may be automated; review and approval
artifacts must bind actual bytes. Critical errors quarantine affected batches/
lineage. No content-review execution is implied by this policy.

Preserve original source hashes, byte spans and verified token/prefill mapping;
edits create new versions. Record origin, parents, generator/seed, label method,
review/adjudication, model/checkpoint/GPU/forcing metadata when applicable, split
and release hashes. Pilot/worked cases are design-exposed; fresh locked families
remain separate. Annotation disagreement is not a semantic Unresolved label.

Initial scale is 6000 core examples + 2000 replay, 1000 development and 2000
locked test cases (500 component, 500 interventions, 1000 free prompts). These
are starting quantities. Separate model review checks locked semantic labels
before testing; unresolved proposals are excluded. Subjective scores/intervals and effort preferences must
be reviewed; example .04/.05/.06 budgets and .30 severity are not approved gold.

Training progresses through operation/receiver semantics, native latent writing
and isolated reading, complete actual call/return trajectories, then unforced
invocation. Early teacher-forced actions/returns teach skills and are marked as
such. Final-answer loss is combined with semantic/binding/status/value losses;
never report bridge-only fitting as native result learning.

The [loss configuration](loss-configuration-and-training-schedule.md) owns
normalized starting coefficients, reference-validity masks, acceptable-set
Choice loss, raw-value regression and forcing stages. Guide a contiguous
decision prefix shrinking linearly to zero; joint episodes use predicted slots
and actual Qev results. Keep isolated gold-reader exercises distinct. The
startup budget's feedback value is deliberately detached, while continuous
task results remain connected. Replay begins at 25%; all defaults need pilot/
development verification. No token-saving/answer-improvement reward is assumed.

Starting normalized weights are 1.0 for native continuation/replay, gate, writer
actions, pointers, exact attributes, semantic alignment, status, Noul, Choice
and task Score; 0.5 for counts/types/presence, reconstruction, feasibility and
effort; contrastive weight/temperature 0.1/0.1. Use reference-validity masks,
not predicted abstention. Do not apply original packet labels after incompatible
rollout changes. Reference-codec reconstruction starts at weight 1.0. Full
definitions, eligible-count normalization and intentional gradient routing
remain in the loss owner. The recorded
[A4 training run profile](training-run-profile-v1.md) fixes phase lengths,
optimizer groups/schedule, exact-value subtypes, corrections and promotion.

The recorded [exact module/optimizer contract](module-and-optimizer-contract-v1.md)
resolves F6's design details: safe non-affine normalization, explicit cross-attention
and emission equations, shared native/evaluator trunks, complete typed heads and
pointers, initialization and unique optimizer ownership. Accumulate separate
component gradients on CUDA, divide each by its window's eligible EXAMPLE count,
combine once and then clip/step. Unused parameters retain grad=None. No count-first
rollout replay or multi-episode graph retention is required; actual buffer/state
memory and gradient agreement remain unmeasured G1 checks.

### 8.1 Consolidated A4 run settings

| Phase | Updates | Guidance |
|---|---:|---|
| Reference codec | 1000 | Approved semantic records; train reference pooler/codec and T; then freeze/version |
| Components | 2000 | Reviewed writer actions, reference receiver packets, isolated gold-return reader exercises |
| Guided joint | 1000 | Compatible reviewed decision prefix; predicted slots and actual Qev returns |
| Transition | 2000 | Prefix fraction q=1-k/1999 for k=0..1999; no gold reinsertion after divergence |
| Free-control joint | 1000 | No forced decisions, substituted slots or gold returns |

Use eight scheduled examples per update, microbatch one, complete-episode
backward and whole-window eligible-count normalization: 7000 scheduled windows,
56000 presentations including repetitions/reference records. Components sample
2 writer/2 receiver/2 reader/2 replay; joint samples 4 interleaved/2 direct/2 replay.
Replay is 1000 language/1000 reviewed decision examples, one of each per window.
Reference preparation is separate from this native/decision replay mixture.
Training seed 17; save RNG/sampler state. Component decision sampling is 50%
Assessed and 50% balanced supported non-assessed statuses across cycles, with
operation/domain coverage; never impose these proportions on production routing.

Reference annotation/reconstruction ceilings are 256 tokens/512 positions;
component/joint episodes are <=512/1024 total positions. Early component/guided
examples may use explicit B=512/A=128; remove B overrides during transition/free
control. These are training limits, not a new context maximum or a hidden budget
cap. Overlength attempts fail without truncated-episode backward or silent replacement.

AdamW beta=(0.9,0.95), epsilon=1e-8, global gradient norm limit 1.0; FP32 CUDA
masters/moments; no gradient scaler; foreach/fused false, capturable true without
enabling CUDA graphs. Peak LRs: R=1e-4, D=2e-5, new projections=2e-4,
T=1e-4, scalar rows=1e-4. Only new projection matrices have decay 0.01;
adapters/rows, biases, query/prototype vectors and embeddings have decay zero.
Warm up for ceil(5% of phase updates), then cosine-decay to 10% peak; the A4
owner defines the exact multiplier. Preserve existing moments across phases.
Check gradients before clipping; inactive parameters/rows retain grad=None.

Exact-value losses are byte CE for int64, coefficient-byte/exponent CE for
decimals, length/active-byte CE for UTF-8 and BCE for bool. Copied/parsed values
receive their applicable mode/pointer losses only. Do not double-charge reference
attributes, regress approximate ints or train continuous returns as byte strings.
Keep raw Score/effort regression, valid-set gate/Choice losses, semantic
normalization and reference-validity masking; native text teacher forcing is
distinct from decision forcing and is absent from free evaluation.

Reserve 400 existing core examples for correction (100 writer/reader, 300
trajectories), 80 per question/rubric, bank/attributes, binding/lifecycle,
non-assessed/conflicting-result and malformed/resource family. Corrections match
the actual snapshot/return. Unknown latent meaning masks unsupported downstream
labels, not the failure report. Sealed repairs require a new attempt; cancellation
has no invented answer or semantic value.

### 8.2 Development promotion and locked evaluation

The recorded [promotion matrix v1](promotion-matrix-v1.md) is the detailed F8
owner: fixed1000/2000 slice allocations, 13 status cells, family minima, paired
interventions, separate all-attempt metrics, successful unforced coverage and
effort gates. It specializes the summaries below; actual approved fixtures,
baselines, independent re-review, implementation and GPU evidence remain absent.
Development W/E splits70/130; locked component500 splits RF100/W100/E130/R130/M40.
Never pool guided/clean/transfer/free protocols. Score point tolerance is .05,
reviewed interval width<=.20; unscorable calls fail aggregate status success.

Partition 1000 development cases into 100 reference recovery, 200 writer/receiver,
100 isolated reader, 200 intervention cases (100 pairs), and 400 free jobs.
Reference promotion requires >=95% semantic recovery and >=90% applicable
contrast-group success. Component promotion requires >=95% fully correct writer
structure/typed fields; retain >=90% overall semantic success, >=85% per operation,
>=95% status/reader success and >=90% reviewed Score agreement. Guided/transition
promotion needs >=90% complete flow under declared guidance and >=90% applicable
intervention success, never autonomous evidence. Free promotion requires the
unforced thresholds and coverage below. Applicable A3 and exact runtime invariants
apply at every stage; fabricated values remain <=1% on non-assessable cases.

Native replay NLL may rise <=5% from its fixed matched pre-training baseline;
decision replay reaches >=90% after components, then remains within two percentage
points of that fixed promoted checkpoint. Checkpoint/evaluate every 100 updates;
finish the block and pass at both U-100 and U before advancement. Do not select an
earlier favorable checkpoint, silently extend a phase or relax the gate. Failures
require versioned development revision. Final locked tests remain separate and
untouched until development gates pass; training-length limits do not prove G6.

The [semantic-supervision method](semantic-supervision.md) builds role-whitelisted
reference slots from reviewed per-record annotations using the frozen base and
learned pooling. Train/test reconstruction and meaning contrasts, then freeze
the reference encoder/decoder. Train the writer with slot alignment,
reconstruction and contrastive objectives; ground Qev first on reference packets,
then actual writer outputs. Keep invariance and isolated-packet tests alongside
whole-flow tests. Reference component contexts are offline training exercises,
not extra prefills inside a served run. This reduces/tests shortcut behavior;
it does not prove zero information about an answer in every writer output.

The recorded [training-input rules](training-input-rules-v1.md) resolve F7:
request targets exclude annotator-created current verdicts/winners/scores;
grounded facts remain available; post-decision Reason may describe justified
non-assessed status/support. Branch-specific builders, actual-carrier Reason
masks and logged outcome-bearing teacher-forced prefixes preserve this boundary.
Guided reader/Reason exercises do not replace actual joint returns or establish
autonomous outcome derivation. Implementation/re-review/validation remain absent.

The recorded [clean-test contract](clean-test-construction-v1.md) resolves F3's
construction design: rebuild all slots/states/values/lookup records from the
permitted view, freshly encode excerpt-only sources, reset component state and
verify hidden-fact twins. Removing visible text does not clean contextual tensors.
Clean components, contextual production-tensor transfers and one-prefill whole
flows remain separate protocols. This is not passing isolation/GPU evidence.

Use one canonical semantic normalization for alignment, reconstruction and
receiver presentation; do not normalize only the loss while exposing unchecked
raw slot magnitude to the receiver. This applies to learned semantic slots,
not to the scalar rows' direct dot products. Exact normalizer/pooling/projection
dimensions and reconstruction-adapter settings are recorded in
[model profile A2](model-configuration.md).

Readiness targets remain proposed: zero observed execution/binding violations;
verified gradient/forward checks; at least 90% acceptable semantic flow overall
and no operation below 85%; 95% status accuracy with at most 1% fabricated values
on non-assessable cases; 90% reviewed Score-interval agreement; 95% isolated-reader
success. Free coverage requires 50 valid unforced calls per operation across
multiple domains and 100 successful direct-answer jobs. Coverage does not force
routes per prompt or permit selecting only successful calls for scoring.

Scoring units/denominators and annotation/error/interval rules are recorded in
their owners. Freeze valid-route evaluation and concrete numeric/run profiles
before locked tests. Report failure counts and uncertainty.
If test failures drive tuning, replace the test set for a fresh claim. Original
natural prompts remain excluded from training/tuning but, having been discussed,
are illustrative rather than fresh independent tests.

Owners: [data/readiness](training-data-and-readiness.md),
[generation](training-data-generation.md) and [data operations](data-operations.md).
This is verification that the flow
works, not a study of answer improvement, token savings or optimal allocation.
Short trajectories do not establish full-context or maximum-budget reliability.

## 9. Implementation sequence and evidence

### Recorded profiles and implementation readiness

| Item | Status |
|---|---|
| A1: typed values | Recorded ABI/body lengths/bounded return envelope; unimplemented/unvalidated |
| A2: model configuration | Recorded dimensions/projections/adapters/precision; uninstantiated/unvalidated |
| A3: numeric profile | Recorded near-zero arithmetic/kernels/fixtures; uncalibrated/unvalidated |
| A4: training run profile | Recorded phase/optimizer/loss/correction/promotion defaults; no instantiated run or approved dataset |
| B1: exact neural input encoding | Recorded IDs/layouts/helpers/counters/Noul initialization; section 11 adds every role's field/tag/owner/length/absence rules, target grammar and deterministic handle traversal; no implementation or passing vectors |

A1-A4 and B1 have recorded initial designs, not passing empirical gates.
[B1](neural-input-encoding-v1.md) owns the exact shared input map: roles 1-40,
reserved 41-63, explicit None/presence coding, big-endian metadata/atoms,
role-selected header/descriptor/control layouts and fixed tag order. Numeric
helpers use sign, bounded coefficient magnitude and exponent/18; counter
denominators are 131072/262144/64/262144/32/16/32 plus the phase bit. Noul starts
from independent FP32 copies of false=3721/true=1802 vocabulary rows, with
dot-product logits divided by 16. The owner fixes all formulas and field offsets.
Its section 11 is normative for role-specific application: request target attrs
follow the two target slots; returns echo zero target attrs. Missing midpoint
keeps two zero canonical payloads with syntax tags and masked semantic losses.
Headers use following length=N-1; boundaries use zero. Historical presentation
preserves original contents/status/counts/lengths while translating handles and
setting historical flags. Sealed rendering assigns handles in canonical field
order, independently of writer catalogue indices. BUDGET_VALUE is detached
clamped predicted s; effective grant/overrides are delivered in counters.
Implement matching reference fixtures and validate before G2 interoperability;
do not let training and serving invent independent maps. Actual reviewer assignment, pilot approval and
GPU feasibility are operational/empirical work, not choices settled by prose.
The [audit](final-decision-audit.md) is the detailed closure/limitations record.

Recorded responses now cover F1 resource admission, F2/F4 input/target mapping,
F5 control flow, F6 modules/optimizer/accumulation and F7 training-input whitelists,
subject to implementation, independent re-review and validation. F3 clean-isolation
construction and F8 promotion matrix are also recorded. Their actual approved
artifacts, independent re-review and empirical validation remain. Do not treat
the initial profile list as an independently validated complete contract.

R is the native adapter; T in the model profile is the training-
only reconstruction adapter. It adds no deployed mode: the runtime remains
W+R/W+D. Keep D initialized from the pinned Qev architecture.

The [current handoff](next-agent-handoff.md) records measured mechanisms,
qualified arithmetic profiles and remaining prerequisites. The
[expansion plan](post-proof-expansion-plan.md) defines dependency order. A
bounded engineering pass does not certify learned invocation, full context or
readiness to deploy.

Follow G0 contract freeze -> G1 short GPU feasibility -> G2 runtime/memory ->
G3 component skills -> G4 complete trajectories -> G5 free generation/UI ->
G6 length expansion/release preparation. See the
[handoff](implementation-handoff.md) for each package's exit evidence, actual
prototype migration gaps and remaining choices. No existing checkpoint is
silently upgraded to the new protocol; preserve artifacts and validate manifests.

The UI performs one Send and streams the normal answer. Budget/call events are
observations of that same run, never a required decision-first confirmation.
Keep cancellation, run ownership and existing access validation. A deployment
change requires reviewable code and appropriate later authorization; this
consolidation changes no running service or checkpoint pointer.

## 10. Examples and research context

Use the [example index](complete-training-examples.md) and full
[Noul](noul-training-walkthrough.md), [Choice](choice-training-walkthrough.md)
and [Score](score-training-walkthrough.md) walkthroughs for inspectable teaching
objectives. [Natural prompts](natural-invocation-prompts.md) and original
[refund](refund-noul-flow.md), [meeting](meeting-choice-flow.md) and
[demo](demo-score-flow.md) flows preserve the semantic discussion.

[Coconut](https://arxiv.org/abs/2412.06769) motivates continuous internal steps;
[Toolformer](https://arxiv.org/abs/2302.04761) motivates learned call formation
and result use. [Prompt tuning](https://arxiv.org/abs/2104.08691) supplies precedent
for learned continuous inputs. None validates this exact shared-state interface.
[PEFT cache guidance](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora#using-and-reusing-cache-for-generation)
does not establish arbitrary mixed-adapter history compatibility. Runtime/GPU
feasibility and semantic result use still need evidence.
