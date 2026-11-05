# Position limits and budget accounting

Status: recorded design choice, not implemented or validated capability. No
code changes, model workloads or deployment are authorized by this document.

Owner: context/position accounting and resource admission for positions, local
handles and new immutable records. The post-Astra admission section records F1's
design response; implementation, independent re-review and GPU validation remain
unperformed. Use the
[implementation specification](architecture-and-training-plan.md) for lifecycle
and [handoff G0](implementation-handoff.md) to freeze codec lengths before
claiming that the packet ceilings fit real requests.

The [packet codec](latent-packet-codec.md) records structural framing, including
13 common request/result positions, 11 per rubric, 5 per candidate before
attributes, and a three-position fixed startup budget return charged to H.
The [typed ABI](typed-value-abi.md) now fixes exact bounded variable bodies;
their implementation and all-status admission still require validation.

## One reasoning allowance throughout the job

Use one reasoning allowance B for reasoning text and all additional latent
decision work, including calls made during answering. Starting the answer does
not replenish B. Keep visible answer allowance A separate.

This revises the earlier architecture draft: **B = 0 permits a normal answer,
but no additional Qev calls**, including during answering. The fixed initial
budget readout still happens at startup. There is no separate hidden-work
reserve for answer-phase calls. This is a design revision, not a change to the
live prototype.

| Counter | Meaning |
|---|---|
| C | Total context capacity: 262144 positions |
| B | Reasoning text plus additional latent work, across both phases |
| A | Configured visible answer text allowance |
| H | Restricted control reserve, initially 64 positions |

H covers bounded startup delivery, phase closure, termination and resource-failure
signaling. It cannot fund semantic reasoning or further decision calls. All
positions, including controls, count against C. The initial prompt counts as P;
its fixed budget readout uses the same prefill, with subsequent startup delivery
explicitly charged to H.

P includes the full chat template and supplied prompt. Manual overrides remain
explicit, bounded by the same context admission, and are identified separately
from predicted/granted values; they are not autonomous budget-prediction evidence.

For continuous score s clamped to [0,1]:

    B_pred = floor((3*s*s - 2*s*s*s) * 131072 + 0.5)
    B_granted = min(B_pred, C - P - A - H)

The [A3 numeric profile](numeric-profile-v1.md) fixes raw FP32 validation,
ordered FP64 evaluation/half-up rounding and checked int64 ledger arithmetic.
The formula above is mathematical shorthand, not permission to reassociate it.

Reject input/configuration if P + A + H exceeds C. Show predicted and granted
allowances separately if context capacity reduces the prediction. Half the total
context is a maximum, not a guarantee that this much fits beside the input and
reserved answer. A general Score return does not implicitly change B.

Per B1 section 11, startup BUDGET_VALUE delivers detached clamped predicted
score s. The structured counters carry the actual grant/override; an override
does not rewrite this prediction. Raw effort regression still uses raw readout.

Qwen may end reasoning early to leave allowance for calls during answering.
The learned policy and writer receive remaining counters as structured numeric
state, without another text prompt or prefill. Their delivery must itself have
explicit accounting if it adds model positions.

## What advances a position

One soft input vector is one model position, not 1024 positions because its
hidden width is 1024. Questions use eight positions per presentation;
alternatives use four per entry; rubrics use ten per presentation.

Charge native request formation steps, receiver packet processing, bindings
and framing, returned vectors and latent reasons, retrieved content delivered
to the reader, and further latent processing to B and C. If formation and
receiver evaluation process positions separately, charge both. Presenting a
previously used rubric again is not free.

Stored immutable records and reference lookups alone do not advance the model
context. Their contents do when presented as model inputs. Heads/projections
may perform computation without adding positions; bound attempts and record
creation separately. This is a position budget, not a FLOP or wall-time budget.

## Initial engineering ceilings

| Component of an additional request attempt | Maximum positions |
|---|---:|
| Additional native request preparation | 32 |
| Qev packet, including arguments, bindings and evaluation framing | 256 |
| Typed return and retrieved contents delivered to native Qwen | 64 |
| Extra consumption ticks in control-flow v1 | 0 |
| Total associated attempt | 352 |

A packet has at most sixteen alternatives, eight evidence references, one
question, and a target/rubric as applicable. The serialized packet must still
fit 256 positions. Allow at most sixteen attempts per job and thirty-two new
records per attempt. Rejected attempts count toward the attempt limit.

Count an attempt when an optional-operation intent starts it, before the writer
forward; count an actual Qev call only on dispatch. Structural FINISH errors may
be revised within the same draft bounds. Resource non-admission ends that attempt;
retry starts a new one. The new-record cap counts allocated immutable Ref objects/
versions, not every inline attribute/vector. Revisions still allocate/count.

Every model position receives one primary B/A/H debit plus its C debit, never
two B charges merely because it is reasoning associated with a request.

These are starting engineering ceilings, not proven semantic capacities. If a
request is too large, revise it explicitly or decline it. Do not silently drop
alternatives, truncate a question or omit rubric fields.

## Admission and reservation

The following is the recorded post-Astra F1 admission design. Three independent
reservations must pass before Qev executes: model positions, local handles and
new immutable records. Do not predict the future semantic status, winner or value
to reduce the envelope. It covers structurally legal completions, not malformed
model output or unmeasured physical GPU memory/kernel behavior.

### Immutable allocation manifest

| Item | New records charged to this attempt |
|---|---:|
| New question, target, candidate or rubric | 1 each |
| New version of such an object | 1 each |
| Reuse of an unchanged existing Ref | 0 |
| Evidence/source-span selection | 0 |
| Inline attributes, semantic slots and scalar values | 0 beyond their owner |
| Final Choice bank | 1 at sealing |
| Request snapshot | 1 at sealing |
| Result | 1 |
| Non-assessed Reason | 1 |

Source spans are inline selections from an existing immutable source, not new
Source Refs. Keep the draft bank as a mutable list of candidate Refs and publish
ONE immutable bank at sealing, not a version after every candidate addition.
Every Choice attempt creates this final bank container even if candidates are
reused. Object publication mints the Ref after validating the proposed record;
temporary decoder tensors/builders have no published identity. Once published,
deleting/abandoning an object does not refund its allocation count. Preserve
required gradients in record tensors; immutable publication is not detachment.

### Sealing/completion record holds

Reserve two completion-record slots from attempt entry: Result plus maximum one
Reason. Reserve the sealing objects separately:

| Operation | Sealing hold | Completion hold | Maximum writer allocations before sealing |
|---|---:|---:|---:|
| Noul | 1 snapshot | 2 | 29 |
| Choice | 1 bank + 1 snapshot | 2 | 28 |
| Score | 1 snapshot | 2 | 29 |

Writer invariant: allocated + sealing_hold + completion_hold <=32. New versions
still count even when no longer selected. Reuse costs zero. After sealing:

    allocated_before_dispatch + 2 <= 32

The two is the EXISTING completion hold, not another two-slot reservation.
A sealed request with 31 allocated objects cannot dispatch even though an
assessed result alone might fit: 31+2=33. B1's raw counter stays 32-allocated;
it reports unallocated slots, not unreserved writer capacity. Structural masks/
validation subtract holds before permitting new publication. An intent/draft
is not an admitted Qev call; it can end not-admitted without a semantic result.

### Bounded reason construction and frozen reference catalogue

Reasons may contain four semantic slots, inline constructed attributes, spans
from existing sources, existing references and presentations of historical
results. They cannot allocate additional Target/Source/Candidate/Rubric or other
Ref-bearing objects. All such attachments belong inline to their owner or reuse
existing objects; at most the Result and Reason are newly published after dispatch.

At sealing freeze return-reference catalogue K from compatible, published
objects available in this run, including sealed request objects and excluding
future objects/unpublished builders. Every attachment's provenance, copied-from
owner, unit and reference-valued payload must resolve through K. Historical
presentations retain exact original bindings/versions. New Result/Reason IDs
are mechanically attached, not selectable self-supporting evidence. Catalogue
pinning is bookkeeping, not presentation of all contents, another model prefill,
semantic relevance filtering or attention isolation. Selection remains learned.

### Position hold

Debit preparation as it occurs, even on rejected attempts. After sealing compute
exact q including target attributes and evidence; q<=256. Use A1's all-status /
all-candidate envelope, requiring every Choice A_i<=46:

    return_max(Noul) = 50
    return_max(Score) = 61
    return_max(Choice) = max(50, 18+max_i(A_i))
    position_hold = q + return_max

Require:

    B_free = B_left - other_B_holds
    B_free >= position_hold
    C_left - A_left - H_left - other_context_holds >= position_hold

C_left counts context not yet spent, not context already reduced by A/H or this
new hold. Position arithmetic is checked int64. B/C holds represent the same
future semantic positions in separate ledgers, not two sets of work. All other
call holds must be resolved before serial admission of a new call. Preparation
is already spent and must not be held/charged again. Maximum post-preparation
hold is 256+64=320; adding preparation<=32 gives 352 per attempt under
[control-flow v1](control-flow-v1.md). Minimal Noul q=13 requires 13+50=63
remaining B/context capacity; 63 fits, 62
fails even if Qev would happen to return an assessed 14-position result.

### Handle hold and bound

Let T_q be the request's rendered handle table, h_q=|T_q| and m=|K minus T_q|,
counted by exact Ref/version identity. Existing objects can need new local handles
without allocating new immutable records. Hold:

    handle_hold = 2 + min(32, m)
    h_q + handle_hold <= 256

Two covers Result/Reason. A reason attachment cannot introduce more distinct
existing-object handles than its position cost:

| Attachment | Maximum additional existing handles | Position cost |
|---|---:|---:|
| Reference-valued attribute | 4: provenance/copy owner/unit/value Ref | 4 |
| Other attribute | <=3 | >=3 |
| Source evidence | 1 | >=2 |
| Historical result presentation | Its explicitly presented references | At least that many positions |

For historical results the bound follows from fixed framing and bounded child
attachments, with no recursive Ref expansion. With aggregate reason attachments
R<=32, additional existing handles<=32; all legal completion handles<=34. If K
adds no objects beyond T_q, hold only two. Assessed candidate contents/refs were
already presented in the request, so assessed completion introduces only Result.

Holds do not preassign handle IDs, intern sidecar-only catalogue objects or mint
phantom Refs. Actual assignment follows B1 section 11's canonical traversal,
preserving the request table prefix. Only distinct newly presented identities
consume handle credits. Handle, position and Ref capacities never substitute
for one another.

### FINISH, seal and atomic dispatch admission

1. Validate required fields, types, spans, counts, lengths and bindings.
2. Publish final bank if Choice and request snapshot from their sealing hold.
3. Compute exact q, request handle table and frozen K.
4. Check q<=256, all-candidate/all-status return<=64, attachments<=8/R<=32,
   and every position/handle/record inequality.
5. Atomically commit the outstanding completion reservations and dispatch.

Invalid structure leaves the draft unsealed and revisable within its remaining
bounds. Resource non-admission after sealing ends the attempt. Published records
and spent preparation remain charged. Release unused holds, never actual work;
keep the snapshot immutable. Retry has a new attempt/identity. No failed
admission fabricates a semantic Result or Reason. Reserved sealing capacity
does not authorize an extra 33rd writer position or bypass the attempt cap.

### Consume credits and release unused holds

| Event | Accounting |
|---|---|
| Process a reserved semantic position | Debit B/context once and decrement corresponding holds |
| Publish a reserved object | Increment allocated record count and decrement its hold |
| Assign a new distinct local handle | Grow table/decrement handle hold |
| Reuse an existing local handle | No new handle or Ref charge |
| Finish with unused capacity | Release unused holds; retain actual charges |

Assessed completion uses one new record and releases unused Reason credit;
non-assessed uses both. After return delivery release unused return-position,
handle and record holds. Control-flow v1 has ZERO extra consumption positions
or holds; resume at RETURN_END in the same native phase. Other call holds must
be resolved before another call. Do not automatically debit the 352 ceiling.

During answering latent work still spends B, visible text A, and control signals
H, with each position's context debit. Result validation/publication must precede
delivery. Oversized reasons, invalid references or nonfinite values are execution
failures, not truncation or a different semantic status. Release unused credits
on cancellation/failure; spent positions/published objects remain charged. Resume
only if the causal state is usable; no rollback or CPU fallback is assumed.

### Simultaneous boundary example

Choice with six candidates, eight target Reference attributes and candidate
attribute counts 8/8/8/8/8/5 has 53 four-position attributes. With distinct existing
provenance/copy-owner/unit/value Refs on each, and no overlaps with header owners:

    q = 13 + 6*5 + 53*4 = 255
    h_q = 4 header objects + 6 candidates + 53*4 = 222
    return_max = max(50,18+32) = 50
    position_hold = 255+50 = 305
    handle_hold = 34, when K has >=32 additional existing objects
    maximum final handles = 222+34 = 256

Eight four-position reason Reference attachments can use all R=32 and introduce
32 additional existing handles. New question/target/six candidates/bank/snapshot/
result/reason total 2+6+1+1+1+1=12 newly published records. Candidate attrs max32
stay below46. Existing references must be valid CopyTyped/provenance/unit bindings
in K, not invented identities. Scalar checks of these counts passed; this is not
a generated neural packet or GPU test.

Startup Budget is separate: its three positions spend H and it has no optional
attempt Result/Reason records. GPU VRAM/kernel feasibility and trained semantic
correctness remain empirical gates; admission guarantees protocol capacity for
structurally legal completions, not those untested capabilities.

If admission fails, report a mechanical not-admitted/budget-exhausted outcome.
Do not manufacture a semantic status, decision value, or partial valid result.
Bounded failure signaling uses H; repeated rejection cannot create a free loop.
If H cannot support another required control transition, terminate with an
out-of-band limited/error status instead of borrowing A/B or adding free model
positions. UI telemetry alone is not a model position or a semantic inference.

For example, eight formation positions, thirty-two receiver positions, sixteen
return positions and four subsequent ordinary reasoning-text positions spend
sixty positions of B. Those last four are not consumption ticks in v1.
If followed by a 150-token answer, A spends 150 and context grows by 210
positions, excluding other controls. Answer text does not subsidize hidden work.

## Exhaustion and interruption

- B exhausted: close reasoning incrementally on the same state and continue
  the normal answer within A. No more additional Qev calls.
- Answer-phase call cannot fit: decline mechanically; native Qwen continues or
  qualifies its answer using information already available.
- A exhausted: stop and mark output as limited/incomplete.
- Execution failure: return no semantic value. Resume only if the execution
  state remains usable; otherwise terminate rather than claim rollback.
- Cancellation: stop work and discard late returns.

No second prefill, rollback or hidden allowance is assumed. Training and later
verification must include these limits and failure continuations. This design
does not claim the predicted allowance is optimal or improves answer quality.
