# Latent writer schedule

Implementation update: The bounded runtime writer and typed decoding heads now has a scoped implementation in
`qwen_one_two/codec/`. See the [implementation](full-codec-implementation.md)
and [measured qualification](codec-qualification-results-2026-11-05.md).
The original design status below is historical; training and frontend
expansion remain excluded, and expanded mixed-precision qualification
is not passed. The recorded normative rules/tolerances remain in force.


Status: recorded design choice, not implemented or GPU-validated. This document
authorizes no coding, model workload, service change or deployment.

Owner: native writer steps, parallel record emission, learned references and
completion. Follow the [specification](architecture-and-training-plan.md),
[packet codec](latent-packet-codec.md), [memory contract](memory-and-request-snapshots.md)
and [position budget](position-limits-and-budget-accounting.md). The
[semantic](semantic-supervision.md) and
[loss](loss-configuration-and-training-schedule.md) designs are recorded.
[Audit](final-decision-audit.md) records A1-A4 and the now-recorded
[B1 neural input encoding](neural-input-encoding-v1.md), not passing GPU evidence.

## Core choice

Use a bounded sequence of native backbone steps. Each step can emit an entire
typed record through parallel learned heads; do not allocate a separate native
backbone step for every emitted semantic vector. This preserves the 32-position
preparation ceiling while permitting sequential construction and revision.

## 1. Enter writer mode

The ordinary learned gate selects Noul, Choice or Score. Keep W+R active, pause
normal text generation and append one learned WRITER_START input carrying the
selected operation and remaining resource counters. It is one model position
on the existing cache, not generated instructional text.

B1 fixes the role IDs/control bits/counter normalization. WRITER_START uses its
encoded control metadata; WRITER_TICK adds encoded control metadata to lifted
draft feedback before the existing wire normalization. This adds no extra tick.

B1 section 11 fixes role-by-role owner/tag/field rules and distinguishes writer
catalogue indices from packet-local handles. The prospective request length is
zero until Q/T initialization, then 13+target attributes+evidence+rubric+candidates;
it is telemetry, not an early position debit. Handle IDs are assigned during
sealed rendering in canonical field order, not by parallel-head completion.

The [resource contract](position-limits-and-budget-accounting.md#admission-and-reservation)
holds sealing/completion Refs from intent entry: snapshot+Result+Reason (3) for
Noul/Score, bank+snapshot+Result+Reason (4) for Choice. Thus writer publication
capacity is 29/28 respectively. B1's 32-allocated counter remains raw unallocated
capacity; masks/validation also subtract holds. Existing unchanged reuse costs
zero. Publish one final Choice bank at sealing, not after every list update.

Its native hidden state feeds the writer heads. Native R and the heads learn
how to begin forming the selected request from the current task/context.

Per [control-flow v1](control-flow-v1.md), WRITER_START only initializes Q/T and
their fields; no seven-action event runs there. Each subsequent WRITER_TICK
selects/executes one action from its resulting hidden state. Recoverable errors
go in the next charged tick's metadata; a closed failed/aborted attempt gets at
most one H notice. FINISH/ABORT each occupy a real writer position.

## 2. Initialize question and target

Parallel heads from the first writer hidden state produce eight question
vectors, two target vectors and target/source/existing-memory bindings.

    native hidden state
      -> question slots 0-7
      -> target slots 0-1
      -> target/reference heads

A shared slot decoder conditions on native hidden state, field role and slot
index. It emits different payload vectors without eight question-specific
backbone forwards. Store the results in the draft; they are not yet Qev input
positions. Learn structured question/target meaning and bindings rather than
assuming an unstructured hidden state already has those semantics.

## 3. Select subsequent writer actions

| Learned internal action | Effect |
|---|---|
| ADD_EVIDENCE | Select and bind an evidence record |
| ADD_CANDIDATE | Produce or reference one Choice alternative |
| SET_RUBRIC | Produce or reference a Score rubric |
| REVISE_QUESTION | Create a new version of the draft question |
| SET_TARGET | Correct/change the draft target |
| FINISH | Declare the packet ready for structural validation |
| ABORT | Abandon this attempt without a semantic result |

These are internal control outputs, not generated vocabulary tokens. The runtime
may mask structurally illegal actions such as a seventeenth candidate, but does
not select relevant facts, judge feasibility or choose the scoring dimension.
Native R and the action head learn construction order, sufficient supporting
material, revision and stopping. No operation re-gating or nested request occurs
inside a draft in the first version.

## 4. Emit a complete record within a step

An ADD_CANDIDATE event produces four semantic vectors, bindings, attribute
counts/types, attribute-key vectors and typed values/source-content pointers.
SET_RUBRIC can produce all ten rubric vectors plus metadata together, or select
an existing immutable rubric. Reusing an object does not alter its old version.

Use shared field/slot decoders rather than one large independent matrix for
every possible output position. Counts and lengths select active output slots.
All expansion remains bounded by packet/object limits. Source-backed values
use learned pointers; constructed numeric values use supervised typed-value
outputs. Discrete indices/values need their own supervision. Do not hide a
second unbounded autoregressive generator inside the record head.

Native R and the record/reference heads learn contents, exact-value fidelity,
source grounding and reuse. The runtime assembles selected fields, not their
semantic meanings. Oversized output is explicitly rejected/revised, never
silently shortened. Exact typed-atom generation/encoding and computational
bounds must be frozen with the encoder before claims of measured packet fit.

## 5. Select references through learned pointers

For an existing object, a pointer head selects from the typed memory catalogue.
For original evidence, learned heads select source identity and start/end span.
The runtime checks existence/type/bounds and resolves the selection to an exact
immutable reference. It does not choose which clause is the relevant fact.

For example, Qwen chooses the source clause describing Pine's throughput; the
runtime resolves the chosen source version/token span. Training pointer targets
come from annotations. Wrong but structurally valid selections are model errors,
not requests for a runtime semantic correction. Do not predict arbitrary UUIDs.

## 6. Feed draft progress into the next step

Before the next native step, a learned draft encoder summarizes current records,
their roles, previous action and remaining counters into one WRITER_TICK input.
That is one further position on the same causal history.

Writer heads can also attend directly to bounded draft tensors and source-memory
representations. The summary is not the sole surviving copy of the semantic
fields. This is a small learned memory interface in the same model bundle,
not another LLM, independent prompt prefill or second causal cache.

Native R learns sequential construction/revision with awareness of the current
draft. Keep record and summary gradient connections during training; immutable
storage must not detach each output. Head/decoder work is not free computation
merely because it does not advance the backbone position counter; bound and
record its expansion.

## 7. Finish, validate and dispatch

Qwen signals FINISH. Runtime checks required fields, reference validity/types,
counts, encoded lengths and bounded completion within remaining allowance.
A structurally valid completed request is sealed and dispatched under W+D.
There is no second semantic approval model or human confirmation action.

Checks cannot establish semantic question quality or evidence sufficiency. An
invalid completion stays unsealed; a subsequent bounded writer step may revise
or abort if resources remain. All used preparation positions and rejected
attempts remain charged. Do not force incomplete output to become valid.

Attempt count increments at optional-intent entry, before WRITER_START, whether
or not the attempt eventually dispatches. Structural failure can be revised
within that draft's step/object bounds; resource non-admission terminates the
attempt. A retry consumes a new attempt/identity. Sealed snapshots are never
edited. Count D calls only when dispatched, separately from attempts/ticks.

## 8. Preparation accounting

| Operation | Start | Maximum normal record steps | Finish | Total without revisions |
|---|---:|---|---:|---:|
| Noul | 1 | 8 evidence | 1 | 10 |
| Choice | 1 | 8 evidence + 16 candidates | 1 | 26 |
| Score | 1 | 8 evidence + 1 rubric | 1 | 11 |

Every writer step is one additional native backbone position charged to B/C.
The separate 32-new-object, 256-request-position and 64-return-position limits
still apply. These schedules are structural ceilings for the defined events,
not measured computation costs or guarantees of adequate semantic capacity.

At the 32-position preparation ceiling, further writing is disallowed. Finish
is valid only after structural checks; otherwise abort. Preserve a bounded exit
using the recorded control policy. A zero allowance permits no optional draft.

Producing eight question vectors through a head does not itself add eight native
backbone positions. Presenting them to Qev costs eight positions; any separate
native presentation/feedback costs its actual positions too. The native writer
step itself is already charged. Inline fields, object allocations, typed-value
head work and model-forward positions need distinct counters, not one misleading
record count.

The 32-new-record limit counts allocated immutable Ref objects/versions, including
bank/result/reason objects as applicable. Inline attributes/slots belong to their
parent, while revised versions count again. Inline expansion still obeys packet
and compute limits; object-count interpretation does not permit an unbounded
parallel value decoder. The [typed ABI](typed-value-abi.md) now resolves A1 as a
design with explicit atom costs, eight attributes and per-type heads; model
instantiation and GPU correctness remain unverified.

## 9. Example trajectories and supervision

For Noul workshop access:

1. Start: question and Sam target.
2. Add workshop-policy evidence.
3. Add completed-orientation evidence.
4. Add expired-badge evidence.
5. Finish.

For Choice, emit requirements/supporting evidence, Cedar/Birch/Pine records,
then finish. For Score, emit target/question, set the disruption rubric, add
relevant evidence, then finish. These are guided example routes, not forced
free-generation boundaries or required operations for those human prompts.

Train valid partial drafts, unnecessary additions, corrections, premature
completion and resource exhaustion as well as complete demonstrations. Native
steps provide sequential computation; parallel heads provide structured
payloads; learned pointers bind contents; learned completion triggers mechanical
admission. Whether the slot decoder/draft feedback communicate intended meaning
still requires semantic supervision and the GPU pilot.
