# B1: neural input encoding v1

Implementation update: The B1 runtime field/role mapping now has a scoped implementation in
`qwen_one_two/codec/`. See the [implementation](full-codec-implementation.md)
and [measured qualification](codec-qualification-results-2026-11-05.md).
The original design status below is historical; training and frontend
expansion remain excluded, and expanded mixed-precision qualification
is not passed. The recorded normative rules/tolerances remain in force.


Status: recorded design choice, not an implemented codec, model run or passing
GPU result. No coding, model execution, training, service change, commit or
deployment is authorized. All future neural encoding/model/optimizer workloads
use the explicitly selected RTX 3090; never fall back to CPU.

Owner: B1 role/class IDs, metadata bit layouts, atom helper formulas, counter
normalization, tag/handle conventions and Noul prototype initialization.
Section 11 records the post-Astra role-specific mapping and target-attribute
grammar. It specializes the earlier generic applicability wording; no field or
tag may be inferred independently by a training/serving implementation.
This specializes [A1 typed values](typed-value-abi.md),
[A2 model configuration](model-configuration.md) and the
[packet codec](latent-packet-codec.md), without changing their dimensions or
position costs. Follow [A3](numeric-profile-v1.md),
[A4](training-run-profile-v1.md) and the
[implementation specification](architecture-and-training-plan.md).

Roles encode syntax, not predetermined question meaning or verdicts. Qwen/Qev
must still learn the semantic content and bound-result interpretation through
the recorded training plan. These numbers are internal indices, not vocabulary
tokens or generated tool text.

## 1. Role IDs and head class order

Use the existing 64-row role table:

| ID | Role |
|---:|---|
| 0 | RESERVED_PADDING |
| 1 | REQUEST_HEADER |
| 2 | QUESTION_SLOT |
| 3 | TARGET_DESCRIPTOR |
| 4 | TARGET_SLOT |
| 5 | SOURCE_EVIDENCE_DESCRIPTOR |
| 6 | SOURCE_EVIDENCE_SLOT |
| 7 | PRIOR_RESULT_EVIDENCE_DESCRIPTOR |
| 8 | RUBRIC_DESCRIPTOR |
| 9 | RUBRIC_DIMENSION_SLOT |
| 10 | RUBRIC_CONDITIONS_SLOT |
| 11 | RUBRIC_ZERO_ANCHOR_SLOT |
| 12 | RUBRIC_ONE_ANCHOR_SLOT |
| 13 | RUBRIC_MIDPOINT_SLOT |
| 14 | CANDIDATE_DESCRIPTOR |
| 15 | CANDIDATE_SLOT |
| 16 | ATTRIBUTE_DESCRIPTOR |
| 17 | ATTRIBUTE_KEY_SLOT |
| 18 | BOOLEAN_ATOM |
| 19 | INT64_ATOM |
| 20 | DECIMAL_COEFFICIENT_ATOM |
| 21 | DECIMAL_EXPONENT_ATOM |
| 22 | REFERENCE_ATOM |
| 23 | ATTRIBUTE_SOURCE_SLOT |
| 24 | UTF8_ATOM |
| 25 | EVALUATE |
| 26 | RESULT_HEADER |
| 27 | NOUL_VALUE |
| 28 | SCORE_VALUE |
| 29 | REASON_DESCRIPTOR |
| 30 | REASON_SLOT |
| 31 | RETURN_END |
| 32 | BUDGET_HEADER |
| 33 | BUDGET_VALUE |
| 34 | BUDGET_END |
| 35 | WRITER_START |
| 36 | WRITER_TICK |
| 37 | CONSUME_TICK |
| 38 | PHASE_CLOSE |
| 39 | RUNTIME_LIMIT |
| 40 | TERMINATE |

IDs 41-63 are reserved/rejected. Padding is a zero vector, not a real position.
Assigning a role adds no mandatory model positions. Startup remains exactly
BUDGET_HEADER -> BUDGET_VALUE -> BUDGET_END, charged to H.

| Head | Classes in zero-based index order |
|---|---|
| Operation | Continue, Noul, Choice, Score |
| Writer | ADD_EVIDENCE, ADD_CANDIDATE, SET_RUBRIC, REVISE_QUESTION, SET_TARGET, FINISH, ABORT |
| Semantic status | Assessed, Insufficient, Conflicting, Unresolved, NoFeasibleChoice |
| Attribute type | Boolean, Int64, Decimal, Reference, SourceString, UTF8 |
| Source mode | CopyTyped, ParseNumeric, Construct |
| Feasibility | Feasible, Infeasible, Insufficient, Conflicting |
| Noul | False, True |

Preserve structural masks, including Choice-only NoFeasibleChoice and
numeric-only ParseNumeric. These class orders do not replace learned decisions.

## 2. Metadata encoding and common prefix

Every metadata payload is a fixed 128-bit record expanded into 128 FP32 inputs
of exactly 0 or 1, then passed through A2's 128->256->1024 encoder. Lowest-numbered
feature is the field's most significant bit (big-endian). Reserved/inapplicable
fields are zero. Reject invalid codes, counts or nonzero reserved fields.

Role selects packet-header, ordinary-descriptor or writer/control layout; the
model does not select a format. Packet headers are roles 1/26/32. Descriptor
metadata and ordinary evaluation/return boundaries use the common prefix and
descriptor body, with only their applicable fields. Roles 35-40 use the control
layout. Semantic/source/atom/continuous payloads use their declared encoders,
not a metadata encoder invented for every position.

Packet headers and ordinary descriptors share bits 0-63:

| Bits | Field |
|---|---|
| 0-3 | Protocol version: 1 |
| 4-6 | Port |
| 7-9 | Semantic status |
| 10-12 | Value kind |
| 13-14 | Source mode |
| 15-18 | Object kind |
| 19-31 | Presence/binding flags |
| 32-39 | Owner handle |
| 40-47 | Item index |
| 48-56 | Following content length, in model positions |
| 57-63 | Slot index |

Metadata codes deliberately include None and differ from some head indices:

| Field | Codes |
|---|---|
| Port | 0=None, 1=Noul, 2=Choice, 3=Score, 4=Budget |
| Status | 0=None; otherwise semantic-status head index+1 |
| Value kind | 0=None, 1=Boolean, 2=Int64, 3=Decimal, 4=Reference, 5=SourceString, 6=UTF8, 7=Continuous |
| Source mode | 0=None; otherwise source-mode head index+1 |
| Object kind | 0=None, 1=Source, 2=Target, 3=Question, 4=Alternative, 5=Bank, 6=Rubric, 7=Result, 8=Reason, 9=RequestSnapshot |

Other codes are rejected. Budget has no semantic-status head: startup status is
None; a successful finite delivery has a present continuous value. This is the
fixed startup port, not a fifth optional operation-gate action. Request inputs
do not carry future gold statuses, winners or feasibility labels.

| Flag bit | Meaning |
|---:|---|
| 19 | payload_present |
| 20 | owner_handle_present |
| 21 | item_index_present |
| 22 | primary_reference_present |
| 23 | secondary_reference_present |
| 24 | unit_reference_present |
| 25 | question_reference_present |
| 26 | target_reference_present |
| 27 | bank_reference_present |
| 28 | rubric_reference_present |
| 29 | rubric_midpoint_present |
| 30 | historical_presentation |
| 31 | snapshot_sealed |

Handle/index zero is valid when its presence flag is true. Absence is not inferred
from numeric truthiness. Presence has its typed role meaning: a present Boolean
false or assessed scalar zero is different from an absent value. Header/descriptor
fields refer only to known current contents; no annotation-only answer channel.

## 3. Packet-header body

| Bits | Field |
|---|---|
| 64-71 | Question handle |
| 72-79 | Target handle |
| 80-87 | Bank handle |
| 88-95 | Rubric handle |
| 96-99 | Presented evidence count |
| 100-104 | Bank candidate count |
| 105-108 | Presented target-attribute count |
| 109-113 | Attempt ordinal; zero for startup |
| 114-122 | Complete packet length |
| 123-124 | Scale direction |
| 125-127 | Reserved zero |

Direction is 0=NotApplicable, 1=Increasing, 2=Decreasing; reject 3. Score's numeric
range remains [0,1]; semantic anchors convey scale meaning. Exact rubric version,
global IDs and original source offsets remain authoritative sidecar bindings.

## 4. Ordinary-descriptor body

| Bits | Field |
|---|---|
| 64-71 | Primary reference |
| 72-79 | Secondary reference |
| 80-87 | Unit reference |
| 88-91 | Presented child-record count: attributes for target/candidate; attachments for reason |
| 92-93 | Scale direction |
| 94-95 | Reserved zero |
| 96-111 | Parameter A |
| 112-127 | Parameter B |

| Type/use | Parameter A | Parameter B |
|---|---|---|
| Source-backed content | First token's inclusive clipping offset | Last token's exclusive clipping offset |
| UTF-8 attribute | Original byte length 0..128 | Zero |
| Other descriptors | Zero | Zero |

Clipping uses the token-byte view with a verified mapping to the original source;
exact original offsets stay in the sidecar. If faithful mapping is unavailable,
revise the representation, never silently normalize or truncate it.

Primary identifies the bound source/object. For historical-result evidence,
secondary identifies the original request snapshot. For copied attributes,
secondary identifies the immutable source owner, with the exact attribute path
in the sidecar. Fields with no defined use are absent and zero.

Following-content length counts the presented nested contents: an attribute
includes its two key slots; a candidate includes four semantic slots plus its
presented attributes. Counts describe the actual presentation, not an implicit
expansion of every stored field. Preserve A1's framing and return formulas.

## 5. Writer/control layout

Roles 35-40 use this complete alternative 128-bit layout:

| Bits | Field |
|---|---|
| 0-3 | Protocol version |
| 4-6 | Port |
| 7 | Phase: 0=Thinking, 1=Answering |
| 8-25 | Remaining B |
| 26-44 | Remaining A |
| 45-51 | Remaining H |
| 52-70 | Remaining context |
| 71-76 | Remaining writer positions |
| 77-81 | Remaining attempts |
| 82-87 | Remaining object allocations |
| 88-90 | Previous writer action: 0=None, otherwise head index+1 |
| 91-99 | Prospective serialized request length; zero before question/target initialization |
| 100-105 | Objects allocated in this attempt |
| 106-110 | Current candidate count |
| 111-114 | Current evidence count |
| 115-118 | Latest committed record's attribute count |
| 119 | Snapshot sealed |
| 120 | Historical presentation |
| 121-124 | Mechanical outcome |
| 125-127 | Reserved zero |

| Outcome code | Mechanical meaning |
|---:|---|
| 0 | None |
| 1 | InvalidReference |
| 2 | Malformed |
| 3 | BudgetExhausted |
| 4 | ContextLimit |
| 5 | WriterLimit |
| 6 | AttemptLimit |
| 7 | ObjectLimit |
| 8 | ExecutionFailure |
| 9 | Cancelled |
| 10 | AnswerLimit |
| 11 | PacketOrEncodingLimit |
| 12 | WriterAborted |

Codes 13-15 are reserved. These are execution outcomes, never semantic statuses.
Cancel stops the job; it does not require a further model position or apology.
Reject an oversized proposed record before committing it. Prospective request
length remains <=256; rejected work still follows the established accounting.

WRITER_START payload is encoded control metadata. WRITER_TICK uses:

    payload = lifted_draft_feedback + encoded_control_metadata

Then use A2's wire normalization. These are the existing projections/one tick
position, not extra helper-model steps. Optional control roles do not create
new mandatory delimiters, semantic work from H or post-cancellation generation.

## 6. Tags, slots and local handles

Sum applicable tag embeddings in this fixed FP32 order:

    role + slot + owner + attribute_index + previous_writer_action

Omit inapplicable contributions entirely, then use the shared 256->1024 tag
projection and A2 wire formula. An absent contribution is not embedding row zero;
valid index zero uses its embedding. Role zero padding remains a zero vector.

Slots are zero-based within their field: question 0..7, candidate 0..3, each
rubric field 0..1, UTF-8 atom 0..15. Parallel byte heads use byte indices 0..127.
Preserve the existing target/key/reason/source limits and attribute indices 0..7.

Use an append-only local table per attempt. Assign 0..255 in first-encounter
order during sealed presentation rendering, using section 11's traversal rather
than writer catalogue order or dictionary iteration. Never rebind a handle's
immutable Ref/version. Preserve the
request table prefix through its return; reserve capacity for bounded result/
reason objects before dispatch. Actual allocations still obey the separate
32-new-Ref limit, including versions/results/reasons as specified in A1.

Historical presentation translates references into the current presentation
table while preserving exact original bindings, without mutating the stored
result. Historical boundaries are data, not dispatch instructions. Re-encoding
tags/reference atoms does not change stored semantic slots or detach continuous
values. Global identities, versions and exact source offsets remain in the
authoritative sidecar; handles are not semantic labels.

## 7. Atom bits and numeric helper formulas

An atom is eight big-endian bytes. Features 0..63 are its bits, most significant
first, as exact FP32 zero/one inputs. Int64 uses two's complement. Decimal atom
one is the int64 coefficient; atom two is seven zero bytes followed by the signed
int8 exponent. For example, -2 is 00 00 00 00 00 00 00 FE, not int64 sign extension.
Boolean/reference unused bytes and UTF-8 final padding remain zero as in A1.

For numeric atoms, features 64/65/66 are sign, bounded coefficient magnitude and
decimal exponent/18. Treat int64 n as (coefficient=n, exponent=0). Both decimal
atoms receive helpers derived from the complete canonical (c,e) pair.

Sign is -1/0/+1. Let m=unsigned magnitude of c; do not overflow signed abs at
INT64_MIN. For m=0, M=0. Otherwise k=highest set-bit index of m, and:

    M = (k + 1 + (m - 2^k) / 2^k) / 64

This is a bounded log-scale approximation, not an unspecified log kernel.
Compute k/remainder as exact integers. Convert remainder to FP64, divide by
2^k, add k+1, divide by 64, then cast to FP32, with separate operations and A3
round-to-nearest-even rules. Compute exponent helper as FP64 e/18, then FP32.
Future neural helper computation is on CUDA, not a CPU model-encoding fallback.

| Value | Sign | Magnitude | Exponent helper |
|---|---:|---:|---:|
| Int64 0 | 0 | 0 | 0 |
| Int64 1 | 1 | 1/64 | 0 |
| Int64 -1 | -1 | 1/64 | 0 |
| Int64 -2^63 | -1 | 1 | 0 |
| Decimal (120,-2), representing 1.20 | 1 | 63/512 | FP32-rounded -1/9 |

Boolean/reference/UTF-8 helpers are all zero. Exact values come from bytes and
typed records, not these approximate magnitude features. Continuous Noul/Score
and budget values keep A2's separate [1,v,v*v] encoders and gradient rules;
they are not converted into discrete atom bits.

## 8. Counter normalization and timing

Use this fixed order for the eight FP32 gate/writer features:

    B_remaining        / 131072
    A_remaining        / 262144
    H_remaining        / 64
    context_remaining  / 262144
    writer_remaining   / 32
    attempts_remaining / 16
    objects_remaining  / 32
    phase              (0=Thinking, 1=Answering)

Bounded integer numerators and power-of-two denominators are exactly representable
in FP32. Validate ranges before normalization; do not hide underflow by clamping.
Sample at the current accounted frontier, after charging the current position
and before applying its predicted action. The attempt is already counted before
WRITER_START. Outside a draft, writer/object allowances are 32; inside, subtract
that attempt's actual use. Reservations do not count as spent model positions.
Starting the answer does not reset B; all A1/A3 admission/ledger rules still apply.

An illustrative post-first-writer-position state is B=511, A=128, H=61,
context=262000, writer=31, attempts=15, objects=32, phase=0. Its feature vector is
exactly [511/131072,128/262144,61/64,262000/262144,31/32,15/16,1,0].

## 9. Noul prototype initialization/readout

On the pinned base revision 2fc06364715b967f1860aea9cf38778875588b17, the tokenizer
JSON vocabulary entries are lowercase false=3721 and true=1802. Static vocabulary
inspection verified these IDs; this is not model execution. Tokenizer JSON SHA256:

    5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42

| Prototype index | Meaning | Frozen vocabulary embedding row |
|---:|---|---:|
| 0 | False | 3721 |
| 1 | True | 1802 |

Clone the rows to independent FP32 trainable prototypes on CUDA. Do not alter the
frozen vocabulary, add special tokens, take only the first token of an arbitrary
spelling, or normalize prototypes before the pointer projection. Use aligned
Qev query/key projections:

    logit_i = dot(Q(h_evaluate), K(prototype_i)) / 16
    probability_true = sigmoid(logit_true - logit_false)

Train the logit difference with the agreed stable BCE. The separate semantic
status head determines whether that probability is returned. These word-row
initializations do not establish latent question understanding or calibration.

## 10. Required reference fixtures and closure

Specify fixtures for every role/layout and masked/reserved class; handle/index
zero versus absence; assessed scalar zero versus missing; empty UTF-8; decimal
exponent bytes; int64 extrema; counter boundaries; and historical reference
translation. Require matching metadata bits and appropriate neural-output numeric
agreement under A3. Tag/atom/source/continuous graph paths retain their required
gradients. Verify no new positions or Ref allocations are hidden by encoding.

The metadata interval specification was checked statically for complete coverage
with no overlaps. No golden neural-vector artifact, implemented codec, GPU test,
training result or operational readiness is established by this document.
B1 is recorded as a design; A1-A4 and B1 still require implementation, reviewed
fixture/run artifacts and the existing empirical gates.

## 11. Normative role application and target-attribute presentation

Status: recorded response to Astra findings F2/F4, not implemented, independently
re-reviewed or validated by matching neural vectors. Preserve the historical
[Astra report](review-astra-2026-11-05.md) and its original source hashes. Other
review findings remain open. Resource admission and control emission/stop policy
are recorded in their separate owners; this mapping alone does not establish
those contracts, exact module wiring, evaluation design or passing evidence.

### 11.1 Owners, tags and lengths

X=request snapshot, Y=result, Q=question, T=target, U=rubric, B=candidate bank,
C_i=candidate at original bank index i, S=source, Z=non-assessed reason. An
attribute is inline and owned by T, C_i or Z, without its own attribute Ref.

Keep tag addition order role -> slot -> owner -> attribute index -> previous
writer action. Omit absent contributions; a valid index zero uses its embedding.
Every field/tag not enabled below is zero/absent. Reject unexpected nonzero
fields rather than silently accepting a second representation.

Following length counts the positions belonging to the record AFTER its
descriptor, including nested presented contents. On a header it is complete
packet length minus one; on a boundary it is zero. Metadata's slot-index field
is always zero in this profile; slot tags apply to payload positions below.

### 11.2 All roles

M=metadata encoder, Sem=canonical semantic payload, Src=cached-source-state
encoder, Atom=B1 67-feature encoder, Value=task continuous encoder, Budget=its
separate control-value encoder. N=packet length; A_T/A_i=presented attribute
costs; R=reason-attachment cost; L=historical presentation length. Attribute a is
the attribute index, or the attachment index for an attribute inside a reason.
Following length applies only to metadata positions. All real positions get
their role tag. Previous-action tag is absent except for role 36 as defined below.

| ID | Role | Payload | Owner tag | Slot tag | Attribute tag | Following length |
|---:|---|---|---|---|---|---|
| 1 | REQUEST_HEADER | M | X | None | None | N-1 |
| 2 | QUESTION_SLOT | Sem | Q | 0..7 | None | Not a metadata field |
| 3 | TARGET_DESCRIPTOR | M | T | None | None | Request 2+A_T; return 2 |
| 4 | TARGET_SLOT | Sem | T | 0..1 | None | Not a metadata field |
| 5 | SOURCE_EVIDENCE_DESCRIPTOR | M | S | None | None | m |
| 6 | SOURCE_EVIDENCE_SLOT | Src | S | 0..m-1 | None | Not a metadata field |
| 7 | PRIOR_RESULT_EVIDENCE_DESCRIPTOR | M | Historical Y | None | None | L |
| 8 | RUBRIC_DESCRIPTOR | M | U | None | None | 10 |
| 9 | RUBRIC_DIMENSION_SLOT | Sem | U | 0..1 | None | Not a metadata field |
| 10 | RUBRIC_CONDITIONS_SLOT | Sem | U | 0..1 | None | Not a metadata field |
| 11 | RUBRIC_ZERO_ANCHOR_SLOT | Sem | U | 0..1 | None | Not a metadata field |
| 12 | RUBRIC_ONE_ANCHOR_SLOT | Sem | U | 0..1 | None | Not a metadata field |
| 13 | RUBRIC_MIDPOINT_SLOT | Sem or canonical zero | U | 0..1 | None | Not a metadata field |
| 14 | CANDIDATE_DESCRIPTOR | M | C_i | None | None | 4+A_i |
| 15 | CANDIDATE_SLOT | Sem | C_i | 0..3 | None | Not a metadata field |
| 16 | ATTRIBUTE_DESCRIPTOR | M | Containing T/C_i/Z | None | a | 2+value-body length |
| 17 | ATTRIBUTE_KEY_SLOT | Sem | Containing T/C_i/Z | 0..1 | a | Not a metadata field |
| 18 | BOOLEAN_ATOM | Atom | Containing T/C_i/Z | 0 | a | Not a metadata field |
| 19 | INT64_ATOM | Atom | Containing T/C_i/Z | 0 | a | Not a metadata field |
| 20 | DECIMAL_COEFFICIENT_ATOM | Atom | Containing T/C_i/Z | 0 | a | Not a metadata field |
| 21 | DECIMAL_EXPONENT_ATOM | Atom | Containing T/C_i/Z | 1 | a | Not a metadata field |
| 22 | REFERENCE_ATOM | Atom | Containing T/C_i/Z | 0 | a | Not a metadata field |
| 23 | ATTRIBUTE_SOURCE_SLOT | Src | Containing T/C_i/Z | 0..m-1 | a | Not a metadata field |
| 24 | UTF8_ATOM | Atom | Containing T/C_i/Z | 0..ceil(b/8)-1 | a | Not a metadata field |
| 25 | EVALUATE | M | X | None | None | 0 |
| 26 | RESULT_HEADER | M | Y | None | None | N-1 |
| 27 | NOUL_VALUE | Value | Y | 0 | None | Not a metadata field |
| 28 | SCORE_VALUE | Value | Y | 0 | None | Not a metadata field |
| 29 | REASON_DESCRIPTOR | M | Z | None | None | 4+R |
| 30 | REASON_SLOT | Sem | Z | 0..3 | None | Not a metadata field |
| 31 | RETURN_END | M | Y | None | None | 0 |
| 32 | BUDGET_HEADER | M | None | None | None | 2 |
| 33 | BUDGET_VALUE | Budget | None | 0 | None | Not a metadata field |
| 34 | BUDGET_END | M | None | None | None | 0 |
| 35 | WRITER_START | Control M | None | None | None | Not a metadata field |
| 36 | WRITER_TICK | Control M + lifted draft feedback | None | None | None | Not a metadata field |
| 37 | CONSUME_TICK | Control M | None | None | None | Not a metadata field |
| 38 | PHASE_CLOSE | Control M | None | None | None | Not a metadata field |
| 39 | RUNTIME_LIMIT | Control M | None | None | None | Not a metadata field |
| 40 | TERMINATE | Control M | None | None | None | Not a metadata field |

Role 36 receives the most recent executed seven-action writer event's tag, or
none when no such event exists. Initialization is not an eighth writer action.
Padding is outside the real position count; IDs 41-63 remain rejected.

### 11.3 Packet-header profiles

This table, the role table and zero defaults cover every common/header field.
Primary/secondary/unit refs are absent on all headers. Item index is absent;
its value/presence flag and metadata slot index are zero.

| Field | REQUEST_HEADER | RESULT_HEADER | BUDGET_HEADER |
|---|---|---|---|
| Protocol | 1 | 1 | 1 |
| Port | Requested operation | Original requested operation | Budget |
| Semantic status | None | Actual returned status | None |
| Value kind | None | Assessed Noul/Score Continuous; assessed Choice Reference; otherwise None | Continuous |
| Source mode | None | None | None |
| Object kind | RequestSnapshot | Result | None |
| Payload present | False | Exactly when Assessed | True |
| Owner | X present | Y present | Absent |
| Following length | N-1 | N-1 | 2 |
| Question/target refs | Q/T present | Original Q/T present | Absent |
| Bank ref | B for Choice only | Original B for Choice only | Absent |
| Rubric ref | U for Score only | Original U for Score only | Absent |
| Evidence count | Top-level request evidence count | 0 | 0 |
| Bank candidate count | Original bank size for Choice; otherwise 0 | Same original bank size | 0 |
| Target-attribute count | Presented request count | 0 | 0 |
| Attempt ordinal | This request's ordinal | Original request's ordinal | 0 |
| Complete length | N | N | 3 |
| Direction | Score rubric direction; otherwise NotApplicable | Same | NotApplicable |
| Midpoint flag | Score rubric flag; otherwise false | Same | False |
| Sealed flag | True | True | False |
| Historical flag | False | Current false; historical replay true | False |

An assessed Choice return retains the original bank size and the selected
candidate's ORIGINAL bank index, not a renumbered index zero. Header payload
presence refers to the decision value, not mere existence of the packet.

### 11.4 Descriptor/boundary profiles

All use protocol 1, enclosing request/result port, semantic status None and
source mode/value kind None except attribute descriptors. Object kind matches
the owner; owner is present except BUDGET_END. Header-only question/target/bank/
rubric refs are absent. Metadata slot index is zero. Sealed is true in request/
result packets, false for budget boundary. Historical flag follows presentation.
Only RUBRIC_DESCRIPTOR sets descriptor direction/midpoint. Other such fields zero.

| Role | Item index | Primary ref | Secondary ref | Unit ref | Child count | Parameters |
|---|---|---|---|---|---|---|
| TARGET_DESCRIPTOR | Absent | Absent | Absent | Absent | Request attrs; return 0 | Zero |
| SOURCE_EVIDENCE_DESCRIPTOR | Evidence/attachment ordinal | Absent; owner is S | Absent | Absent | 0 | Start/end clipping |
| PRIOR_RESULT_EVIDENCE_DESCRIPTOR | Evidence/attachment ordinal | Absent; owner is historical Y | Original request X | Absent | 0 | Zero |
| RUBRIC_DESCRIPTOR | Absent | Absent | Absent | Absent | 0 | Zero |
| CANDIDATE_DESCRIPTOR | Original bank index i | B | Absent | Absent | Candidate attrs | Zero |
| ATTRIBUTE_DESCRIPTOR | Attribute/attachment index a | Declared source provenance | CopyTyped's source owner | Declared unit ref | 0 | By type below |
| REASON_DESCRIPTOR | Absent | Enclosing result Y | Absent | Absent | Attachments | Zero |
| EVALUATE | Absent | Absent | Absent | Absent | 0 | Zero |
| RETURN_END | Absent | Absent | Absent | Absent | 0 | Zero |
| BUDGET_END | Absent | Absent | Absent | Absent | 0 | Zero |

The descriptor bits 88-91 mean presented CHILD records: target/candidate
attributes or reason attachments. This clarifies the earlier field name without
changing offsets/widths. Source/result evidence item ordinal refers to its
position in the enclosing request evidence list or reason attachment list.

Record descriptors have payload_present=true except an attribute uses its
actual value-presence flag. Boundaries EVALUATE/RETURN_END/BUDGET_END use false.
BUDGET_END has port Budget, object kind None and no owner.

An attribute keeps its declared value kind even if absent. Present values use
their recorded creation mode. Absent values use mode None, no value body and no
fabricated source/copy provenance; their known unit may still be present.
SourceString parameters are clipping A/B; UTF8 A=byte length/B=0; other types
use zero. CopyTyped requires the exact existing owner/path binding. Transport
copy into a return does not change the recorded creation mode to CopyTyped.

### 11.5 Target grammar and position formulas

Requests serialize in this exact order:

    REQUEST_HEADER
    QUESTION x 8
    TARGET_DESCRIPTOR
    TARGET x 2
    TARGET_ATTRIBUTES in declared order
    EVIDENCE_RECORDS in declared order
    RUBRIC_RECORD for Score
    CANDIDATE_RECORDS in bank order for Choice
    EVALUATE

Target attributes are complete records indexed 0..n-1, with no placeholders for
unused capacity. Returns have TARGET_DESCRIPTOR/TARGET x2 but ZERO target
attributes. Preserve the exact target Ref/version and canonical slots; do not
automatically expand other stored fields.

Let E=sum of presented evidence-record costs. Then:

| Presentation | Position count |
|---|---|
| Noul request | 13+A_T+E |
| Choice request | 13+A_T+E+sum_i(5+A_i) |
| Score request | 24+A_T+E |
| Assessed Noul return | 14 |
| Assessed Score return | 25 |
| Assessed Choice return | 18+A_i |
| Non-assessed Noul/Choice | 18+R |
| Non-assessed Score | 29+R |

One target int64 attribute costs four positions: minimal Noul request 17,
assessed return still 14. Target omission preserves the existing 64-position
return envelope; changing this later requires a new codec/capacity review.

### 11.6 Absent values and reason attachments

- Absent attribute: descriptor plus two meaningful key slots, following length 2,
  no value body. Keep declared type; absence is not false or zero.
- Present Boolean false: descriptor, two keys and one zero-valued Boolean atom.
- Present empty UTF8: descriptor/two keys, byte length zero, presence true.
- Assessed scalar zero: one present continuous value position containing zero.
- Non-assessed decision: no value/selected candidate; reason instead.
- No optional bank/rubric: omit its whole record and clear header reference.
- Missing Score rubric or empty Choice bank: invalid request, not a fake rubric
  or semantic NoFeasibleChoice.

Absent midpoint still occupies two role-13 positions. Canonical payloads are
EXACT ZERO; retain U owner and 0/1 slot tags. Mask their alignment/reconstruction/
contrast losses. The final wire vectors need not be zero because syntax tags
remain. Unused predicted midpoint floats must not become a hidden channel.

Reasons have an ordered list of <=8 attachments, aggregate cost R<=32:
attribute records owned by Z, source-evidence records owned by their source, or
historical-result evidence owned by that result. Attribute/evidence item indices
use attachment-list position. No free reference expansion is introduced.

### 11.7 Historical rendering

Render the stored presentation, not a fresh expansion of its whole object store.
Preserve original operation, status, presence, kind, attempt ordinal, bank
indices, canonical semantic tensors, attached continuous tensors, presented
counts, clipping, direction, midpoint flag and lengths. Keep sealed true.

Rewrite ONLY local handles/owner tags, reference-atom handle bytes, and historical
metadata flags (true). Preserve exact Ref/version bindings; never fetch newer
versions, add omitted target attrs or detach a scalar. The enclosing evidence
descriptor gets the NEW outer evidence/attachment ordinal; inner indices remain
original. An inner RETURN_END is data, not native resumption or outer completion.

### 11.8 Deterministic handle traversal

Writer catalogue indices are distinct from packet-local wire handles. Build the
wire table while rendering the sealed presentation, scanning positions in order:

1. Header: owner -> question -> target -> bank -> rubric.
2. Descriptor: owner -> primary -> secondary -> unit.
3. Reference atom: its referenced immutable object.
4. Assign next unused handle only on first encounter of that exact Ref/version.

Do not recursively traverse objects or intern sidecar-only refs. Preserve this
table as an immutable return prefix; append result/reason/newly presented refs
using the same traversal. Historical refs translate through original bindings.
Dictionary iteration, parallel-head completion and writer catalogue order are
not alternative wire-table traversal policies. Allocation holds follow the
separately recorded [resource contract](position-limits-and-budget-accounting.md#admission-and-reservation):
two completion Refs, dedicated sealing objects and 2+min(32,m) completion handles,
with m counting compatible frozen-catalogue identities outside the request table.
It preserves this traversal and does not preassign handles or intern K.

### 11.9 Control field sources and startup value

All control fields come from one event-state snapshot under B1's timing. The
recorded [control-flow v1](control-flow-v1.md) now fixes emission/stop policy:
roles 37/38/40 are not emitted, code12=WriterAborted, extra consumption=0 and
native chat delimiters perform phase closure/termination. Field layouts remain
defined for future profiles; defined IDs do not imply v1 emission.

| Role | Port | Draft fields | Previous-action tag |
|---|---|---|---|
| WRITER_START | Selected operation | New draft | Absent |
| WRITER_TICK | Selected operation | Current draft | Last executed writer action if any |
| CONSUME_TICK | Delivered operation | Absent | Absent |
| PHASE_CLOSE | None | Absent | Absent |
| RUNTIME_LIMIT | Causal operation or None | Originating draft if any | Absent |
| TERMINATE | None | Originating draft if any | Absent |

Without draft state, writer/object remaining=32 and draft length/allocated
objects/child counts/previous-action code=0. Snapshot-sealed=false, except
consumption identifies its sealed delivered call. With a draft, length is the
prospective wire request size: zero before Q/T initialization, then
13+A_T+evidence+rubric+candidates. Size telemetry does not debit positions early.

BUDGET_VALUE is the CLAMPED PREDICTED SCORE s, with its intentional stop-gradient.
Actual granted/overridden B is in structured counters; an override never rewrites
the predictor's feedback value. Raw regression remains against the raw readout.
Task continuous returns keep their connected gradients.

This records the role mapping and target-presentation decisions. Actual field/
tag/reference fixtures and GPU numeric agreement remain required. Exact module
wiring and other Astra findings are not silently resolved by this input schema.
Admission and control flow are recorded in their separate owners; implementation,
independent re-review and validation remain unperformed.
