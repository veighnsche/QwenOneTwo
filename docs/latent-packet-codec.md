# Latent request and result packet codec

Status: recorded design choice, not implemented or GPU-validated. These are
versioned starting layouts/capacities, not proof of shared semantic understanding.
This document authorizes no coding, model workload, service change or deployment.

Owner: packet records, neural position layout, typed-value representation and
request/result framing. The [semantic layout](latent-representation-v1.md) owns
field meanings; [memory](memory-and-request-snapshots.md),
[statuses](result-statuses.md) and [budget](position-limits-and-budget-accounting.md)
own bindings, validity and charging. Follow the
[specification](architecture-and-training-plan.md) and [handoff](implementation-handoff.md).

The recorded [B1 encoding](neural-input-encoding-v1.md) owns exact role/class IDs,
big-endian metadata offsets, tag/handle conventions, atom helpers and Noul
initialization. Use that map for both training and serving. It adds no required
positions and does not constitute a validated codec.

[B1 section 11](neural-input-encoding-v1.md#11-normative-role-application-and-target-attribute-presentation)
now records the exact per-role field/tag/owner/length/absence rules, historical
rewrite and handle traversal. It specializes the generic summaries here; use
it rather than deciding which fields are applicable independently.

## Two linked representations

1. An immutable typed record preserves exact identities, attributes, values,
   source bindings and versions. It is authoritative for bookkeeping.
2. A sequence of 1024-wide model input vectors presents the record's contents
   to Qwen/Qev on the existing causal history.

Learned semantic vectors are not lossless arbitrary-identifier/integer storage.
Transport preserves exact data; interpreting the neural presentation remains
learned and fallible. Keep all model encoding on the selected GPU and preserve
required graph connections during training.

| Content | Representation |
|---|---|
| Question | Eight learned semantic vectors |
| Alternative | Four learned semantic vectors per candidate |
| Rubric | Five semantic fields, two vectors each |
| Target | Starting default: two semantic vectors |
| References | Exact typed/versioned record refs; small packet-local handles for model inputs |
| Attributes, names and source spans | Typed recoverable values and explicit source bindings |
| Noul/Score result | Authoritative FP32 scalar plus continuous learned value embedding |
| Operation, status and presence | Typed enums/flags and learned embeddings |

Global run/request IDs and UUIDs stay in bookkeeping. Qwen selects local handles
from the bound object catalogue; it need not generate opaque global identities.

## Role-aware neural positions

Each position has an explicit role, such as QUESTION_SLOT, TARGET_SLOT,
EVIDENCE_SLOT, CANDIDATE_SLOT, RUBRIC_ZERO_ANCHOR, ATTRIBUTE_VALUE, RESULT_VALUE,
EVALUATE or RETURN_END. A shared input encoder combines the typed payload and
learned role, slot and local-object embeddings into one model input vector.

Both adapter modes use this role system. Roles define interface syntax; the
questions, criteria and alternatives remain learned content. Adding role data
to an existing vector does not add a position. Explicit descriptor/boundary
vectors do. Encoder parameters, normalization and mode-specific numerical
behavior remain subject to training/serving verification.

For learned semantic slots, the canonical normalization used by alignment is
also used for reconstruction and receiver presentation. Do not expose raw
magnitude or another per-record gain that the semantic objective discarded.
This does not normalize scalar readouts or replace continuous numeric payloads.
The [model configuration](model-configuration.md) records exact normalizer and
encoder settings for A2; GPU behavior remains unverified in the
[final audit](final-decision-audit.md).

## Request layout, evaluated under W+D

    REQUEST_HEADER
    QUESTION x 8
    TARGET_DESCRIPTOR
    TARGET x 2
    TARGET_ATTRIBUTES  (declared order; none when count is zero)
    EVIDENCE_RECORDS
    RUBRIC_RECORD       (Score only)
    CANDIDATE_RECORDS   (Choice only)
    EVALUATE

The header carries operation, protocol version, counts and presence flags.
The immutable record pins exact references/versions. Common framing costs
1+8+1+2+1 = 13 positions before evidence, attributes and operation-specific records.

Target attributes immediately follow the two target slots, indexed 0..n-1.
Their descriptor following length is 2+A_T, where A_T is full attribute cost;
the header carries the presented count. No unused attribute slots are emitted.

An evidence record is one descriptor plus content positions. Its descriptor
binds source/object and content length. For source evidence, present projected
cached source states from the selected immutable span, keeping original token
IDs/offsets in the exact record. Each presented source-state vector costs a
position. A handle alone does not supply content. Do not replay the entire
prompt as an independent decision prefill or claim selected references isolate
attention from the existing mixed history.

For offline clean tests only, the [clean-test profile](clean-test-construction-v1.md)
freshly encodes permitted excerpt-only sources; original contextual states are
transfer fixtures. Production still uses its actual uninterrupted shared history.

A candidate record is one descriptor, four meaning vectors and its exact
attribute records. Its semantic minimum is five positions, plus attributes.
Bank order is fixed in the sealed request. No gold feasibility/winner labels
are candidate input facts.

A rubric record is one descriptor plus ten vectors in the fixed order dimension,
conditions, zero anchor, one anchor and midpoint. Keep ten slots even when the
midpoint is absent; the descriptor carries an explicit presence flag. It also
carries exact range/direction/version metadata. A rubric costs eleven positions.
It must not encode a desired target score.

EVALUATE is one learned boundary input vector. Read status and operation values
from its receiver hidden state after the full packet. For Choice, form candidate
keys from the receiver hidden state at each candidate record's final position;
the evaluation state queries those keys. These choices need alignment training.

## Exact attributes

The [typed-value ABI](typed-value-abi.md) owns exact expansion, value-head caps
and return admission. The type meanings below are a summary, not an alternative.

    ATTRIBUTE_DESCRIPTOR
    ATTRIBUTE_KEY_MEANING x 2
    TYPED_VALUE_BODY

The descriptor contains value type, presence, source binding and unit reference
where relevant. The key vectors convey what the attribute means. Pine's rate,
for example, has a throughput key, integer 120, labels/hour unit binding and the
source clause that supplied it. Runtime validation does not infer its relevance.

| Type | Exact representation |
|---|---|
| Boolean | Boolean value |
| Integer | Signed 64-bit integer |
| Decimal quantity | Integer coefficient, decimal exponent and unit reference |
| Source-backed name/string | Immutable span and original token IDs |
| Newly constructed string | Length-delimited UTF-8 bytes |
| Reference | Typed object identity and version |

Never cast a large integer directly to BF16 and call it exact. Integer/byte
encoders can use lossless bit features with generic magnitude features to help
interpretation; canonical values remain in the typed record. Source-backed
strings use bound source representations. Constructed strings require bounded
byte payloads. All presented lengths count for admission, including key and
descriptor positions. Exact-attribute recovery/comparison must be tested.

The [writer schedule](latent-writer-schedule.md) records native steps and parallel
record emission. The [typed ABI](typed-value-abi.md) now fixes eight-byte atoms,
per-type costs, bounded parallel value heads and source/UTF-8 limits. Model
instantiation and GPU correctness remain unverified; counts are not test results.

## Typed Qev outputs

At EVALUATE, a learned status head produces Assessed, Insufficient evidence,
Conflicting evidence, Unresolved or Choice-only No feasible choice. The Noul
readout estimates truth of the bound proposition; Choice selects from the
frozen bank; Score returns a continuous value under the bound rubric.

Only Assessed exposes a decision value. Mask/discard an internally computed
numeric readout for other statuses. The runtime attaches the pending request
binding; Qev does not predict its identity. No gold substitution is allowed.

## Result layout, consumed under W+R

    RESULT_HEADER
    ORIGINAL_QUESTION x 8
    TARGET_DESCRIPTOR
    TARGET x 2
    RUBRIC_RECORD              (Score)
    VALUE                      (assessed Noul/Score)
    SELECTED_CANDIDATE_RECORD  (assessed Choice)
    REASON_RECORD              (non-assessed outcome)
    RETURN_END

Retrieve the original question unchanged and the exact Score rubric version;
do not regenerate their meanings. Resolve a Choice index through the frozen
bank and retrieve only the selected entry's semantic and exact contents. A
naked scalar, index or UUID is insufficient.

Returns present ZERO target attributes: TARGET_DESCRIPTOR following length=2,
target child count=0 and header target-attribute count=0. Preserve the exact
target Ref/version and two canonical slots without automatic stored-field
expansion. Candidate selected attributes still follow their candidate record.
Thus the existing 14/25/18+A_i and non-assessed return formulas stay unchanged.

Starting default for a non-assessed reason is one descriptor plus four learned
reason vectors and bounded relevant references/content. Missing information
does not receive a fabricated source span. The reason and associated contents
must fit the return ceiling. A value cannot reveal supporting facts by itself.

RETURN_END is a learned boundary input under R, supplying the native continuation
state. Under [control-flow v1](control-flow-v1.md), consumption completes at that
boundary and gating resumes in the same native phase. No extra CONSUME_TICK or
consumption hold exists. No gate runs inside D evaluation/return delivery.

## Continuous values and presence

    FP32 scalar -> continuous learned encoder -> one RESULT_VALUE vector

Preserve this tensor's graph connection. Do not convert Noul/Score values into
integer bins, decimal characters, detached floats or discrete lookup IDs. Native
continuation losses can then flow through the result encoder into Qev's scalar
readout and request formation. Hard status/Choice/reference selections still
need explicit supervision; this does not create a derivative through argmax.

An assessed zero has value_present=true and a value slot. A non-assessed result
has value_present=false and no value slot. Status and value validity are
structural, not numeric truthiness or proximity to 0.5.

## Position ledger and admission

| Packet | Structural positions before variable contents |
|---|---:|
| Common request framing | 13 |
| Noul request | 13 plus evidence/attributes |
| Choice request | 13 plus evidence and 5 per candidate plus attributes |
| Score request | 24 plus evidence/attributes |
| Common result framing | 13 |
| Assessed Noul result | 14 |
| Assessed Score result | 25 |
| Assessed Choice result | 18 plus selected attributes |

Non-assessed results add the bounded reason instead of a value. Question/rubric
echoes, selected contents, source representations and every descriptor count.
These are structural counts, not measured lengths for the worked examples.

Keep the existing 256-position request and 64-position return ceilings. Calculate
complete expansion before dispatch, reserve bounded return/consumption, then
debit actual use. Oversized requests must be explicitly revised or declined;
never silently truncate fields or alternatives. Returning a large selected
entry must be checked before dispatch as well as evaluating the request.

Check the worst possible selected-entry return, not only an expected winner.
With this framing, every Choice candidate needs encoded attributes A_i such
that 18+A_i<=64. Non-assessed Noul/Choice returns have 18 positions before extra
reason/reference contents; non-assessed Score has 29. These are derived bounds,
complemented by the [typed ABI](typed-value-abi.md): at most eight attributes per
record, 32 aggregate extra reason positions and eight attachments. Return maxima
are 50 for Noul, 61 for Score and max(50,18+largest_attribute_cost) for Choice.
No partial reason/attribute is silently cut.

The [full resource contract](position-limits-and-budget-accounting.md#admission-and-reservation)
adds independent completion handle and immutable-record holds to this position
envelope. Reason attachments are inline or reuse sealed catalogue K; they create
no extra Ref-bearing objects. Hold Result+Reason slots and dedicated sealing
objects from intent entry, and verify all three resource inequalities before D.
Handles are assigned only for actually presented identities in B1's order.

The fixed startup budget port uses its implicit fixed meaning rather than an
optional semantic question/rubric. Its minimal return is budget header, continuous
value and return boundary: three positions charged to startup H, not additional
call allowance B. Use W+R for native consumption on the same state history.

This codec makes field roles, binding, continuous paths and structural costs
explicit. It does not establish that its latent representations communicate
their intended meaning; that remains a training/GPU feasibility requirement.
