# Typed-value ABI and bounded returns

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

Owner: typed-body counts, bounded value heads and return envelope. Follow the
[codec](latent-packet-codec.md), [writer](latent-writer-schedule.md),
[budget](position-limits-and-budget-accounting.md) and
[specification](architecture-and-training-plan.md).

## 1. Attribute wrapper

Each attribute is one descriptor, two key-meaning vectors and its value body:
total cost 3+body_length. Descriptor fields include type, presence, length,
attribute index and local owner/source/unit handles. Exact global identities,
versions and source offsets remain immutable sidecar data. Permit at most 256
local handles. A reference never recursively expands contents; required content
must be explicitly presented/charged. Key vectors carry attribute/unit meaning;
an opaque unit ID is not sufficient.

## 2. Atom and position costs

One discrete atom contains eight big-endian bytes/64 payload bits and becomes
one 1024-wide input vector. Use bit features/type/position, not one BF16 integer
cast. Exact records remain authoritative; neural interpretation is learned.

| Type | Body | Whole attribute | v1 limit |
|---|---:|---:|---|
| Absent | 0 | 3 | Explicit present=false |
| Boolean | 1 | 4 | False/true |
| Int64 | 1 | 4 | Signed 64-bit |
| Decimal | 2 | 5 | Int64 coefficient, exponent -18 through +18 |
| Reference | 1 | 4 | Valid packet-local handle |
| Source-backed string | m | 3+m | 1-32 covered source-token states |
| Constructed UTF-8 | ceil(b/8) | 3+ceil(b/8) | 0-128 bytes |

These are capacities, not permission to round/truncate/drop data. Unsupported
values need another valid complete representation or explicit revision/rejection.

## 3. Body details

Boolean is 0/1 with unused bits zero. Int64 uses two's complement. Decimal is
coefficient*10^exponent, coefficient in atom one and signed exponent in atom two
with unused bytes zero; descriptor binds the unit. Preserve scale, e.g. 120 and
-2 represent 1.20. A reference atom contains the local handle with unused bits zero.

[B1](neural-input-encoding-v1.md) pins the exponent atom to seven zero bytes plus
the signed int8 exponent in the final byte: -2 is 00 00 00 00 00 00 00 FE. This
is not int64 sign extension. B1 also fixes bit order, helper features and the
explicit presence flags distinguishing zero/empty/handle zero from absence.

Source-backed strings retain exact byte spans/token mappings and present one
projected cached state per covered token. Boundary offsets preserve a literal
substring if the token window is broader. Constructed strings use consecutive
eight-byte UTF-8 chunks, final zero-padding and explicit original length. Validate
the complete UTF-8 sequence, not chunks individually. Empty string is present=true,
length zero; it differs from absence despite the same three-position wrapper cost.

Continuous Noul/Score returns cost one continuous value position preserving the
FP32 tensor/gradient, never discrete byte predictions. Budget control feedback
has its separately declared gradient boundary.

## 4. Writer production

At most eight inline attributes per target/candidate record. Within one native
event predict count, active types/presence/source modes, two key vectors and
typed bodies. Shared index-conditioned decoders produce bounded parallel outputs.

Per B1 section 11, request target attributes follow TARGET x2 in declared order;
ordinary returns echo no target attributes. Selected candidate attributes still
return in full within A_i<=46. Target Ref/version and canonical slots remain
bound even when its stored attributes are omitted from the return presentation.

Modes: learned pointer to existing typed value; strict parse of a learned-selected
numeric literal; or construct. The parser neither searches the prompt nor decides
which fact matters, and does not interpret available as a boolean.

| Constructed type | Output |
|---|---|
| Boolean | Binary head |
| Int64 | Eight parallel byte classifications |
| Decimal | Eight coefficient bytes and 37-way exponent |
| Reference | Pointer into allowed catalogue |
| Source string | Source/span pointers and boundary offsets where needed |
| UTF-8 | Length head and at most 128 parallel byte classifications |

These are head operations, not 64 extra backbone steps or an unbounded helper
generator. Validate before draft acceptance. Bad UTF-8, overflow or refs trigger
revision/abort, not factual correction. Losses follow active type/mode; copied/
parsed branches receive no invented constructed-byte targets.

## 5. Evidence

Source evidence costs 1+m positions with 1<=m<=32. Prior-result evidence costs
1+L for its explicitly presented typed packet L<=64. Preserve original meaning/
bindings; historical evidence is not the current completion. Inner boundary
markers do not dispatch nested calls/end the outer packet. Never recursively
expand refs. Re-presented continuous tensors retain required training connections.

Both forms count toward eight evidence records and the 256-position packet cap.
Individual maxima need not fit simultaneously.

## 6. Return limits

| Outcome | Positions |
|---|---:|
| Assessed Noul | 14 |
| Assessed Score | 25 |
| Assessed Choice | 18+A_i |
| Non-assessed Noul/Choice | 18+R |
| Non-assessed Score | 29+R |

A_i is complete selected-candidate attribute cost; require A_i<=46 for every
selectable entry. R includes ALL extra reason attachment descriptors, key slots,
bodies, refs/presented content. Require R<=32 aggregate and at most eight
attachments. The reason descriptor/four semantic vectors are already in framing.

Thus Choice assessed <=64, Noul/Choice non-assessed <=50, Score non-assessed <=61.
EVALUATE uses bounded parallel reason heads, not another hidden decoding loop.

## 7. Admission envelope

    R_max(Noul) = 50
    R_max(Score) = 61
    R_max(Choice) = max(50, 18 + max_i(A_i))

Choice has 1-16 entries. Before dispatch require packet<=256, every A_i<=46,
and remaining B >= packet_length+R_max under [control-flow v1](control-flow-v1.md),
which emits/reserves zero extra consumption ticks.
Remaining context also preserves remaining A/H. Preparation is already debited,
not charged twice. Reserve upper bounds, debit actual use/release unused capacity.
Check all statuses/candidates, never just a predicted winner.

This is the POSITION envelope. Full admission also follows the recorded
[resource contract](position-limits-and-budget-accounting.md#admission-and-reservation):
hold two completion Refs, dedicated sealing Refs (one snapshot, plus one bank
for Choice), and 2+min(32,|K minus request_handle_table|) completion handles.
Require total new Refs<=32 and local handles<=256 independently of positions.
Reason attachments are inline or refer to frozen existing catalogue K; they
cannot mint extra objects. Do not count the existing completion hold twice.

## 8. Printer calculation

With literal UTF-8 name/format: Pine (four bytes) costs 4 positions; self-adhesive
shipping labels (29 bytes) costs 7; int64 rate 120 costs 4; boolean available
costs 4. Attributes total 19, candidate 24 and assessed return 37. If all three
have that attribute length, reserve 50 for a possibly longer non-assessed result.
Request cost is 85+actual_evidence_cost. These are exact costs for those encodings,
not measured source-token counts for a source-span variant.

## 9. Guarantee

With the full resource contract, this guarantees protocol capacity for every
structurally legal return after admission. The position formula alone is not an
object-allocation guarantee. It cannot guarantee correct values, valid generated bytes,
adequate explanations or correct interpretation. Invalid output follows execution
failure, never truncation or relabelling as semantic unknown. Model configuration
must instantiate the interfaces; GPU/numeric/semantic checks remain required.
