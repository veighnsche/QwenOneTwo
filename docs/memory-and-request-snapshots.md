# Memory references and immutable request snapshots

Implementation update: The append-only registry and immutable binding path now has a scoped implementation in
`qwen_one_two/codec/`. See the [implementation](full-codec-implementation.md)
and [measured qualification](codec-qualification-results-2026-11-05.md).
The original design status below is historical; training and frontend
expansion remain excluded, and expanded mixed-precision qualification
is not passed. The recorded normative rules/tolerances remain in force.


Status: recorded design choice, not implemented or validated capability. No
coding, training, service changes or deployment are authorized by this document.
This records the memory proposal discussed after the latent representation.

Owner: references, immutable snapshots and one-time consumption. Use the
[implementation specification](architecture-and-training-plan.md) for dispatch
timing and the [handoff](implementation-handoff.md) for remaining codec work.

## Core contract

Use an append-only store, typed versioned references, one sealed snapshot per
request, exact return binding, and one-time consumption. Qwen learns which
objects to reference; the runtime preserves the chosen identities.

Conceptually a reference contains run identity, object type, object identity and
version. Object types include sources, cases/issues, questions, alternatives,
candidate banks, rubrics and results. Logical object identities are never
reassigned to unrelated objects. An edit creates a new version and preserves
the old one. Sealed requests never contain “fetch the latest version.”

## Learned reference selection

Qwen receives a neural view of available memory entries, including their types,
semantic representations and relevant contents. A learned pointer selects an
entry; the runtime resolves that local position to its exact reference. Qwen
does not have to invent arbitrary identifiers or predict UUIDs.

Source references point into immutable source material and source-token spans.
These offsets belong to the original source, not the growing inference cache.
Appending latent positions must not shift the meaning of a source reference.

The runtime checks reference existence, allowed type and run membership. It does
not decide which customer, policy, alternative or rubric is semantically right.
That remains a learned binding problem requiring supervision and verification.

## Sealing a request

The learned operation-intent gate starts a native draft; it does not dispatch
an incomplete request or invoke a second semantic gate at completion. When
the learned writer signals complete arguments, the runtime validates structure
and seals the snapshot, pinning:

| Field | Fixed meaning |
|---|---|
| Request identity and operation | This particular invocation |
| Question reference | Exact question version and vector values |
| Target and evidence references | Exact cases, issues, sources and prior results |
| Candidate-bank reference | Exact entry versions and their ordering |
| Rubric reference | Exact criteria, anchors, numeric range and direction |
| Context frontier | Where this request began in the inference history |
| Model/protocol versions | The execution bundle producing and interpreting it |

Referenced objects must themselves be immutable. A read-only wrapper around
mutable vector storage is not sufficient: later buffer reuse must not mutate
an old question or alternative. Use owned values or appropriate copy-on-write
semantics. A snapshot can pin existing immutable objects without copying the
whole store; changed versions need new storage.

## Return binding

The runtime attaches the current request identity to Qev's return. Qev need not
predict that identity. Validate the completion against the pending request and
run before delivering it to native Qwen.

**Noul:** the result refers to the exact proposition, case and criteria in the
snapshot. A later negated question is a different request; it cannot change an
old true result's meaning.

**Choice:** map the selected array position immediately through the frozen bank
to a stable entry reference. If an old bank begins Tuesday, Wednesday, Thursday,
selecting position zero refers to Tuesday. A later bank reordered Wednesday,
Tuesday, Thursday cannot reinterpret that return. Deliver the selected entry's
representation and recoverable contents to Qwen.

**Score:** a value remains attached to its target and rubric version. Reversing
a rubric's direction later does not reverse the historical value's meaning.
A new assessment requires a new request.

An old result may be a valid historical record but unsuitable for the current
task. Never silently rebind it to updated inputs. Relevance to a changed task
remains something the native reader must learn; identity bookkeeping is not
semantic understanding.

## Serial lifecycle and one-time consumption

For the first version use:

    draft → sealed → evaluating → validated return → consumed

Only one request is pending. Native generation pauses during Qev evaluation.
A completion is accepted only for that pending request and run. Cancellation
invalidates the pending invocation; late completions must not resume generation.
Duplicate delivery must not inject feedback twice. A retry receives a new
request identity.

Only unsealed structurally invalid drafts can be revised. Resource non-admission
ends that attempt; a later retry has a new identity. Count Ref allocations and
versions, including banks/results/reasons, separately from inline fields and
physical tensor copies. The mutable draft selects immutable object versions;
it does not mutate already stored record contents.

Semantic outcomes such as insufficient evidence still belong to the request.
Do not turn them into fabricated false, zero or selected alternatives. The
[typed result-status contract](result-statuses.md) records this next layer.

## Allocation holds and completion references

Follow the [resource-admission contract](position-limits-and-budget-accounting.md#admission-and-reservation).
New question/target/candidate/rubric versions each consume one allocation;
unchanged reuse consumes none. Source/evidence spans and attributes are inline.
Keep the draft bank mutable and publish one final bank for every Choice attempt
at sealing, plus one snapshot for every operation. Temporary validated builders
have no published Ref identity. Published allocations are never refunded.

Hold Result+Reason slots from attempt entry, plus the sealing slots. Writer
allocations before sealing are <=29 for Noul/Score and <=28 for Choice; after
sealing allocated+2<=32. Freeze compatible published return catalogue K at seal.
Reasons may create inline data but no additional Ref-bearing objects beyond
their one Reason. All attachment references resolve through K, not future or
self-supporting Result/Reason IDs. Pinning K neither presents all contents nor
isolates attention. Preserve exact bindings and required tensor gradients.

Local handle credits are separate: existing objects may require new handles
without new allocations. Hold 2+min(32,|K minus request_handle_table|), at most
34, with total<=256. No sidecar-only pre-interning or phantom object creation.
Use the B1 traversal for actual bindings. Non-admission, failure and cancellation
release only unused holds, preserving spent work/published records.

## Logical snapshots are not cache copies

The snapshot freezes what the request refers to, not a historical copy of
Qwen's attention and recurrent state. The working cache continues through the
decision and return. A context-frontier reference does not restore an earlier
hybrid-cache state or prove that Qev attends only to referenced evidence.

For the first version, new human input queues for the next inference job or
cancels the current job; it does not edit pending inputs in place. The
[cache/gradient strategy](cache-and-gradient-strategy.md) is recorded; its GPU
feasibility remains unverified.

## Learning and verification obligations

Train bindings with multiple similar cases, alternative reorderings, changed
rubrics and isolated result-reader exercises. Exact references prevent identity
mix-ups but do not prove the writer selected the correct evidence or the reader
understands it.

Verify that reordering memory preserves referenced meaning, new rubric versions
do not relabel old scores, cancellations and duplicates do not inject extra
feedback, and source references survive added latent positions.

During training, immutable record values must retain the gradient connections
required to train writer and reader. Immutability must not accidentally detach
the learned request/return path.
