# Clean test construction v1

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

Owner: information boundaries, clean tensor/materialization lineage and distinct
clean/transfer/whole-flow evaluation protocols. Follow the
[training-input rules](training-input-rules-v1.md), [semantic supervision](semantic-supervision.md),
[B1](neural-input-encoding-v1.md), [numeric profile](numeric-profile-v1.md),
[data operations](data-operations.md) and [specification](architecture-and-training-plan.md).
The [F8 promotion matrix](promotion-matrix-v1.md) separately records slices,
minimum counts, denominators and promotion; no passing artifact is implied.

## 1. Governing rule and test categories

Every tensor supplied to a clean test must be constructed exclusively from that
test's permitted inputs. Removing text after encoding does not remove information
it contributed. A cached state for 'orientation completed' can also encode an
earlier 'badge expired' fact. Moving it to a new context does not hide that fact.

| Category | Construction | Establishes |
|---|---|---|
| Clean component | Fresh state/representations built only from explicitly permitted inputs | Interpretation of supplied semantics, evidence or result |
| Production-tensor transfer | Actual contextual source states/live writer outputs transferred into component exercise | Whether actual produced representations work in that exercise |
| Whole flow | Original prompt, learned invocation, actual request/return and continuous cache | Whether the integrated system works |

Transfer fixtures may retain original prompt information. Never call them clean
withheld-evidence fixtures. Fresh component contexts are offline tests, not extra
prefills inside production; production retains one prompt prefill and one cache.

## 2. Declare the information boundary first

Each fixture declares task/criterion, permitted facts/attributed claims, semantic
records/exact attributes, any permitted prior/supplied result, withheld facts and
labels, and whether it tests writing, evaluation or result interpretation.

For receivers the expected current outcome is unavailable. For readers the bound
supplied outcome is intentionally available: interpreting it is the task, and
underlying evidence may be unavailable. Apply F7 per-role whitelists within the
view. Review labels against that view, never against withheld world truth.

## 3. Rebuild all potential information carriers

| Representation | Clean construction |
|---|---|
| Question/target/alternative/rubric/key slots | Frozen reference encoder on role-whitelisted canonical annotations only |
| Source-state evidence/source-string attributes | Re-encode only permitted excerpt in separate source preparation context |
| Exact typed values | Permitted facts with valid fixture-local provenance |
| Historical results | Explicitly permitted contents, including reasons/selected candidate contents |
| Supplied return | Approved clean question/target/rubric/candidates plus declared result |
| Reason slots/attachments | Approved status/support permitted in reader's view |
| Pointer keys/pooled summaries | Recompute from clean records/source states |
| Tags/handles/lengths/positions/counters | Derive from clean fixture independently of withheld facts/outcome labels |

Do not reuse full-information live writer slots as clean targets. A separately
labelled clean-writer exercise may generate slots from its allowed-input-only
prompt, but those are writer outputs under test, not frozen reference targets.
Detaching, cloning, normalizing or projecting a contaminated tensor does not
establish cleanliness. Remove withheld facts from canonical annotations before
encoding, not by masking a few dimensions afterwards.

## 4. Source encoding profile

For v1:

1. Materialize permitted excerpt as derived immutable source containing only it.
2. Tokenize afresh using pinned tokenizer/template.
3. Encode under evaluated checkpoint's W+D, matching production prefill mode.
4. Use fresh source-encoding state with one user message containing that excerpt,
   no case-specific instructions, other facts or outcome labels.
5. Extract source-body states and apply existing source projection.
6. Discard preparation state before constructing receiver/reader exercise.

Preserve verified byte/token mappings and existing span limits. Retokenization
can change boundaries: validate rather than copy old indices, crop or truncate.
Original source coordinates and parent hashes stay in external audit manifest.
Model-facing refs/offsets address the derived CLEAN source. No original preceding
text length, inherited position or full-source registry expansion can carry the
withheld fact. Derive excerpt selection/order from the declared permitted view.

## 5. Fresh component execution state

Reset attention K/V, recurrent/convolution state, position/cache counters, draft
feedback, pooled summaries, registries, pointer catalogues/retrieval and any
persistent execution buffer whose prior contents can affect forward execution.
The component receives only declared clean prefix/packet. A clean handle must
not resolve into the original full record. Keep full source/audit manifests
outside model-facing lookup paths.

Parameters remain the fixed evaluated checkpoint. Never fit adapters/reference
encoders or other parameters on locked cases during fixture preparation. Frozen
artifact encoding is a forward computation, not permission for test-set fitting.

## 6. Hidden-fact twins and visible-evidence controls

Use two underlying scenarios differing only in withheld fact. Example:

| Hidden scenario | Permitted receiver information | Clean label |
|---|---|---|
| Badge active | Orientation complete; access requires orientation and active badge; badge status withheld | Insufficient |
| Badge expired | Exactly same permitted information | Insufficient |

Complete model-facing inputs must match across twins: metadata, candidate
contents, positions and controls included. Hidden difference exists only in
external manifest. Valid fixture-relative bindings remain enforced; never accept
foreign IDs to force equality. Canonical local-handle presentation can match
while global run/request identities differ outside model inputs.

Then expose active badge for true, expired badge for false, neither for
Insufficient. Distinguish evidence intervention from swapping an expected label.
Apply analogous constructions to Choice/Score with view-grounded labels:
unknown availability is not unavailability; unsupported Score is not zero.

## 7. Clean reading and return interventions

For clean Noul reading supply clean question/neutral target and correctly bound
assessed return, withholding original evidence/reasoning. A separate fixture
supplies alternative return. Interpret the supplied result without inventing its
cause: false alone does not reveal that badge expired.

Choice includes selected entry's permitted contents. Score includes actual rubric
and scalar. Non-assessed Reason may contain F7-approved status/justification.
Use separate valid snapshots/results for interventions, never mutate immutable
results or bypass pending binding checks. Supplied results are declared component
inputs, not claimed outputs of actual free Qev evaluation.

Full-evidence contradictory-return tests are separate: recognize/reconsider
conflict instead of teaching universal trust or universal ignoring.

## 8. Provenance gate before behavioral scoring

Every supplied tensor records construction lineage: permitted bytes/canonical
annotation, encoder/profile, checkpoint, transformations and resulting artifact.
Require future checks that:

1. No model-facing dependency originates in withheld facts/current receiver labels.
2. Withheld-only changes leave clean input artifacts unchanged.
3. Allowed-fact changes update appropriate representations and reviewed labels.
4. Fact-order changes cannot import removed context.
5. Reference resolution cannot expand into forbidden content.
6. Original reasoning/draft/cache/attachments do not survive accidentally.

Exact records and cached input artifact identity are checked directly. Independent
floating-point recomputation uses A3 tolerances, not a new numeric allowance.
Report construction failures explicitly; they invalidate the clean claim and
cannot be silently dropped to improve semantic rates. F8 fixes denominators.

This establishes traceable construction, not absence of learned prior knowledge
or logical inference from legitimate evidence. Fresh synthetic cases and existing
family split rules help distinguish memorization. No passing tensor-provenance,
semantic, gradient or GPU test is claimed by recording this design.

## Implemented input identity revision — 2026-11-07

`CleanView.materialize` now emits `schema=clean-view-v2` and an explicit
`rubric_direction` (1 higher, 2 lower, null without a rubric). Component builders
reject unspecified/unknown rubric directions, and `CleanPackets.request` rejects
a renderer direction different from the hashed view. Direction-only changes
therefore change the input hash even when reference/source bytes are identical.
Model-free tests cover both directions, invalid types and request mismatches.

Old receipts retain their original hashes and source snapshots. They are not
silently accepted as v2: diagnostic reuse requires an explicit old/new mapping,
complete row/reference/source binding and unchanged-model replay evidence.
This changes audit identity and validation, not tensor arithmetic or packet ABI.
