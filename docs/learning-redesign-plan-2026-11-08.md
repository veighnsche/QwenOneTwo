# Learning redesign implementation plan — 2026-11-08

**Status: planning only.** The user requested a concrete implementation plan,
reusing working infrastructure and existing data, with W frozen and latent-only
communication. No agents, data generation or training are authorized by this
plan. No new model execution or measurements were performed to write it.

The proposal below changes experimental learning paths, not the current serving
default. Existing contracts remain authoritative until their named owners are
versioned in the corresponding implementation slice. Historical failures remain
failures. A passing implementation check is not a learned capability certificate.

## Goal and architectural decisions

Establish useful Noul, Choice and Score behavior, then a connected native
request/evaluate/return/answer demonstration, before optimizing compression or
expanding training. Use existing reviewed examples and their actual labels.

1. Preserve the shared frozen W, R/D mode separation, runtime, registry, exact
   values/bindings, continuous startup score/S-curve, cancellation, single
   prefill/cache lineage and qualified v3 arithmetic. Ordinary model work is
   3090-only: `GPU-225509c3-d702-6c27-de03-76b5e99b33b1`; no CPU fallback.
2. For **offline component inputs**, start with stable frozen-W source features
   and an explicit live consumer. Remove the changing-D/no-gradient source
   boundary from this candidate. Keep the old path as a named control.
3. Test a **richer ordered role-state packet**, without the separately fitted
   reference pooler, against the current pooled-role interface. A small live
   projection can adapt those states; no new translator model or reconstruction
   training stage is required. The existing reference artifact stays immutable.
4. For **native episodes**, source states come from the actual prompt prefill;
   question/rubric vectors come from the native writer. Never run the offline
   encoder inside a live job. These are different input regimes and need an
   explicit transfer check; success on prepared components is insufficient.
5. Train status and values together when learning is later authorized. Preserve
   Choice with replay/retention checks and distinct owner accounting. Avoid
   introducing another auxiliary loss in the first comparison.
6. Keep exact numeric atoms and existing Score intervals. There is no calculator,
   formula-specific runtime, computed answer attribute, threshold adjustment or
   tolerance relaxation in this plan. Exact arithmetic support would require a
   separate product/architecture decision if the measured neural path fails.

## Evidence motivating the changes

- `training/clean.py:CleanEncoder._states` uses current D under `no_grad`.
  `ReceiverProducer` refreshes those features as D changes. Frozen reference
  payloads are cached independently. This is a real gradient boundary, not a
  demonstrated unique cause of failure.
- The [simpler-input experiment](simple-latent-results-2026-11-07.md) already
  rejected ordered means with the frozen receiver: Choice 40/40 to 11/40. It
  retained fixed role-slot compression. Do not rerun it or call a richer,
  jointly adapted interface a proven improvement.
- The [quantity experiment](quantity-receiver-learning-results-2026-11-08.md)
  learned training values but damaged Score status and failed transfer. It
  intentionally omitted status/retention losses. INT64 did not solve reading
  quantities or applying operators.
- The [semantic audit](training-x-semantic-audit-2026-11-07.md) records narrow
  assessed receiver coverage: positive AND for Noul; ordinary completion with
  denominators 20/100 for Score. Development also changes operators, ranges and
  reference wording. The actual reference fitting catalogue had 3,376 views;
  the historical count of five annotations per role is not its full exposure.
- `training/binding.py` currently requires exact canonical reference payloads
  and complete grounded evidence to bind request labels. `producer.py` masks
  unsupported continuation losses. Replacing a pooler does not remove this
  supervision problem.

## Infrastructure to reuse

| Existing implementation | Reuse / narrowly scoped change |
|---|---|
| `codec/backbone.py`, `recurrence.py`, `precision.py`, `finite.py` | Preserve qualified v3 computation; reuse measured execution optimizations without adding new ones |
| `codec/session.py`, `runtime.py`, `records.py`, `render.py`, `writer.py` | Keep one causal job, immutable connected tensors, admission, real returns and cancellation; version only richer semantic payload handling |
| `training/release.py`, `manifest.py`, `inputs.py`, `component_producer.py` | Keep reviewed releases, permitted views, provenance and source mappings |
| `training/engine.py`, `episodes.py`, `losses.py`, `checkpoint.py` | Reuse eligible-example accumulation, unique owners, actual-status masks, export/resume and RNG restoration |
| `training/learning_report.py`, `behavior_qualification.py` | Extend reports with coverage and joint behavior; preserve original graders |
| `training/producer.py`, `binding.py` | Extend native input/target contracts explicitly; retain conservative masking |
| Existing local checkpoints, captures and receipts | Reuse when input/profile/weight identities match; never relabel stale tensors as new-path features |

No new orchestration framework, workers, service, frontend or deployment. Keep
The Operating Table and live services unchanged. Use one process, one CUDA
context and one resident W for sequential comparisons. Parallel agent work is
excluded.

## Slice RD-01 — pin data coverage, comparisons and reporting

**Implement:** extend `training/architecture_audit.py`, the existing semantic
audit and learning report with a versioned metadata-only comparison manifest.
Reuse their existing results rather than rebuilding already measured facts.

- Inventory existing receiver, writer, reader, interleaved and direct/replay
  records by actual permitted input, operation, status, label polarity, numeric
  range, rubric direction, reviewed operator and family. Classify operators only
  from existing structured/reviewed meanings; ambiguous prose stays unclassified.
- Record three overlapping descriptive axes: supported operations/ranges on
  unseen families; changed wording/bindings; unseen operations/ranges. Do not
  call a cohort a single-variable test when multiple axes change.
- Preserve the original train/development split and family membership. Repeatedly
  inspected development is development, not a new blind test. Do not open locked
  data or manufacture a fresh holdout from previously fitted training families.
- Select panels deterministically from metadata, before candidate predictions.
  Keep full E/X results beside subgroup results, failures in denominators, and
  assessed results separate from abstentions. Score reports separate endpoints,
  interior values, interval distance and exact delivered success.
- Map each reusable native target to the exact reviewed request/return conditions
  under which it is valid. Missing source pointers, Reason labels or continuation
  support are explicit gaps, not invented annotations.

**Acceptance:** every selected record resolves to original bytes/review identity;
no train/development family overlap; deterministic manifests and counts; all
existing receiver rows accounted for; known unsupported operator/range cells
explicit; no imports executing model workloads. Tests cover overlap, ambiguous
coverage, hidden-label leakage and stale identities.

**Evidence/commit:** portable coverage manifest and a concise report, with raw
record lists under local `data/`. No feature extraction or model run.

## Slice RD-02 — extend one runner for these comparisons

**Implement:** extend `scripts/train_receiver_probe.py` and its receiver/report
helpers with explicit input-profile selection and metadata-only prepare mode.
Use small profile-specific construction functions; do not build a plugin system
or copy another full training loop. Existing scripts remain historical/replayable.

One manifest binds release/panel, input profile, component/native regime, initial
checkpoint, frozen dependencies, trainable owners, objectives, exposures and
required evidence. Preparation has no side effect that launches fitting. The
existing `Window` and checkpoint code remain the optimizer/resume implementation.
Checkpoint identities distinguish old reference-based and new direct-state
profiles; no fabricated reference qualification or placeholder evidence hashes.

**Acceptance:** prepare works with model imports blocked; execution is explicit;
wrong GPU/profile/release/cache/checkpoint is rejected; failed attempts remain
scheduled exposures; receipts distinguish preparation, offline encodings, real
Sessions and actual optimizer updates. Reuse existing optimizer/window tests.

**Evidence/commit:** deterministic plan output and rejection tests. Timing fields
reuse CPU/dispatch instrumentation; no fresh throughput project or elapsed-time
cutoff. Cache only dependencies whose complete fingerprints remain frozen.

## Slice RD-03 — stable component source features

**Implement:** factor source encoding in `training/clean.py` into explicit named
policies. Add `frozen-w-source-v1`: zero adapter contribution, immutable W,
qualified v3 computation, permitted source bytes and unchanged byte/token
coordinates. Keep current D-source behavior available under its original name.
Select the no-adapter computation explicitly per forward; do not merge adapters,
replace weights or toggle shared training ownership to obtain a frozen encoding.
Initially retain the old pooled role payloads, so this slice changes only source
feature ownership. Its source cache key includes W/tokenizer/template/precision,
permitted content and source mapping; it never depends on a mutable D hash.

Native prefill states retain their actual mode and graph. They must not be
re-encoded with this offline policy or passed off as identical component inputs.

**Acceptance:** source/profile lineage is complete; byte bindings are unchanged;
changing a live D fixture cannot change frozen-W source features; consumer
gradients are connected while W has no gradients; repeated valid frozen cache
lookups reproduce uncached features. These tensor checks require the 3090 and
are executed in RD-07, not on CPU. Static policy/leakage tests run with this slice.

**Evidence/commit:** policy/cache implementation and tests, with GPU checks marked
pending until run. No optimizer step or learned improvement claim.

## Slice RD-04 — richer role-state packet, opt-in

**Implement in two commits:** first version the layout/admission contract, then
add rendering/encoding support. Work in `codec/records.py`, `render.py`,
`neural.py`, `writer.py` and the clean packet builders.

- `direct-role-states-v1` carries the ordered content-token hidden states for
  each permitted semantic role, with explicit role, owner, length and order.
  Extract only verified content spans; template context is fingerprinted but
  template positions are not semantic payload positions. No learned reference
  pooler or fixed bin means compress these sequences.
- Use a single live FP32 residual projection shared across roles, initialized
  to identity, followed by existing normalization/wire framing. Preserve exact
  attributes, units and source provenance. All new weights have one named owner.
- Start with at most 32 vectors per semantic field and the existing 256-position
  request ceiling. These are structural bounds, not permission to crop. Report
  every overlength field/request and affected operation/family; fail admission
  before a partial model request. Do not quietly shrink the evaluated panel or
  raise context/return allowances. If coverage cannot support a planned test,
  revise the layout explicitly before that test.
- Version role offsets/lengths, rubric subfields, candidate ends, return echoes,
  native writer length heads and emission capacity together. Existing source-span,
  atom and immutable-memory machinery remains shared. Reserve the full return
  cost, including question/rubric echoes and the largest possible selected entry;
  the old 50/61-position return assumptions cannot be reused blindly.
- Offline role states come only from existing answer-blind annotations under
  frozen W. Native requests contain writer-produced vectors or actual source
  references. Role annotation text is never appended to a live Qev request.

**Acceptance:** old-profile fixtures remain exact; new sequence order/coverage,
roundtrip identity, all-status reservations, overlength rejection and snapshot
immutability pass. Exact attributes roundtrip unchanged. All presented vectors
are charged. No implicit profile conversion or textual invocation is permitted.
Native emission has a defined shape/length contract before learning is proposed.

**Owners to update:** latent representation, packet codec, model configuration,
writer schedule, memory/accounting and input-rule specifications. Changes are
experimental profile additions, not blanket replacement of the qualified path.

**Evidence/commit:** versioned layout, cost examples and static tests; targeted
GPU forward/gradient acceptance in RD-07. Larger payloads are an information
baseline, not a promised speed improvement.

## Slice RD-05 — joint objectives and honest binding/eligibility

**Implement:** reuse `decision_losses`, `Eligibility` and `Window`. The first
receiver comparison uses existing status, Noul, Choice, Score, applicable
feasibility and supported Reason/structure objectives. Keep existing coefficients
and eligible-example means initially; do not repeat the value-only objective or
add count regression. Development labels never control sampling or coefficients.

- A prepared receiver request built from the exact reviewed permitted view has
  explicit component provenance. The new profile can train such components
  without asserting that a separate reference has passed semantic qualification.
  Create an explicit experimental admission path; do not set old qualification
  flags to true or disable the diagnostic-reference guard globally.
- Status loss remains eligible on supported cases regardless of predicted
  status. Value targets follow reviewed Assessed eligibility, preserving existing
  actual-status rules for Reason. Freeze W, R, startup row and reference during
  receiver learning. Train D, the selected live input consumer and the existing
  receiver owners, with exact owner lists in each run manifest.
  Changing D can change startup predictions even with the budget row frozen;
  report that retention separately and preserve the continuous mapping exactly.
- Bind labels to the actual snapshot and source/role/candidate identities.
  Arbitrary writer-generated vectors do not acquire semantic validity from a
  matching record ID, vector proximity or the expected answer.
- Preserve the Reason exception: approved outcome-bearing Reason targets can
  supervise a supported post-forward Reason/isolated-reader branch. They never
  enter request/source features, caches, field lengths or routing decisions.
  Missing new-profile Reason support is masked and reported, never zero-filled
  and counted as a qualified explanation.
- Report eligible counts, actual owner gradients, status confusion, per-operation
  raw values and delivered joint success. In particular, an aggregate gain from
  abstentions cannot hide assessed Noul failure or Score status collapse.

**Acceptance:** negative tests reject foreign/divergent snapshots, leaked outcomes,
unsupported Reason and dev training. Controlled gradients reach only declared
owners; private-head freezing is never reported as shared-feature protection.
GPU VJP/accumulation and resume checks are performed in RD-07.

**Evidence/commit:** objective/owner/binding manifest and mask tests. Update module,
semantic-supervision, input-rule and training-profile contracts for this named
experimental path; preserve old evidence and masks outside it.

## Slice RD-06 — native connected path and supervision dry run

**Implement:** extend `EpisodeProducer`, `request_targets`, writer emission and
return binding to the new profile before spending another receiver fit.

The live path stays: original prompt prefill -> native latent writer -> actual
Qev evaluation -> bound latent return -> native continuation. Use prefill source
states and the same functional cache. No offline encoder, reference annotations,
gold result, source replay or second prompt inside the live Session.

Existing frozen-W annotation states may supply **training-only** writer alignment
targets. They are not a runtime semantic oracle or an exact-reconstruction gate.
Discrete role lengths, pointers and actions use existing supported supervision;
no derivative through argmax is claimed. Continuous paths remain connected even
through frozen receiver weights.

Keep three explicit exposure types: prepared receiver components; isolated
reader exercises with approved supplied returns; native episodes consuming actual
returns. Only the last can support connected-flow evidence. Guided schedules
must record every forced action, pointer, field or budget override.

**Acceptance before any native fit:**

- Metadata audit identifies which existing writer/interleaved/reader records have
  supported targets. Arbitrary predicted requests/returns retain their masks.
  The dry run reports eligible continuation and formation examples by operation
  and the exact reason for each zero. A zero-loss path is not a trainable demo.
- An actual-return trace reaches native continuation with one prefill/lineage,
  exact binding/accounting and no invocation text. A separate gradient fixture
  proves native loss -> continuous return -> receiver -> writer connectivity;
  fixture forcing is disclosed and establishes no semantic skill.
- Incorrect/unsupported returns do not silently receive incompatible approved
  continuations. Existing correction supervision is used only when its binding
  is valid. Missing correction labels or semantic binding remains a blocker for
  that branch; no new response/annotation is generated to close it.
- Return consumption alone is not result understanding. Prepare evaluation of
  source-backed final values/identities and existing input-change pairs, plus
  existing supported misleading/withheld-return reader cases, separately labelled.

**Evidence/commit:** native adapter/eligibility changes and a dry-run report. GPU
traces/gradients run in RD-07. If existing supervision cannot bridge predicted
requests, report that concrete limitation before proposing native training;
independent receiver/profile work can continue.

## Slice RD-07 — targeted 3090 qualification, no training

**Run only when implementation/execution is requested.** Use existing data and
declared diagnostic fixtures, W/reference frozen, no optimizer steps.

- Old-profile anchors reproduce saved outputs, gradients where available,
  immutable bindings and full ledgers when dependencies match.
- New source/projection/payload paths compare against explicit FP32/FP64
  calculations using the existing pointwise/RMS/cosine bounds. Check finite
  values, connected gradients and zero frozen-W/reference gradients.
- Exercise Noul/Choice/Score, non-assessed returns, variable-length fields,
  return reservations, cancellation and envelope boundaries. Include affected
  long-context traces through 1024 positions where supported. Do not infer new
  profile qualification solely from old-profile matrix passes.
- Verify all 320 W tensors, physical tying, reference identity and frozen owners.
  Restore new-profile model/optimizer/RNG/sampler state exactly at complete-window
  boundaries. No optimizer update is needed to test serialization; nonzero Adam
  moment loading can use disclosed deterministic fixture state, not a fitted run.
- Reuse unchanged 324-case and 1024 matrix evidence for unchanged kernels. Rerun
  relevant numeric coverage only when a changed dependency requires it. Do not
  reopen the failed original BF16-transformer arithmetic path.

**Acceptance:** every applicable invariant passes with unchanged tolerances;
failed/unsupported cases remain visible; report clearly separates legacy reuse,
new checks and incomplete coverage. If the larger packet does not fit the
existing envelope, revise its contract instead of silently truncating or
declaring the old numeric certificate sufficient.

**Evidence/commit:** commands, source/profile/checkpoint hashes, exact results,
position counts and execution failures. This qualifies machinery only.

## Slice RD-08 — matched receiver learning, future authorization required

**Proposed first experiment, not authorized or launched:** compare two arms with
the same frozen-W source policy and same initial receiver/consumer weights,
objectives, seed, optimizer state, data and exposure. Control uses existing
pooled role slots; candidate uses direct role states. This isolates the role
interface after source ownership is stabilized. The original changing-D path
remains historical context, not a falsely matched control.

Use all 1,500 existing receiver records if RD-01/04 show both arms can represent
them without cropping. Proposed exposure is four passes per arm: 6,000 examples,
750 eight-example windows each. This is a finite comparison protocol, not an
elapsed-time/cost cap or permission for automatic extra passes. Any unsupported
records require an explicit common-support manifest and visible coverage loss
before execution; never choose support using model performance. Same named
parameters receive identical initialization/moments. New profile-only parameters
have disclosed initialization/fresh moments; different layouts are not claimed
to produce identical hidden states or parameter counts.

Measure initial/final training behavior and the existing full E/X development
panels plus RD-01 subgroup reports. Both arms finish the fixed exposure before
development evaluation. No best-checkpoint selection, development fitting or
locked-set access. Compare exact statuses and values together. Report all
supported Reason metrics separately from mere structural validity.

**Acceptance for a useful receiver improvement:** on the predeclared familiar-task
development cohorts, Noul delivered behavior improves over control and its truth
prediction beats the training-defined constant-label baseline; Score improves
interval distance and delivers correct interior values under unchanged grading;
status success does not regress within each operation/status cell. Preserve every
previously passing Choice case in the designated retention panel. Report the full
case-level win/loss table and both-sides-correct intervention counts. Missing
cohorts, failed checks or weaker subsets prevent the corresponding claim.

These conditions establish a useful candidate, not general competence or MVP
acceptance. Existing broad semantic qualification remains separately reported.
If both arms fail, the comparison still completes with actual results; it does
not trigger more passes or additional architecture searches. If only unseen
operators/ranges fail, retain that coverage limitation instead of declaring them
trained or weakening the grader.

## Slice RD-09 — connected native learning and demonstration, later authorization

Proceed only after RD-06 exposes real eligible supervision and RD-08 establishes
usable receiver behavior. Freeze the accepted receiver consumer/D and reference
for the first native fit; train R and its declared writer/reader owners. Frozen
receiver weights must still transmit gradients to continuous writer outputs.
Any parameter shared with receiver computation stays frozen in this first fit;
an owner name alone is not evidence of isolation. Verify receiver outputs on
identical packets before and after native updates.

Use the existing writer, reader, interleaved and direct/replay records. Keep
isolated supplied-return reader learning separate from actual-return episodes.
The exact finite exposure manifest follows RD-01's eligible inventory and must
be fixed before a later run; this plan does not authorize an unspecified training
job. Begin with supported guided episodes, then explicitly scheduled reduction
of forcing and separate free invocation evaluation. No newly generated curricula,
annotations, semantic grader calls or automatic retries.

**Acceptance for the connected demonstration:** on predeclared existing held-out
families, all selected supported demonstration cases must pass their original
semantic/value/binding checks, including both members of selected change pairs.
The manifest must include Noul true and false, Choice identity/order changes,
interior Score values, available abstention cases, actual-result native use and
legitimate direct/no-call cases. Missing examples mean missing qualification;
do not replace these checks with handpicked successes. Report full development
results alongside the demonstration, including unsupported operator/range cells.

Learned-flow claims require unforced operation choice, writer completion and
actual returns, with no budget override or hidden call scaffolding. Native final
answers must preserve the bound values/identities and use permitted evidence;
exact wording reconstruction is not required. Score/status failures, ignored or
misbound results, unsupported Reason meaning or no-call collapse keep those
behaviors unqualified. Cancellation, budgets and one causal lineage remain exact.

No automatic promotion/deployment follows. Only after this demonstration should
compression or larger training be considered; neither is scheduled here.

## Order, evidence and known limits

Order: **RD-01 -> RD-02 -> RD-03 -> RD-04 -> RD-05 -> RD-06 -> RD-07**,
then separately authorized **RD-08**, then **RD-09**. Each implementation slice
is a coherent main-branch commit with relevant checks and a handoff entry. Split
layout/implementation or CPU/GPU evidence commits when useful. Report acceptance
as pending/pass/fail/unsupported, with actual commands/results; never create a
passing report merely because an interface was implemented.

Immediate next work, if implementation is requested: **RD-01**, a metadata-only
coverage and native-supervision manifest. Existing receiver optimization ideas
are superseded as the proposed work order, not erased as evidence.

Known limits under existing-data-only scope:

- No assessed receiver examples for some requested operator/range changes have
  been found. Other existing categories may be reused only with exact permitted
  inputs and already reviewed labels; RD-01 must establish eligibility first.
- Rich role sequences may exceed current envelopes. Complete coverage and native
  emit/return capacity must be measured before fitting.
- Stable offline W features and live one-prefill features differ. Component
  success cannot certify native transfer.
- Semantic binding and valid correction continuations may remain insufficient
  for free writer outputs. Masking is preserved; metadata identity is not meaning.
- Exact Score, Reason meaning, learned invocation and native result use remain
  unqualified. There is no guarantee that this redesign resolves them.

Plan validation: existing implementation and measured reports inspected; **197
local Markdown link targets across three files resolve**, and whitespace checks
pass. No model, GPU probe, training,
agent, new data/review or service operation is part of this documentation slice.
