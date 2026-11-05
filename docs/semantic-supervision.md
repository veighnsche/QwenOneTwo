# Semantic supervision without hidden outcome labels

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

This is an empirical supervision strategy, not a proof that writer outputs
contain zero information about an answer.

Owner: allowed semantic inputs, frozen reference targets, writer/receiver
alignment and semantic interventions. Follow the
[specification](architecture-and-training-plan.md),
[codec](latent-packet-codec.md), [writer](latent-writer-schedule.md) and
[data/readiness contract](training-data-and-readiness.md).

The [training-input rules v1](training-input-rules-v1.md) record F7's precise
per-role/branch whitelist, permitted outcome-bearing Reason supervision, actual-
status mismatch masks and teacher-forcing exposure. They specialize references
to answer blindness below: request targets exclude annotator-created current
outcomes; post-decision Reason targets may state justified non-assessed outcomes.

## 1. Allowed semantic content

Start with reviewed canonical annotations, separate from ordinary human inputs
and reference outcomes. Whitelist inputs to the semantic reference encoder:

| Record | Allowed | Excluded |
|---|---|---|
| Question | Proposition, criterion, qualifications and negation | Expected verdict and case facts that belong in evidence |
| Alternative | That entry's own supplied attributes | Winner/rank labels or unsupported suitability tags |
| Rubric | Dimension, conditions, direction and anchors | Current target's assessment or desired scalar |
| Target | Neutral subject/referent identity | Derived eligibility/severity/assessment summaries |
| Attribute key | Property and unit meaning | Current value or derived assessment |
| Reason | Justified non-assessed status and material missing/conflicting facts or constraints | Fabricated evidence or current assessed truth/winner/score |

For workshop access, ask whether the bound person currently meets the supplied
requirements; keep the expired badge in evidence. Pine's rate and availability
are legitimate candidate facts even though they help determine selection. Remove
unsupported outcome labels, not useful facts. Rubric anchors define a scale;
they do not say where the current target falls on it.

## 2. Answer-blind reference encoder

Use the pinned frozen Qwen backbone to encode the approved annotation, then
learned role-specific pooling queries to obtain eight question slots, four
alternative slots or two slots per rubric field. The encoder sees only its
approved role-specific record contents. Request targets exclude annotator-created
current verdicts, selected candidates, final answers and target scores. Reason
annotations may explicitly describe a justified non-assessed status and support.
Exclude opaque run IDs and winner-correlated IDs.

These are offline GPU component exercises. Reuse the same frozen base at
different times; no extra full model is required resident or deployed. Their
contexts are separate labelled training examples, not a second prefill inserted
inside actual single-cache inference or whole-flow demonstrations.

## 3. Establish meaning in the reference slots

Train a reconstruction decoder sharing the frozen backbone with a small
training-only reconstruction adapter. Give it only latent slots and field roles,
then reconstruct the canonical annotation. The source annotation must not be
decoder input. Do not optimize reference preparation with decision-outcome
objectives. Reconstructing a permitted Reason's status is semantic supervision.

Teach and test contrasts such as all versus any requirements, eligible versus
ineligible, any feasible versus fastest feasible candidate, severity versus
occurrence probability, and reversed scale direction. Swapping/removing slots
must change or break recovery of distinguishing content; familiar template
completion alone is insufficient.

Once reference encoder and decoder pass development checks, freeze/version
them. Recompute/cached targets must be deterministic for a fixed canonical
record. They must not drift alongside writer and Qev. This artifact defines a
semantic target basis, not proof of arbitrary Qev interoperability.

## 4. Train the native writer against fixed targets

The writer sees the ordinary human prompt and predicts its own semantic slots.
Compare them against frozen targets derived from the separate annotation:

| Objective | Purpose |
|---|---|
| Slot alignment | Bring each produced field toward its fixed reference representation |
| Reconstruction through frozen decoder | Preserve intended question, alternative and rubric content |
| Contrastive discrimination | Separate close but materially different meanings |

A starting alignment is mean squared distance between normalized predicted and
reference slots. Equivalent paraphrases use the same canonical target. Keep
teacher targets detached/frozen; predicted writer tensors remain connected.
Reconstruction/alignment update native R and writer components. Frozen decoder
weights do not mean no-grad computation through its predicted latent inputs.

Exact source references, values and attributes receive independent fidelity
supervision. Semantic agreement cannot excuse the wrong case or source span.
Readiness depends on behavior as well as reconstruction and vector similarity.

Use the same canonical normalized semantic payload for alignment, frozen-decoder
input and receiver presentation. Normalizing only the loss while exposing raw
payload magnitude would leave a direct unpenalized channel. The
[model profile](model-configuration.md) fixes normalizer/scale and independent
reference/live codec storage; this does not establish formal
absence of all hidden channels or normalize the numeric result itself.

## 5. Receiver grounding before writer dependence

First train the receiver using reference packets and separately supplied
evidence. One fixed workshop eligibility question must support true with an
active badge, false with an expired badge and Insufficient evidence with badge
status omitted, assuming completed orientation in each case.

Then progressively introduce actual writer outputs, retaining reference-packet
and semantic-alignment exercises. This anchors receiver meaning instead of
allowing writer/receiver co-adaptation to an arbitrary private verdict code.
Actual interleaved-cache execution still requires training and verification.

## 6. Record-specific invariances

Questions: keep meaning fixed across changed evidence and require the assessment
to follow evidence; keep evidence fixed while changing polarity/criterion and
require assessment to follow the new question.

Alternatives: keep an entry's facts fixed while changing competitors or the
criterion. Its representation continues to describe itself while feasibility
or selection may change. Pine at 120/hour qualifies for 100/hour but not 130/hour;
do not encode either outcome as an intrinsic property of Pine.

Rubrics: keep a scale fixed across different targets/outcomes. Different issue
evidence may change the value; a changed direction/anchor creates a new rubric
version. Desired score must not change rubric formation. Conditional severity
remains distinct from occurrence probability when frequency changes.

Build valid new test snapshots/bindings for interventions; never accept stale
or foreign references by bypassing immutable request validation. Keep related
variants within the same split and randomize incidental identities/order.

## 7. Isolate packet semantics from full-history shortcuts

The shared history already contains the human prompt. Qev might answer from
that history while ignoring the question. Require both actual whole-flow
single-cache tests and isolated component cases where only the semantic packet
and its declared evidence are available. Do not count isolated exercises as
proof of free invocation or mutate serving history to obtain their result.

Likewise, a reconstructed gloss is supporting evidence, not sufficient proof.
The receiver must behave appropriately under question, evidence, candidate
and rubric interventions, including missing/conflicting evidence.

The recorded [clean-test contract](clean-test-construction-v1.md) specializes F3:
build every supplied slot/state/value/reference from the permitted view, freshly
encode allowed excerpts and reset all component state. Full-history tensors do
not become clean when visible text is removed. Report clean components,
production-tensor transfer and actual whole-flow tests separately. Construction
lineage/twins and matching GPU fixtures remain unimplemented/unvalidated.

## 8. Preserve semantic constraints during joint training

Keep semantic losses alongside operation/status/value, reference/attribute,
native-reading, continuation and replay losses. Do not discard semantic
objectives once final answers look correct. The codec must not add an untyped
current-native-hidden-state payload bypassing the defined semantic fields.

Preserve continuous task-result gradients and native R updates. Hard gates,
status choices and integer pointers need separate supervision, not claimed
ordinary derivatives through argmax. The fixed reference encoder/decoder stay
frozen while runtime writer, receiver and native reader learn. The
[loss configuration](loss-configuration-and-training-schedule.md) records
starting weights, reference-validity masks and forcing/joint schedules.

## 9. Meaning and limits of no leakage

Enforce the role-specific whitelist: request targets exclude current reference
outcomes, while Reason may carry a justified non-assessed outcome. Verify matching to
stable semantic targets, recovery of material distinctions and transfer across
different outcomes/contexts. Do not treat the absence of an easy detectable
shortcut as a formal impossibility of hidden channels.

Qwen's writer has already seen evidence. Tiny deviations can carry information,
and legitimate candidate facts naturally predict outcomes. Frozen targets,
reconstruction, contrastive training and interventions reduce/expose shortcuts;
they do not prove mathematically zero answer information in every latent output.

The target is reusable, independently inspectable semantics that the receiver
follows under controlled changes, rather than merely a correct final verdict.
All model encoding, teacher work and verification remain on the selected GPU.
