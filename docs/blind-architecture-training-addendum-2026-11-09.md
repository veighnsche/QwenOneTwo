This addendum makes the proposed fixed vector space concrete. It is a design specification, not a claim that the required annotations, artifacts, or capabilities exist.

1. Human labels define meaning; a frozen encoder defines coordinates

Reviewers annotate understandable request fields: proposition, criterion, target, evidence scope, alternatives, rubric, conditions, and source references. They never approve individual vector coordinates.

A deterministic training-only serializer converts those reviewed fields into an explicitly typed sequence. For example, proposition polarity and criterion conditions occupy distinct fields rather than being inferred from an unlabeled paragraph. Exact names and quantities retain their source bindings.

A frozen reference encoder \(E\) maps that sequence into vector targets. My concrete choice is a separately instantiated, frozen copy of the foundation model, with Qev inactive, used only during authorized offline GPU extraction. It is an encoder, not a generative judge. Its ability to provide useful representations remains an assumption to test.

Each target slot contains:

- A type and field index.
- A contextual vector from \(E\).
- A frozen input-embedding component for the annotated token or span.
- Source handles and applicable scalar metadata.

The embedding component supports exact-content reconstruction checks; the contextual component supplies a learning target for field meaning. Neither component proves semantic preservation. Variable-length slots avoid claiming that an arbitrary request fits losslessly into a fixed small vector.

The serializer and encoder version define the reference space. They are frozen before dependent training. Any later encoder change creates a new interface version and requires fresh compatibility acceptance.

2. Ownership prevents circular targets and leakage

The reference encoder starts from the preserved pretrained base and receives no gradient. It is not trained to reproduce the native model’s requests, decisions, or answers.

The request encoder input contains only reviewed request meaning and facts accessible at dispatch. It excludes assessment labels, expected answers, withheld facts, future evidence, and reviewer explanations that reveal those facts. Assessment supervision lives in a separate training-only record section.

Trainable owners are explicit:

| Owner | Initialization | Training role |
|---|---|---|
| Reference encoder \(E\) | Preserved foundation checkpoint | Frozen target construction only |
| Input codec \(C_{\text{in}}\) | Deterministic dimension-preserving initialization where possible; otherwise declared random initialization | Maps reference vectors into decision-input embeddings |
| Auxiliary decoder/checking heads \(D\) | Declared random initialization | Recover annotated fields and exact contents from vectors |
| Aligned Qev derivative and decision heads | Original Qev plus separately stored initialized deltas | Learn reviewed assessment semantics |
| Result codec \(C_{\text{out}}\) | Declared initialized projection | Produces the accepted result-vector space |
| Native request/consumption extension \(N\) | Zero-effect adapter initialization plus declared initialized heads | Learns invocation, formation, and continuation |

The auxiliary decoder is training-only. It never supplies deployed semantic judgments or repairs requests. A frozen decoder used as a training loss must first pass held-out checks; its own predictions are not new labels.

Result targets may encode reviewed statuses, assessed values, and supported reasons because those are the decision outputs being supervised. They must not encode the expected final user answer. Reasons identify available evidence, missing requirements, or conflicts without inserting withheld facts.

All extraction and fitting would execute on the selected GPU. This addendum authorizes none of that execution.

3. Illustrative schema—not newly created training data

```text
record:
  provenance:
    record_id: <stable identifier>
    family_id: <split unit>
    review_version: <identifier>

  dispatch_snapshot:
    accessible_source_spans: <immutable span IDs and contents>
    fact_bindings: <reviewed facts with source references>
    bank: <ordered stable IDs and source-bound contents, if applicable>
    rubric: <version, conditions, anchors, direction, if applicable>

  reviewed_request:
    operation: <Noul | Choice | Score>
    semantic_fields: <reviewed proposition/criterion/target fields>
    evidence_refs: <references into dispatch_snapshot>
    exact_bindings: <identity/date/quantity references>
    field_presence_masks: <legitimate present/absent fields>

  assessment_supervision:             # excluded from request-encoder input
    status: <reviewed semantic status>
    assessed_value: <present only for Assessed>
    acceptable_choice_ids: <set, when applicable>
    reasons: <reviewed reason types and bound references>
    value_presence_mask: <boolean>

  native_supervision:
    invocation_action: <continue or operation>
    invocation_location: <reviewed trajectory location>
    continuation_constraints: <permitted facts and required behavior>
    matched_intervention_group: <optional identifier>
```

A compiler can produce vector targets from this schema only after reviewed records exist. The schema does not establish annotation availability or quality.

4. Explicit losses and gradient boundaries

First, train the interface checks using frozen \(E\) targets. Exact-content reconstruction, source-pointer, field-type, polarity, operator, condition, and rubric-direction losses update \(D\) and, where applicable, the codecs. No gradient reaches \(E\).

Then train decision behavior:

- Status cross-entropy updates the aligned Qev derivative and decision heads.
- Noul probability loss applies only to reviewed assessed probabilities.
- Score regression applies only to assessed values for the exact bound rubric.
- Choice loss accepts the reviewed acceptable-identity set; infeasible or unresolved examples receive no fabricated selection target.
- Reason-type and source-reference losses train bound reasons.
- Result-vector matching trains \(C_{\text{out}}\) against frozen encoder targets for reviewed result fields.
- Semantic recovery losses through an accepted frozen \(D\) constrain codec outputs.

Decision losses may update \(C_{\text{in}}\), \(C_{\text{out}}\), and the declared decision owners. They never update the base, original Qev artifact, or reference encoder. Absence masks govern both value losses and result-vector slots.

After interface and decision acceptance, freeze those owners. Train native request formation with:

- Action and invocation-location supervision.
- Field-type, length/termination, source-pointer, and scalar losses.
- Normalized vector matching against stop-gradient reference targets.
- Semantic recovery losses through frozen accepted checking heads.
- Native continuation losses using only accessible context and the delivered result.

These losses update only \(N\). Discrete source selections receive explicit pointer supervision; gradients do not magically pass through an argmax. Decision evaluation is frozen, and its boundary stops gradients into the native request generator, consistent with the original proposal.

Native formation uses teacher-forced target slots initially, followed by a fixed exposure schedule using its own emitted slots. Free-running requests must pass acceptance; teacher-forced compatibility alone is insufficient.

5. Dependency order and semantic proof

The dependency order is:

1. Review source-grounded records and lock family splits.
2. Freeze serializer, reference encoder, and target normalization statistics computed from training data only.
3. Establish exact-content and field-recovery checks.
4. Train and accept codecs and decision semantics.
5. Freeze the accepted interface and decision owners.
6. Train native formation and consumption.
7. Evaluate autonomous formation, serial invocation, and result use on untouched families.

Reconstruction is necessary but insufficient. Semantic checks must include reviewed contrast pairs where a small change alters the correct assessment: negation, “any workable” versus “certified best,” known unavailable versus unknown availability, reversed rubric direction, changed quantity constraints, and reordered alternatives with unchanged stable identities. Conversely, paraphrases and irrelevant facts should preserve the assessment.

Evaluate both:

- **Interface fidelity:** recover the reviewed fields, exact references, and conditions.
- **Task fidelity:** obtain the correct joint status, value/content, and bound reasons after those vectors enter the decision system.

A decoder reproducing text while the decision system ignores a condition fails task fidelity. A decision system succeeding through correlated wording while losing exact bindings fails interface fidelity. Native acceptance additionally requires independently reviewed request meaning and correct result-sensitive continuation.

Missing reviewed request decompositions, probability-target definitions, acceptable-choice sets, rubric interiors, reason bindings, autonomous trajectories, or reliable checking artifacts are blockers for their dependent stages. Encoder usefulness and recoverability are also unproven gates; this design does not substitute opaque vector matching for that evidence.

Provenance: no tools, files, repository history, browsing, implementation context, or other agents were used.
