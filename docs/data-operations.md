# Annotation, review and dataset provenance

Implementation update (2026-11-05): a read-only, model-free
[reviewed-release loader and format](reviewed-release-format-v1.md) now bind
actual source, annotation, review, split and approval bytes. See the
[release-boundary evidence](backend-release-slices-2026-11-05.md). No real reviewed
pilot, assigned reviewers or semantic approval is implied; the historical status
below describes the procedure's original recording.

Status: recorded operating procedure, not an existing reviewed dataset or
assigned reviewer team. No coding, data/model generation, training, publication
or deployment is authorized by this document. Model work remains GPU-only.

Owner: exact annotation instructions, review/adjudication, source coordinates,
provenance and release lifecycle. [Data/readiness](training-data-and-readiness.md)
owns quantities, splits and semantic acceptance; [generation](training-data-generation.md)
owns scenario construction. Follow the [specification](architecture-and-training-plan.md),
[semantic supervision](semantic-supervision.md) and
[loss masks](loss-configuration-and-training-schedule.md).

## 1. Pilot and separated record fields

Begin with 120 distinct authored fictional scenarios, 20 per agreed domain,
and deterministic wording templates. Variations follow scenario construction.
No model generator is required for this pilot. A future GPU teacher may propose
wording/annotations but cannot certify its own outputs.

| Field group | Contents |
|---|---|
| Human input | Natural request and supplied evidence |
| Semantics | Questions, alternatives, rubrics and bindings |
| Reference outcomes | Status, valid values/selections and acceptable continuations |
| Guidance | Selected demonstration route, valid alternatives and writer events |
| Provenance/review | Origin, transformations, checks, reviewers and decisions |

Input builders whitelist fields per training branch. Gold outcome columns never
enter request-question/rubric reference encoders. Ordinary inspectable dataset
serialization is allowed; textless is the deployed internal-call contract.

The [training-input rules](training-input-rules-v1.md) own the complete role and
branch whitelist: current outcomes stay out of request targets; justified
non-assessed status/support is permitted in Reason supervision. Record branch
evidence scope, actual causal frontier, forcing and outcome-bearing prefix
exposure. Review Reason after status/support and continuation for actual state;
do not derive labels from withheld world truth or serialize whole annotated rows
into model inputs. Reviewers and passing artifacts remain unassigned/unperformed.

## 2. Exact annotator instructions

- Use supplied facts/rules only. Invent no dates, timezones, policy exceptions,
  failure rates, repairs, bookings or completed external actions.
- Preserve the user's actual goal/constraints. Keep internal operation labels,
  writer instructions and forced boundaries out of ordinary human prompts.
- Annotate acceptable alternatives, not one mandatory sentence or route.
- Separate supplied facts, derived conclusions and unknown information.
- Bind factual attributes/explanations to evidence. Mark absent labels absent,
  not zero, false or an empty-string substitute.

| Record | Required annotation |
|---|---|
| Question | What to determine, with necessary qualifiers/polarity; separate case evidence |
| Alternative | All supplied alternatives, exact attributes, merged genuine duplicate mentions; no winner tags |
| Rubric | Dimension, conditions, direction and anchors independent of target score |
| Native continuation | Use bound result and available facts; invent no reasons withheld from an isolated reader |

A correct answer cannot excuse a wrong question, omitted candidate or binding.
Case-specific criterion numbers may legitimately belong in a question; the
prohibition concerns smuggled verdicts, not necessary question content.

## 3. Status and value instructions

Identify the precise proposition/criterion/rubric; determine material evidence;
decide whether assessment is supported; assign status; assign a value only when
that status permits it. Missing irrelevant facts do not automatically prevent
assessment. Conflicts matter only where material to the assessment.

- Noul: initially use determinate truth labels justified by supplied rules.
  Do not invent confidence decimals. Missing evidence does not mean false.
- Choice: label entry feasibility and the complete acceptable selection set.
  Unknown availability is not unavailability. No feasible choice requires
  supported rejection of every bank entry under the actual criterion.
- Score: assess the stated dimension/conditions only. Review point/interval
  labels; default maximum interval width is 0.20.
- Effort: annotate an agreed preference from prompt/guidelines before model
  performance is seen; never claim minimum required budget or infer it from
  operation type. Leave unsupported effort labels masked.

The .30 severity and .04/.05/.06 example efforts remain drafts until reviewed.
Annotation disagreement is a workflow issue, not automatically a semantic
Unresolved example. That model status requires task-grounded justification.

## 4. Immutable sources and coordinates

Store original source content and its hash immutably. Annotators select exact
substrings; store source identity and byte ranges. Materialization maps these
to token-aligned source spans under the pinned tokenizer/template and verifies
their mapping to the original prefill. Keep source-local coordinates independent
of later cache growth.

Do not normalize/edit original text after offsets exist. If token boundaries
require a broader span, record/review the change rather than silently shifting
coordinates. Objects, banks, rubrics and results use exact versions; opaque IDs
are bookkeeping, not values the model must invent.

## 5. Independent semantic and outcome review

First review human task, source bindings, question, bank and rubric while hiding
proposed gold outcomes/model predictions. Then independently derive status,
values, selections and continuation requirements. Reviewers still see evidence
and can infer an answer; outcome blindness means not copying the proposed label.

| Material | Required review |
|---|---|
| Every pilot scenario | Two independent semantic/outcome reviews |
| Locked semantic tests | Two independent reviews |
| Subjective Score and effort labels | Two independent reviews |
| Other training examples | One independent review plus stratified second-review audit |
| New/changed rubric definitions | Two independent reviews |

The author cannot be the sole reviewer of their own label. Model-assisted
checking is recorded as such, not an independent human review. Reviewer roles
are currently unassigned; do not claim completion or name fictional reviewers.

## 6. Adjudication and batch audit

Reviewers judge independently before seeing each other's conclusions. Resolve
categorical disagreement using supplied facts/criterion. For subjective scores,
compare anchor interpretations, then agree a justified point/interval. Never
automatically average disagreeing numbers. Revise rubric/example or exclude
unresolved material.

Bulk training batches contain at most 200 examples. Second-review at least 10%,
with minimum 20 where possible; audit all if smaller. Stratify by operation,
status, domain and transformation. All material already requiring two reviews
retains that requirement; the audit is not a substitute.

A critical error (truth, winner, status, source binding, hidden outcome label or
unsupported claim) quarantines the affected batch. Investigate generator/rule/
template and descendants and re-review affected material. Merely deleting the
discovered row is insufficient. Preserve rejected examples and reasons in the
audit history; do not erase lineage to improve apparent review results.

## 7. Lineage and splits

Assign scenario family and split before creating paraphrases, permutations and
counterfactual variants. Descendants remain together. Check duplicate content,
near duplicates and derivation history across splits. Pilot material is for
development; create fresh locked-test families after instructions stabilize.

Original three natural prompts/close paraphrases remain excluded from training/
tuning. Workshop/printer/packing walkthroughs are design-exposed; they may seed
reviewed training, not fresh blind tests. Exclude test material from teacher
prompts, negative pools, reference fitting and correction datasets used for
training. Test-driven tuning retires that set as development and requires a
new locked release for a fresh claim.

## 8. Mandatory provenance

| Category | Required record |
|---|---|
| Identity | Record/version, family, parents and split |
| Origin | Authored, deterministic, model proposal, replay or rollout correction |
| Construction | Scenario spec, generator/template version, seed and transformation |
| Sources | Content hashes, source/version IDs and bindings |
| Labels | Schema version, derivation method and applicable masks |
| Review | Reviewer IDs, independent judgments, adjudication and rejection reasons |
| Model assistance | Exact model/checkpoint, prompt, sampling config, software, GPU and output hashes |
| Rollouts | Producer checkpoint, actual trace, forcing/substitution/override flags |
| Release | Manifest/content/split hashes, validation and exclusion summaries |

Model-free records have no fabricated GPU-run metadata. Replay needs known
origin and reuse rights; weights do not imply original training examples are
available. Re-review legacy curriculum under new semantics. Model proposals,
decoded glosses and prototype answers are not automatically gold labels.

## 9. Release lifecycle

    draft -> mechanically checked -> reviewed -> adjudicated if needed
          -> approved for named split/release

Mechanical checks cover schema, presence masks, ranges, spans, references,
duplicates and lineage, not semantic correctness. Each release reports review
coverage, unresolved/quarantined records and operation/status/domain counts.
Edits create new record/release versions; prior reports keep their original
hashes. Generated datasets/traces stay under ignored data/artifact paths;
documentation authorizes no commit or publication.

The first concrete artifact is the small pilot with scenario specs, independent
reviews and provenance manifest. Scale production only after pilot review and
the GPU feasibility gate. These activities have not been performed by recording
this procedure.

The [clean-test contract](clean-test-construction-v1.md) adds per-view construction
lineage: permitted inputs, derived excerpt sources, encoder/checkpoint, artifacts,
state resets and separate clean/transfer/whole-flow protocol labels. Original
source coordinates/parent hashes remain external audit data; model refs/offsets
resolve to permitted fixture-local contents. Review labels from that view only.
