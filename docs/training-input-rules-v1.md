# Role-specific training-input rules v1

The opt-in [direct-role-states-v1 contract](direct-role-states-v1.md) versions
variable request-role lengths, native emission capacity and complete return
reservations together. The rules below remain the legacy default; the new
profile is experimental and its GPU assertions are pending RD-07.

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

Owner: per-role canonical input whitelists, branch exposure, permitted
outcome-bearing Reason supervision and associated masks/provenance. Specializes
[semantic supervision](semantic-supervision.md), [data operations](data-operations.md)
and [losses](loss-configuration-and-training-schedule.md). Follow the
[specification](architecture-and-training-plan.md), [B1](neural-input-encoding-v1.md)
and [module/optimizer contract](module-and-optimizer-contract-v1.md).
The [F3 clean-test contract](clean-test-construction-v1.md) separately records
permitted-view tensor construction; the [F8 promotion matrix](promotion-matrix-v1.md)
separately records evaluation coverage and scoring.

## 1. Scope and provenance boundary

Answer blindness applies to request targets. Request annotations describe what
to assess and available facts, excluding the annotator-created CURRENT verdict,
winner, target score, final answer or preferred thinking budget. Outcome targets
remain loss labels. Reason is a post-decision semantic role and may explicitly
describe its justified non-assessed status and evidence.

This is a provenance/role rule, not a keyword filter or a claim of zero answer
information. Useful supplied facts can determine the answer. Keep them. A source
may legitimately state that a refund was approved; preserve its attribution
when relevant. Do not manufacture that fact from the current reference label.
User claims and model reasoning remain attributed claims, not verified truth.

The live native writer necessarily sees the actual causal human prompt/history.
Canonical reference inputs are separately whitelisted. Matching reference
targets and interventions constrain writer shortcuts; they do not prove that
every live latent vector is mathematically independent of an outcome.

## 2. Per-role reference inputs

| Role | Permitted canonical annotation | Excluded |
|---|---|---|
| Question, eight slots | Proposition, criterion, scope, qualifications, polarity and necessary constraint values | Annotator's current verdict, selected alternative, target score, preferred budget or case-specific conclusion disguised as question wording |
| Target, two slots | Neutral subject/referent description necessary to identify it | Derived assessment such as the ineligible customer or high-severity incident |
| Alternative, four slots | Its own identity, supplied properties and grounded factual attributes | Current winner/rank, criterion-derived feasibility, unsupported suitability or other alternatives' facts |
| Rubric, five fields/two slots each | Dimension, conditions, direction, zero/one/optional midpoint anchors | Current target's assessment, desired score or case-specific conclusion |
| Attribute key, two slots | Property and unit meaning | Actual current value or derived assessment |
| Reason, four slots | Approved non-assessed status and material missing information, conflicts or constraints supporting it | Fabricated evidence, unsupported justification, current assessed truth/winner/score |

Exact case values belong in explicit typed attributes or evidence, retaining
source bindings. Target slots identify the subject without smuggling its
assessment; target attributes may carry legitimate supplied case facts.
Candidate availability/rate facts remain useful permitted inputs. A generic
rubric anchor describing a complete failure is a scale definition, not the
current case label. Necessary criterion numbers/negation belong in the question.

Example: Question asks whether the bound participant meets workshop access
requirements; Target neutrally identifies the participant; evidence supplies
completed orientation and an expired badge. Do not encode 'confirm this person
is ineligible because their badge expired' as the question target.

## 3. Evidence, exact fields, metadata and controls

Evidence/source-state inputs contain only selected grounded facts and attribution
under their declared view. They may explicitly bear relevant source outcomes.
Previously available results use immutable historical bindings and retain their
fallible-assessment provenance. A prior result is not an unquestionable current
label. Synthetic provided results in component exercises are labelled as such,
never claimed to be actual self-invoked model outputs.

| Channel | Rule |
|---|---|
| Request headers/descriptors/EVALUATE | Derive from actual request and B1; current result status absent; no gold verdict, winner or feasibility flags |
| Typed atoms/source spans | Encode selected grounded data exactly; no fact manufactured from desired answer |
| IDs/handles/order | Bind objects, not labels; no winner-correlated naming or layout |
| Writer feedback/counters | Actual draft, ledger and execution state only |
| Budget feedback | Detached predicted score and actual grant; reviewed effort remains a loss target |
| Runtime notices/boundaries | Mechanical control-flow contract, no semantic labels |
| Result status/value/Reason | Actual evaluator output in joint execution; reviewed substitutes only in declared component exercises |

Early training's configured bounded B override is explicitly logged, never
chosen per example from its gold effort label. Effort labels are reviewed
preferences, not proven minimum thinking requirements. Unused roles remain
unemitted according to the control owner; this adds no protocol fields/positions.

## 4. Branch-specific exposure

| Branch | Model-facing input |
|---|---|
| Reference encoder | Only canonical annotation permitted for that role, never whole annotated scenario row |
| Native writer | Actual causal prompt/history, available evidence, draft and counters; no appended reference annotations/outcome columns |
| Isolated receiver | Declared request packet and permitted evidence; expected status/value/selection are loss targets |
| Joint receiver | Actual shared history and generated request; no substituted gold status or return |
| Isolated reader | Reviewed bound return, including outcome/Reason, may deliberately be supplied to teach interpretation |
| Joint reader | Actual evaluator return; continuation target valid for actual state |
| Reconstruction decoder | Latent record and role/slot/presence; source annotation is reconstruction target, not supplied prefix |

Role-labelled reference encoding contexts are separate. Never concatenate Reason
supervision into a Question/Alternative/Rubric encoding context. Reference fitting
uses semantic reconstruction/contrast objectives, not decision-outcome objectives.
Reason status can be part of reconstructed meaning. Shared reference weights do
not erase role-specific input rules. All reference artifacts freeze/version
before live joint training; this changes no phase lengths or coefficients.

Offline component encoding is not a second prefill inside production inference.
The production contract remains one submission, one prompt prefill, same cache.

## 5. Permitted outcome-bearing Reason supervision

V1 Reason accompanies only non-assessed results. Assessed Noul/Choice/Score
returns do not gain an additional Reason record.

| Status | Permitted meaning |
|---|---|
| Insufficient | Evidence does not establish a material required fact, e.g. orientation completion |
| Conflicting | Available sources materially disagree, e.g. active versus expired badge |
| Unresolved | A specific supported ambiguity prevents resolving this bound question |
| NoFeasibleChoice | Every supplied alternative violates a mandatory constraint, with grounded support |

Missing irrelevant information does not justify abstention. Unknown feasibility
does not prove no feasible choice. Reason can state its non-assessed status and
include bounded relevant facts/references, including available historical results.
Missing information receives no fabricated source span. Attachments obey sealed
catalogue K and existing size/admission limits; no new evidence is invented.

This exception does not permit status/verdict in Question targets, winner tags
in Alternatives, or desired Score in Rubrics.

## 6. Predicted-status mismatch and rollout divergence

Reason targets must match actual request, actual selected non-assessed carrier
status and evidence. Reference Assessed/model abstains: retain status and valid
raw-value losses, but invent no abstention Reason. Reference non-assessed/model
Assessed: retain status loss, invent no numeric target. Mask incompatible
Reason-specific meaning targets; this does not suppress valid numeric losses
through predicted abstention.

A separate explicitly guided Reason exercise may teach the reviewed correct
status/reason. It does not replace the joint return. If actual generated question,
rubric or candidate bank changes meaning, do not inherit old outcome/Reason
labels. Preserve supported formation/binding losses, mask unsupported targets,
and count/report divergence. An unvalidated decoded gloss is not a relabelling
oracle. Keep loss eligibility tied to actual supported packet semantics.

## 7. Teacher forcing and outcome exposure

Ordinary next-token training supplies preceding approved native tokens; a prefix
may already contain a conclusion. Permit this as declared guided training only:
no future tokens, continuation valid for actual state, and outcome-bearing prefix
exposure recorded separately from ordinary teacher forcing. Such exposed cases
cannot demonstrate independent outcome derivation. Free evaluation has neither
native teacher forcing nor supplied decision hints/returns/overrides.

Gold return in isolated reading teaches interpretation, not successful evaluation
or invocation. Canonical request targets stay answer-blind in all branches.
Reconstruction teacher forcing supplies preceding gloss tokens as decoder targets,
not the whole source annotation as an encoder/prefix input.

## 8. Review, manifests and required checks

Every builder uses an explicit whitelist. Record evidence scope, training mode,
source frontier/bindings and provenance outside model-facing packets. Review
request semantics with proposed outcomes hidden, independently derive outcomes
from that view's available evidence, then review Reason with status/support and
continuation for actual return. Do not label from withheld world truth.

Required future checks:

1. Swapping only gold outcome/status/score/final-answer columns leaves request
   reference input bytes/hash unchanged.
2. Same question/rubric across changed case evidence has identical canonical
   reference inputs and deterministic frozen targets.
3. Same alternative's own facts across competitors/criteria retains canonical
   input; order invariance only when order is incidental to the task.
4. Changing attribute value leaves key annotation unchanged.
5. Changing approved Reason affects Reason encoding, not request targets.
6. Reject outcome-bearing Reason annotations in request-reference builders.
7. Labels are justified by evidence available in the named training/test view.

Counterfactual Reason edits must themselves be valid for that Reason exercise;
arbitrary status swaps are not automatically valid task labels. All tensor-based
checks require future GPU execution. No trained result or passing isolation test
is claimed. The clean-test owner addresses contextual tensor construction;
F8 still owns
coverage, denominators and promotion.
