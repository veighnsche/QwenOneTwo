# Typed result statuses

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

Owner: semantic status and value-presence rules. The
[implementation specification](architecture-and-training-plan.md) owns the
common lifecycle and [handoff](implementation-handoff.md) owns readout/codec
preparation. These statuses are trained extensions, not assumed stock Qev skills.

## Core contract

A return is bound to an immutable request snapshot. Its semantic status
determines whether a decision value exists. Missing evidence is never encoded
as false, zero, 0.5, or an arbitrary candidate. A raw numeric readout computed
internally must be masked or discarded for non-assessed returns.

Presence is explicit; never use scalar truthiness to test validity. An assessed
zero is present, whereas a non-assessed result has no decision payload.

| Semantic status | Meaning | Decision payload |
|---|---|---|
| Assessed | Qev can support the requested assessment | Present |
| Insufficient evidence | Material information needed for this assessment is missing | Absent |
| Conflicting evidence | A material conflict cannot be resolved from the request | Absent |
| Unresolved | The receiver cannot interpret or resolve the assessment reliably | Absent |
| No feasible choice | Every alternative in this bank is ruled out under the criterion | No selection; Choice only |

Sufficiency and conflict are relative to the question. Missing irrelevant fields
do not automatically prevent an assessment. A decisive known fact may suffice.
Conflicting delivery ages of twelve and fifteen days need not prevent evaluating
a thirty-day policy; twelve versus forty-five may create a material conflict.

## Noul

Assessed returns contain a probability estimate for the specified proposition
being true. A known qualifying refund case can support a high estimate; a known
case outside the supplied policy window can support a low one. Neither is the
same as insufficient evidence.

A reported defect with no delivery date may require Insufficient evidence.
Material unresolved contradictory dates may require Conflicting evidence. Such
returns contain no probability value for the reader to interpret.

Do not implement “near 0.5 means insufficient evidence.” A balanced probability
estimate can have legitimate meaning; status is a separately trained assessment.
Nor is the estimate automatically calibrated confidence.

The native reader learns to ask for the missing date or clarify conflicting
facts instead of treating the customer as ineligible.

## Choice

Relative candidate ranking does not establish feasibility. A pointer head has
a largest score whenever alternatives exist, even if every alternative is bad.
Learn evidence sufficiency and candidate feasibility alongside ranking.

| Situation | Appropriate outcome |
|---|---|
| Tuesday is confirmed for everyone and Wednesday is uncertain; request asks for a workable slot | Assessed: Tuesday can be selected |
| Tuesday is ruled out and whether Wednesday works is unknown | Insufficient evidence |
| Every supplied slot has a supported disqualifying conflict | No feasible choice |
| Two slots satisfy the criterion equally and no preference is supplied | Assessed: either acceptable slot, without inventing a preference |

No feasible choice applies only to the frozen bank under the specified
criterion, not to all possible alternatives in the world. Missing availability
must not become known unavailability. “No slot is confirmed by the supplied
statements” differs from “everyone is definitely unavailable”; the question
determines which conclusion is being assessed.

If the request asks for the best alternative, an unknown candidate may prevent
certifying a known workable one as best. If it asks for any workable alternative,
one confirmed candidate can suffice. The assessment is not a blanket check that
every possible field or candidate is fully known.

## Score

Assessed with value zero means the rubric's zero anchor applies. Insufficient
evidence has no numeric value. Midpoint 0.5 retains its rubric meaning and is
not a generic uncertainty marker.

For conditional demo-disruption severity, evidence of no material interruption
could support a value near zero. Missing information about the effect cannot
be substituted with zero.

A categorical status head determines whether the scalar is valid at all; it
does not discretize the continuous Score value. Non-assessed examples must not
receive fabricated numeric targets during training.

## Reasons and native continuation

Non-assessed returns include a short latent reason and appropriate bindings:

- Insufficient evidence: what needed information is missing for which target.
- Conflicting evidence: which supplied statements conflict materially.
- No feasible choice: the evaluated bank and relevant constraint failures.
- Unresolved: difficulty with criteria, interpretation or reliable assessment.

For a missing delivery date, describe the missing information and bind it to
the case; do not fabricate a source reference to a nonexistent date.

The native reader learns focused questions, qualified answers, requests for
additional alternatives or acknowledgment that reliable assessment is not
possible. It must not treat all statuses as a single generic failure template,
repeat identical calls as though they create evidence, or blindly follow values
that conflict with the request's facts.

## Runtime outcomes are separate

Invalid references, malformed requests, cancellation, execution failure and
budget exhaustion are execution outcomes. They are not evidence that a
proposition is false, a score is zero, or no alternative is feasible.

The runtime can detect structural failures and enforce resource limits. It
must not infer semantic statuses through domain rules or probability thresholds.
Normal product replies should express the substantive issue, not expose
internal error identifiers as though they were useful human instructions.

## Training and verification

- Supervise semantic statuses on supported, missing, conflicting and unresolved
  examples, including cases with decisive facts despite irrelevant omissions.
- Apply Noul/Choice/Score value losses only where assessed values are valid.
- Supervise Choice feasibility separately from relative preference, including
  unknown alternatives and no-feasible-bank cases.
- Train latent reasons and update Qwen's native LoRA on appropriate continuations.
- Check fabricated assessments and excessive abstention as distinct failures.
- Verify that values remain bound to their question, target, bank and rubric.

These are learned, fallible judgments. A status tag does not prove that an
assessment or abstention is correct. The contract prevents representational
confusion; training and held-out verification must establish correct use.
