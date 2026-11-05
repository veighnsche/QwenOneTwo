# Complete Score training walkthrough: packing-shift interruption

Status: recorded explanation of the proposed supervised training trajectory.
This is not a model run, approved gold data or a claim of trained capability.
No coding, training, model execution, service changes or deployment are
authorized by this document. All model workloads remain GPU-only.

This expands [the complete training examples](complete-training-examples.md),
following the [Noul](noul-training-walkthrough.md) and
[Choice](choice-training-walkthrough.md) walkthroughs. W is the shared frozen
backbone, R the trainable native adapter and D the decision adapter initialized
from Qev and aligned to the latent protocol. Native mode is W+R; decision mode
is W+D, with mutually exclusive adapters on one backbone.

This walkthrough owns its example-specific teaching targets. The
[implementation specification](architecture-and-training-plan.md) owns dispatch
timing/shared contracts; the [handoff](implementation-handoff.md) lists remaining
codec/loss choices. Pretrained W is frozen; reserved scalar rows are explicit
learned extensions with validated gradient/tying handling.

## 1. Ordinary human request

> Help me prepare Saturday's two-hour volunteer packing shift. The label
> printer sometimes jams. Clearing a jam takes about two minutes, after which
> printing resumes. Volunteers can fold boxes while it is cleared. Labels are
> required before parcels can leave, and there is no spare printer. What should
> we prepare?

Only this request enters as the human prompt. Semantic questions, rubric
annotations, reference scores and reference continuations stay in the separate
training record. No operation hint or correct score is added to inference input.

The evidence includes recovery time, useful alternative work, the dependency
on labels and absence of a spare. Sometimes does not supply a numerical
occurrence probability. No failure rate, repair cost or automatic recovery
mechanism is invented.

## 2. Single prefill and continuous startup budget

W+D prefills the human prompt once and produces the initial continuous thinking-
budget score. The draft effort target is s=0.06, mapping through the recorded
S-curve to B_pred=1359 reasoning positions, subject to context admission. The
example configuration reserves A=160 visible-answer positions and H=64
restricted control positions.

What learns: the startup budget readout and its trainable decision path learn
to predict an annotated effort preference. This target is a review placeholder,
not evidence of the effort required by the task or a fixed budget for Score.
The runtime performs the mapping and enforcement.

Select native W+R while retaining the same causal state. The upcoming task Score
has a separate meaning and does not change the startup reasoning allowance.
See [position and budget accounting](position-limits-and-budget-accounting.md).

## 3. Identify a useful scoring question

In this selected supervised trajectory, native Qwen assesses:

> How disruptive would one described jam be to this packing shift, assuming
> it occurs and clears as described?

This is conditional disruption severity: how disruptive the event would be if
it happens, not how likely it is to happen.

What learns: native R learns to identify the target issue and choose a dimension
relevant to preparing the shift. It must distinguish that dimension from jam
probability, total disruption from repeated jams, repair cost, overall readiness
and thinking effort.

This is one useful intermediate assessment. A direct plan or another meaningful
route can also be valid. The selected training trajectory supervises Score;
free evaluation removes supplied actions and invocation boundaries. Dispatch
occurs only after the target, question and rubric are complete.

## 4. Form the latent question, rubric and evidence bindings

The question uses eight learned 1024-wide vectors. The rubric uses ten: two
vectors for each of five semantic fields.

| Rubric field | Intended meaning |
|---|---|
| Dimension | Interruption to completing the packing work |
| Conditions | One occurrence during the supplied shift; exclude cumulative occurrence frequency |
| Zero anchor | No meaningful interruption |
| One anchor | Essential work cannot resume and completion is blocked |
| Midpoint reference | Noticeable, recoverable interruption to an essential activity, with limited useful parallel work |

Exact metadata records the continuous range [0,1], higher-means-more-disruptive
direction, midpoint presence and rubric version.

The target and evidence remain separately bound: printer jam, two-minute
recovery, box folding, required labels, two-hour shift and absence of a spare.
Bind attributes to their actual supplied source clauses. The runtime owns
stable identities; learned pointers choose the relevant objects rather than
predicting arbitrary UUIDs.

What learns: native R, question/rubric writers and reference selection learn
what is being measured, under which conditions, what the scale means and which
issue/facts are being assessed. The rubric must describe the scale independently
of the expected result. It must not encode this jam is mild or return 0.30.

English descriptions above are offline annotations. Inference uses learned
vectors and typed metadata without generated tool-question text or JSON. No
individual latent coordinate is assigned an English meaning. Exact alignment
objectives and weights remain implementation details requiring validation;
this walkthrough does not establish that a working semantic encoding exists.

## 5. Seal the request and enable Qev

The runtime freezes the exact question, target, evidence and rubric versions.
It assigns request identity, validates structure and admits the bounded call
against the remaining reasoning/context allowance. Then it selects W+D on the
same attention/convolution/recurrent state lineage.

What learns: nothing from identity assignment, freezing or limit enforcement.
Those are deterministic responsibilities. Qwen learns meaningful criteria and
relevant evidence selection; the runtime does not decide that a two-minute
interruption deserves a particular value.

No independent question prefill, cache reset or second full model is assumed.
Formation, receiver, return and consumption positions are charged under the
budget contract. Alternating-mode execution and its differentiable training
path remain subject to the unverified
[cache/gradient feasibility gate](cache-and-gradient-strategy.md).

## 6. Produce a status and continuous scalar

Proposed reference annotation:

    Semantic status: Assessed
    Illustrative scalar point target: 0.30
    Proposed acceptable interval: [0.20, 0.40]

The draft label reflects short recovery relative to the shift and useful
parallel work, while a required activity still pauses and there is no spare.

These numbers require independent review before becoming training labels.
There is no objectively established decimal here. If reviewers cannot apply
the rubric consistently, revise it instead of declaring this decimal correct.
See [annotation and readiness procedures](training-data-and-readiness.md).

What learns: the aligned Qev receiver and continuous Score readout learn to
interpret the question/rubric, evaluate supplied evidence and produce a numeric
value. The scalar is returned directly rather than generated as characters
spelling 0.30. It is not a selection among low/medium/high categories.

If reviewers approve a point target, a regression loss can train against it
with the reviewed interval used for acceptance. The
[loss configuration](loss-configuration-and-training-schedule.md) fixes raw
point-MSE versus interval-distance loss and weights; the
[A4 profile](training-run-profile-v1.md) fixes run settings. Scale anchors and acceptable intervals do
not discretize the output into score bins.

Semantic status is trained separately. Missing essential evidence should yield
Insufficient evidence with no valid scalar, not zero or a generic midpoint.
Numeric losses are masked for non-assessed returns. A genuine zero means the
zero anchor applies, not that information is absent.

Whole-flow training observes Qev's actual prediction. A wrong well-bound score
remains a semantic failure and receives loss; the runtime never replaces it
with the gold score. Supplied gold returns used in isolated reader exercises
are explicitly identified as supervised.

## 7. Interpret the bound score in native mode

The runtime validates the return against the pending request, including target
and rubric version, selects W+R and delivers the result once.

What learns: the result encoder and native R learn to interpret 0.30 for this
issue under this particular disruption rubric. That supports treating a single
described jam as a manageable interruption while preparing around the labeling
dependency.

It does not mean a 30% jam probability, 30% confidence, 30% readiness or 30% of
the thinking allowance. Its meaning depends on the bound dimension, conditions,
anchors and direction, not merely the fact that it lies between zero and one.

The scalar alone does not reveal the recovery time or suggest box folding.
Those facts come from the original evidence. In an isolated reader exercise,
provide the rubric and score while withholding original evidence: Qwen should
interpret severity but must not invent two minutes or fold boxes unless those
facts are separately available. This tests result interpretation without
independently re-solving the original assessment.

## 8. Generate the normal preparation plan

Acceptable continuation:

> Keep the jam-clearing instructions handy and assign someone to handle a jam.
> Have the other volunteers fold boxes while printing is paused, then resume
> labeling before parcels leave. A single two-minute interruption should be
> manageable with that plan. Check how often jams occur, since repeated stops
> could create a labeling backlog.

What learns: native R combines the bound assessment with concrete evidence
and the user's planning goal. It must distinguish a manageable single event
from an unknown cumulative impact and avoid invented failure rates, repair
costs or spare equipment. The normal answer need not expose the internal decimal.

Visible text uses the answer allowance; any further latent call during answering
uses remaining reasoning allowance, which does not reset between phases.

## 9. Update the connected training flow

Combine supervision for:

- Target and evidence selection.
- Question meaning and rubric fidelity.
- Score invocation.
- Semantic status and the reviewed continuous value.
- Native interpretation and appropriate final continuation.

Unlike hard Choice selection, the continuous score can carry answer-loss
gradients through result encoding into Qev's readout and latent request
formation, provided that path stays connected. Hard invocation/status decisions
still need their own supervision. Answer loss alone cannot establish meaningful
criteria or actual use of the return. Numeric supervision alone cannot establish
that the native reader understands the score.

W stays frozen while adapters and protocol components learn. Computation through
W must preserve gradients needed by those components; do not detach at the
call boundary or train only a bridge around an entirely frozen native path.
Use complete training-safe call/return unrolls before optimizer updates, subject
to GPU feasibility and training/serving forward checks.

## Companion checks and interpretation

| Change | Expected behavior |
|---|---|
| Change only occurrence frequency | Preserve conditional one-occurrence severity, with other circumstances fixed |
| Printing cannot resume | Higher disruption under the original rubric |
| Omit essential recovery information | Potentially Insufficient evidence, without a fabricated midpoint |
| Reverse the same scale's direction | Reverse numeric interpretation and create a new rubric version |
| Attach the return to another issue or rubric | Reject mismatched binding |
| Supply different valid scores in isolated reader exercises | Continuations follow their rubric-defined meaning |

Keep related variants within the same dataset split. A new rubric cannot
silently relabel an old result's meaning. Score changes are interpreted relative
to the actual rubric rather than a universal high/low response template.

Free whole-flow evaluation must show Qwen forming the rubric, invoking Score,
receiving Qev's actual value and using it appropriately without supplied actions
or boundaries, in the same inference run. Guided execution or a correct final
answer alone does not establish autonomous invocation and result use. Follow
the recorded readiness plan and report scope/length limits honestly.
