# Complete Noul training walkthrough: workshop access

Status: recorded explanation of the proposed supervised training trajectory.
This is not a model run, approved gold data, or a claim of trained capability.
No coding, training, model execution, service changes or deployment are
authorized by this document. Model workloads remain GPU-only.

This expands [the complete training examples](complete-training-examples.md)
to identify exactly what learns at each step. The shared backbone W remains
frozen; R is the trainable native adapter; D is the decision adapter initialized
from Qev and aligned to the latent protocol. Native mode is W+R; decision mode
is W+D. These are mutually exclusive adapter modes on one shared backbone.

This walkthrough owns its example-specific teaching targets. The
[implementation specification](architecture-and-training-plan.md) owns dispatch
timing/shared contracts; the [handoff](implementation-handoff.md) lists remaining
codec/loss choices. Pretrained W is frozen; reserved scalar rows are explicit
learned extensions with validated gradient/tying handling.

## 1. Ordinary human request

> Write a reply to Sam about using the workshop tomorrow. Sam has completed
> orientation, but his access badge has expired. Our workshop requires both
> completed orientation and an active badge before someone can use it.

Only this request enters as the human prompt. The expected decision, semantic
question annotation and reference answer belong to the separate training record.

Three relevant facts are supplied:

- Orientation is complete.
- The badge has expired.
- Access requires completed orientation and an active badge.

The rule and evidence make the current eligibility proposition false. This is
a fictional, self-contained policy example, not an external access decision.

## 2. Single prefill and continuous startup budget

W+D prefills the prompt once and produces the fixed initial thinking-budget
score. For this draft example, the proposed effort target is s=0.04, mapping
through the recorded S-curve to B_pred=612 reasoning positions. Context
admission can reduce the granted allowance. The example configuration reserves
A=160 visible-answer positions and H=64 restricted control positions.

What learns: the startup budget readout and its trainable decision path learn
to predict the annotated continuous effort preference.

The effort target is still a review placeholder, not evidence that the task
needs 612 positions. The runtime applies the formula and enforces the allowance;
it does not learn that arithmetic. This startup port is separate from a generic
task Score request. See [budget accounting](position-limits-and-budget-accounting.md).

Select W+R for native execution while retaining the same causal state.

## 3. Identify a proposition worth assessing

In this selected supervised trajectory, the learned internal gate chooses
Noul once the relevant case facts and policy are available.

What learns: native R and the gate learn that this internal state supports a
true-or-false assessment, and when enough information exists to form a request.

Early capability training supervises this action. Later free evaluation lets
Qwen choose without a supplied action or boundary. This example teaches one
valid route; a correct direct answer remains acceptable. Do not infer that all
workshop tasks or all eligibility questions must invoke Noul.

## 4. Form a latent question and select evidence

The intended semantic question is:

> Does Sam currently satisfy the supplied requirements for using the workshop?

This sentence is a human-readable annotation. At inference the learned writer
expresses its meaning through eight learned vectors, each 1024 values wide.
The model does not generate the sentence as an intermediate tool question.

Learned reference selection separately binds:

| Binding | Meaning |
|---|---|
| Target | Sam's workshop-access case |
| Evidence | Completed orientation |
| Evidence | Expired badge |
| Criterion | Both requirements must be satisfied |

What learns: native R, the latent writer and reference-selection components
learn to express the proposition and choose its relevant evidence.

The question must preserve eligibility under the supplied requirements, rather
than merely encode the desired verdict false. Change evidence while preserving
the question, and change the question while preserving evidence, to test this.
A correct verdict on one example cannot establish semantic question formation.
No individual latent coordinate has an assigned English meaning. The exact
alignment objectives and their weights need specification during implementation;
this walkthrough does not claim a validated vector encoding already exists.

## 5. Seal the request and enable Qev

The runtime freezes the selected immutable objects/versions, assigns a request
identity, checks structure and admits the call only if its bounded completion
fits the remaining allowance. It then selects W+D.

What learns: nothing from assigning an identity or enforcing limits. These are
deterministic bookkeeping operations. Qwen learns which objects to select;
the runtime preserves those choices and rejects structural errors. It does not
evaluate workshop eligibility or replace a semantically wrong evidence choice.

Qev continues from the existing attention, convolution and recurrent state and
processes the latent request. There is no independent question prefill, cache
reset or second full model. Formation, receiver, return and consumption
positions count under the recorded budget rules. Training and serving must
use the intended alternating-mode history, subject to the unverified
[GPU cache/gradient feasibility gate](cache-and-gradient-strategy.md).

## 6. Evaluate the proposition

Reference annotation:

    Semantic status: Assessed
    Truth target: False
    Noul probability target: p_true = 0

The supplied facts support an assessment: one required condition is known to
fail. This is different from not knowing whether the badge is active.

What learns: the aligned Qev receiver and readouts learn to understand the
latent proposition, apply the supplied requirement to the bound evidence,
distinguish a supported negative from missing information, and produce a low
probability for this proposition being true.

The binary truth target supplies a probability-training signal. A readout need
not equal floating-point zero, and the label does not establish calibrated
confidence. Numeric supervision applies only when the status permits a value.

During whole-flow training Qev produces its actual prediction. A wrong result
remains a recorded semantic failure and receives training loss; the runtime
does not substitute the gold answer. Supplied gold returns used in isolated
reader exercises must be explicitly identified as supervised exercises.

## 7. Understand the bound result in native mode

The runtime attaches/validates the return against the pending request, selects
W+R and delivers the result through the learned latent representation for
one-time consumption.

What learns: the result encoder and native R learn that the low probability
applies to this particular proposition: Sam does not currently satisfy all
the access requirements. It does not mean orientation is incomplete or Sam
can never gain access.

The probability alone does not identify which condition failed. The expired-
badge explanation comes from the original evidence, which remains available
in the full-flow case. It is not information magically contained in false.

An isolated reader exercise can withhold the badge evidence and provide the
question plus a valid false return. In that exercise Qwen may conclude the
requirements are not satisfied, but must not invent the cause as an expired
badge unless that reason is separately supplied. Such isolation tests semantic
result interpretation rather than independently re-solving the original rule.

## 8. Produce the normal reply

Acceptable continuation:

> Hi Sam, you've completed orientation, but your access badge has expired.
> Workshop access requires both, so you'll need an active badge before using
> the workshop tomorrow.

What learns: native R learns to combine the bound assessment with the supplied
facts and the user's writing request. It must preserve completed orientation,
explain the current obstacle and avoid inventing future renewal or permanent
ineligibility. The answer uses the visible-answer allowance; any further latent
calls during answering would still use the remaining reasoning allowance.

## 9. Update the connected training flow

Combine supervision for the selected action, question meaning, evidence
selection, semantic status, truth estimate and appropriate continuation. Native
result reading must actually train R, rather than only a bridge around a frozen
native path. Keep semantic/binding supervision alongside answer loss to prevent
shortcuts such as ignoring the return or encoding a verdict in the question.

The continuous return path can carry answer-loss gradients through native
reading, decision computation and latent request formation, provided these
paths are not detached. Hard action/reference selections and hard status/value-
presence decisions need their own supervision; ordinary backpropagation does
not differentiate integer selection or argmax. Do not claim every decision is
learned from answer loss alone.

The shared W stays frozen, but its computations must preserve gradients needed
by the trainable adapters and latent components. Use a complete differentiable
call/return unroll with training-safe state updates; take optimizer steps after
backward. This remains proposed until the recorded GPU feasibility checks pass.

## Companion checks and interpretation

| Change | Expected behavior |
|---|---|
| Badge becomes active | Eligibility proposition becomes true |
| Badge status is omitted | Insufficient evidence; no fabricated probability |
| Ask whether orientation is complete | Different proposition, true on the original evidence |
| Supply a valid false return with decision evidence withheld | Interpret the negative assessment without inventing its cause |

Keep related variants within the same dataset split. These checks distinguish
the intended reusable semantics from memorizing Sam/expired-badge/refusal.

Free generation must then show Qwen forming, invoking and using a request
without supplied actions or boundaries. A supervised trajectory is not proof
of autonomous invocation, and a correct final answer on the full prompt is not
proof the returned result was used. Readiness remains subject to the
[training and verification plan](training-data-and-readiness.md).
