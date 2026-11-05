# Natural invocation prompts

These are ordinary human requests that could give Qwen a reason to invoke an
internal decision operation. They are proposed examples, not validated results
or required routing labels. The complete learned request-and-return flow remains
unfinished.

These prompts have been discussed during design. Keep them and close
paraphrases out of training/tuning, but report their outcomes as illustrative
rather than fresh independent held-out evidence.

Only the quoted request is the human prompt. Do not supply the headings or
operation hypotheses to the model. Do not insert tool instructions, force a
gate action, or require a particular call. Qwen may answer directly, invoke a
different operation, or invoke several operations.

## 1. Customer reply

> Draft a reply to this customer: “My headphones arrived twelve days ago, and the left side has stopped working. Can I get a refund?” Our policy allows refunds for defective products within thirty days of delivery.

Possible internal use: **Noul**, to determine whether the customer qualifies
for a refund. This is a hypothesis, not a mandatory invocation.

See the [proposed refund request-and-return flow](refund-noul-flow.md).

## 2. Meeting invitation

> Draft an invitation for a thirty-minute project review. Maya is available Tuesday at 10 or Wednesday at 2. Leon is available Tuesday at 10 or Thursday at 11. I’m available Tuesday morning or Wednesday afternoon.

Possible internal use: **Choice**, to select the meeting slot. This is a
hypothesis, not a mandatory invocation.

See the [proposed meeting request-and-return flow](meeting-choice-flow.md).

## 3. Demo preparation

> Help me prepare for Friday’s product demo. Sign-up fails about once in five attempts, exporting sometimes takes thirty seconds, and two dashboard icons look blurry. Give me a practical preparation plan.

Possible internal use: **Score**, to assess how serious each issue is before
forming the plan. This is a hypothesis, not a mandatory invocation, and does
not require numerical scores in the human-facing answer.

See the [proposed demo preparation request-and-return flow](demo-score-flow.md).

## Intended use

Use these requests to work out how Qwen would identify a decision, represent its
question and criteria internally, receive a result bound to that request, and
use the result in its continuation. They do not establish that the current
prototype has those capabilities. Do not automatically turn the operation
hypotheses into supervised gate labels.

The [unified architecture and training plan](architecture-and-training-plan.md)
consolidates the target flow. The [implementation handoff](implementation-handoff.md)
records the remaining contract-freeze and GPU-feasibility work before full training.
