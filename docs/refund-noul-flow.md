# Refund prompt: proposed learned Noul flow

Status: design discussion, not an implemented or validated complete flow. No
further code or training is authorized by this document. The current prototype
does not establish general latent question formation or result interpretation.

This original semantic example is an appendix. The
[implementation specification](architecture-and-training-plan.md) and
[documentation map](README.md) own the current cross-component decisions.

## Human request

> Draft a reply to this customer: “My headphones arrived twelve days ago, and the left side has stopped working. Can I get a refund?” Our policy allows refunds for defective products within thirty days of delivery.

Qwen remains free to answer directly. The following describes what should
happen if it elects to consult Qev; this prompt is not a mandatory Noul label.

## End-to-end request and return

1. **Identify the needed decision.** Qwen recognizes that drafting the reply
   requires determining whether the reported defect and delivery age satisfy
   the supplied refund policy.
2. **Form a specific latent question.** The intended meaning is “Does this
   customer qualify for a refund under the supplied policy?” This sentence is
   an explanatory gloss or training annotation, not a string that must be
   generated during inference. A learned query producer creates a sequence
   of continuous vectors representing the proposition. The request contains
   an operation tag and bindings to this customer case and this policy.
3. **Evaluate the question against its evidence.** The evidence remains in
   context. With the decision LoRA enabled, Qev evaluates the latent question
   against that context and produces a Noul probability estimate. Stock Qev
   is not already trained to interpret this latent request format; the receiver
   needs alignment training too.
4. **Return a result bound to the question.** The typed result contains the
   Noul value and a reference to the pending request. True means this specific
   proposition holds; it does not universally mean “approve the refund.”
5. **Interpret and continue.** With Qev disabled, Qwen receives the result
   through learned input vectors and can attend to the question it asked.
   Its native execution path, adapted through LoRA, must learn how this
   question/result pair changes the continued reasoning and customer reply.

An appropriate illustrative reply is: “I’m sorry the headphones have failed.
Based on what you’ve described, they fall within our thirty-day refund policy.”
This does not imply a refund has been processed or that any external system was
accessed.

The runtime transports requests, manages adapter switching and maintains
bindings. It must not compose the semantic refund question or decide policy
eligibility on Qwen's behalf.

## Keep question semantics separate from the verdict

The query must specify what to determine, rather than merely smuggling in a
precomputed yes/no answer. The same evidence supports different propositions:

| Delivery age | Internal question | Correct Noul result |
|---|---|---|
| 12 days | Does the customer qualify under this policy? | True |
| 45 days | Does the customer qualify under this policy? | False |
| 12 days | Has the thirty-day window expired? | False |
| 45 days | Has the thirty-day window expired? | True |

These examples assume a reported defect and the supplied thirty-day policy,
without adding exceptions or claims about external legal rights. True for an
expired-window question and true for an eligibility question have different
implications for the reply.

Hold the question representation fixed while changing the relevant evidence;
Qev should reevaluate the evidence. Also vary the question while keeping the
case fixed. A generic, unspecified hidden vector does not establish which
question was asked.

## Training proposal

**Question formation and receiver alignment.** Use explicit questions in
training annotations to provide semantic targets for latent requests. Text is
allowed in training annotations; deployed calls remain textless. Train Qwen's
query-producing path and Qev's latent receiver on varied questions, policies,
cases and negations, with known decision labels. Final-answer loss alone does
not establish that the latent vectors represent the intended question.

**Result interpretation.** First train controlled reader exercises in which
the returned result supplies information otherwise absent from the exercise.
The reader learns the relationship between the question and its typed return,
including negated propositions. Then train complete natural customer-reply
trajectories with the evidence present. Qwen's native LoRA must receive updates
through result interpretation and continuation; fitting only a bridge around a
fully frozen native execution path does not establish this capability.

**Optional invocation.** Demonstrations can teach how valid calls work without
labeling every refund request as requiring Noul. Include successful direct
answers and successful delegated paths. Keep operation hints out of the human
prompt.

## Checks needed before calling the flow established

- Changing the latent question changes what Qev evaluates.
- Changing the evidence changes the answer to the same question when relevant.
- Qwen interprets the result relative to the question, rather than treating
  true/false as fixed customer-response templates.
- A conflicting return affects the continuation appropriately: Qwen may notice
  the conflict or reconsider, rather than blindly following it or ignoring it.
- Free generation retains the option of no call; annotated training boundaries
  are not evidence of spontaneous invocation.

The [latent layout](latent-representation-v1.md),
[memory/snapshot contract](memory-and-request-snapshots.md),
[typed statuses](result-statuses.md) and
[training/readiness plan](training-data-and-readiness.md) record the later design
choices. Exact codec/alignment preparation and mixed-state GPU feasibility
remain in the [implementation handoff](implementation-handoff.md); the original
discussion above is not a competing protocol definition.

## Research context

[Coconut](https://arxiv.org/abs/2412.06769) provides precedent for trained
continuous hidden-state steps. [Toolformer](https://arxiv.org/abs/2302.04761)
provides precedent for learning call formation and result use. Neither validates
this proposed shared-backbone Qwen–Qev interface.
