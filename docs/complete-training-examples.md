# Complete training examples: index and shared conventions

Status: supervised human-review drafts, not approved gold data, generated latent
vectors or model results. The three complete walkthroughs below own the detailed
examples; this index avoids maintaining a second copy of their steps.
Documentation authorizes no coding, training or deployment.

## Complete examples

| Example owner | Human task | Intended request | Draft reference result |
|---|---|---|---|
| [Noul walkthrough](noul-training-walkthrough.md) | Reply about Sam's workshop access with an expired badge | Does Sam currently meet the supplied requirements? | Assessed, false; binary truth target p_true=0 |
| [Choice walkthrough](choice-training-walkthrough.md) | Recommend an available label printer meeting minimum throughput | Which supplied printer meets format, availability and rate criteria? | Assessed, Pine; index 2 in [Cedar, Birch, Pine] |
| [Score walkthrough](score-training-walkthrough.md) | Prepare for a recoverable printer jam during a packing shift | How disruptive is one described occurrence? | Assessed; proposed 0.30, interval [0.20,0.40], requiring independent review |

Each linked walkthrough preserves the full human prompt, evidence, latent
question/arguments, bound return, normal answer, per-component learning targets,
gradient boundaries and companion interventions. These are one selected route
per prompt; correct direct answers and other meaningful routes remain valid.
No individual prompt is required to invoke its associated operation.

## Shared execution and annotation contract

Only the human prompt is submitted as the user input. Questions, rubrics,
feasibility labels, reference outcomes and answers are separate offline review
annotations. Do not generate or insert their English glosses or JSON as a
second inference prompt.

Review aliases are not opaque identifiers Qwen must invent. Bind supplied facts
to actual immutable source-token spans when the dataset is materialized; do not
fabricate offsets, context positions, predicted scores or measured gradients.

| Example | Review aliases retained from the initial draft |
|---|---|
| Noul | case-sam-v1; evidence-orientation-v1; evidence-badge-v1; policy-workshop-v1; question-access-v1; request-noul-1 |
| Choice | printer-cedar-v1; printer-birch-v1; printer-pine-v1; bank-printers-v1; request-choice-1 |
| Score | issue-jam-v1; rubric-disruption-v1; request-score-1 |

Each sealed snapshot pins exact object versions, bank order/rubric meaning,
context frontier and model/protocol bundle. The runtime assigns identities and
validates pending returns; learned components select the semantic contents.
No gold verdict is supplied in the question, no winner/feasibility flag in the
candidate facts, and no desired score in the rubric.

The unified sequence is W+D initial prefill -> W+R native policy and draft ->
learned completion plus runtime seal/admission -> W+D evaluation -> W+R result
consumption and normal continuation. The gate selects operation intent; it is
not a second semantic confirmation at dispatch. There is one causal state
history, no independent question prefill and no second full model.

Count actual formation, receiver, return, retrieval-presentation and consumption
positions. Snapshot storage/lookup alone is not a model position. Actual token
lengths and costs await a specified codec and GPU execution, not estimates
presented as measured facts.

## Draft effort preferences, not operation budgets

| Example | Illustrative startup preference | S-curve predicted allowance |
|---|---:|---:|
| Noul | 0.04 | 612 |
| Choice | 0.05 | 950 |
| Score | 0.06 | 1359 |

These are unreviewed illustrative labels, not minimum sufficient budgets or an
operation-to-budget mapping. The example configuration uses A=160 and H=64;
actual context admission can reduce the granted allowance. General task Score
does not change it. Review preferences and subjective severity before gold-data
use. Actual predictions are produced by Qwen, never replaced by these labels
in free evaluation.

## Review and implementation use

Start with the [implementation specification](architecture-and-training-plan.md)
and [handoff](implementation-handoff.md). Use each walkthrough to review intended
semantics; use the [data/readiness contract](training-data-and-readiness.md) for
split/annotation rules and functional evidence. A scalar alone does not reveal
its supporting facts, and a successful guided example does not prove latent
meaning, free invocation, native result use or hybrid-cache compatibility.
