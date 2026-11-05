# Demo preparation: proposed learned Score flow

Status: design discussion, not an implemented or validated complete flow. This
document authorizes no code changes or training. Invocation remains optional.

This original semantic example is an appendix. Use the
[implementation specification](architecture-and-training-plan.md) and
[documentation map](README.md) for current protocol decisions.

## Human request

> Help me prepare for Friday’s product demo. Sign-up fails about once in five attempts, exporting sometimes takes thirty seconds, and two dashboard icons look blurry. Give me a practical preparation plan.

Qwen can answer directly, ask for missing context, or consult Qev. This prompt
does not mandate Score or one call per issue.

## Identify targets and a grounded rubric

The known evidence is a reported sign-up failure frequency, an occasional export
delay, and blurry icons. The run-of-show, repair effort, and available workarounds
are not supplied. Severity, probability of occurrence, and repair priority are
different dimensions and must not be silently merged.

A proposed scoring question is: “How disruptive would this issue be if
encountered during the live demo?” This is conditional disruption severity, not
failure probability. An illustrative rubric contains:

| Component | Meaning |
|---|---|
| Dimension | Disruption to the demonstration |
| Condition | The issue occurs during the live flow |
| Low anchor | No material interruption |
| Middle reference | Noticeable interruption, but the demo can continue |
| High anchor | An essential demo step cannot continue |
| Direction | Higher means more disruptive |
| Range | Continuous value from 0 to 1 |

Anchors explain a continuous scale; they are not categories or output bins.
These words are explanatory glosses or training annotations, not strings that
the runtime generates or inserts into an inference prompt.

## Form and evaluate a bound request

Qwen would create a latent rubric entry and issue entries with recoverable
identity. A Score request binds an operation, rubric, target issue and relevant
evidence. The rubric specifies what counts as disruption; evidence specifies
what happened. The rubric must not pre-encode a verdict such as “sign-up scores
high.”

Reuse a fixed rubric when comparing different issues under the same dimension.
If Qwen changes the rubric, its identity/version must change too. Qwen chooses
which issues to assess and whether to call at all; the runtime does not force
three calls because the prompt lists three issues.

With the decision adapter enabled, Qev processes the latent request and evidence.
A continuous regression readout produces the scalar. The receiver needs to be
trained to interpret this latent format; the existing checkpoint does not
establish that capability.

An illustrative return of 0.8 for sign-up under the rubric above means high
disruption if the failure occurs live. It is not an 80% failure probability,
80% confidence, 80% readiness, or an allocation of thinking tokens. The initial
thinking-budget score has a separate fixed meaning and mapping.

The typed return must identify its request, target and rubric along with the
value. A result lacking those bindings has no reliable interpretation.

## Interpret the result and continue planning

Qwen's native LoRA must learn to interpret the return using the same rubric and
scale. The reported failure frequency still matters to planning even though it
is not the dimension of this particular score. Repair effort and run-of-show
must not be invented from the severity value.

An illustrative continuation is: if account creation is shown live, prioritize
its reliability and prepare a backup account; rehearse the export step and keep
a prepared export available; address blurry icons after the demonstrated flows
are dependable. This is conditional planning, not a claim that a score alone
determines the repair order or timetable.

## Training proposal

**Rubric formation:** annotate dimensions, conditions, direction and anchors for
varied human objectives. Train Qwen's latent rubric producer and Qev's receiver
to communicate these semantics. The runtime validates structure and numeric
range; it does not implement rules such as “sign-up is always serious.”

**Rubric-conditioned scoring:** use continuous judgments under explicit rubrics.
Pairwise comparisons can constrain ordering but do not alone define an absolute
0–1 scale; anchors are also needed. Subjective judgments do not have one
universally correct decimal. The readout is scalar regression, not a softmax
over score levels or a probability-weighted categorical score.

**Result interpretation:** update Qwen's native LoRA through full request/return
continuations. Controlled reader exercises supply the rubric, target and return
without original issue evidence, so the return is needed to perform the task.
Then train natural planning trajectories with evidence present. Fitting only a
feedback bridge around a frozen native execution path is insufficient evidence
of learned result interpretation.

## Checks and recorded protocol decisions

- With evidence fixed, changing the rubric changes the dimension evaluated.
- Under conditional severity, changing only occurrence frequency must not
  automatically turn the score into an occurrence probability.
- Reversing scale direction changes the interpretation of a high value.
- The same number under different rubrics does not imply the same conclusion.
- Values stay bound to the correct target and rubric when memory entries or
  presentation order are changed.
- Incomplete evidence is not silently assigned zero, and 0.5 is not used as a
  generic uncertainty marker.
- Free generation can make no call, one call or several calls; these prompts do
  not become fixed routing labels.

The later [result-status contract](result-statuses.md) defines supported versus
non-assessed returns; absent evidence never automatically produces zero or a
midpoint. The [latent rubric layout](latent-representation-v1.md) and
[training/readiness plan](training-data-and-readiness.md) record the corresponding
design choices. Exact codec/alignment preparation, reviewed score labels and
mixed-state GPU feasibility remain in the [handoff](implementation-handoff.md),
not an undefined competing status/rubric protocol.
