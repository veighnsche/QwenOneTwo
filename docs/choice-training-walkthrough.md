# Complete Choice training walkthrough: printer selection

Status: recorded explanation of the proposed supervised training trajectory.
This is not a model run, approved gold data, or a claim of trained capability.
No coding, training, model execution, service changes or deployment are
authorized by this document. Model workloads remain GPU-only.

This expands [the complete training examples](complete-training-examples.md)
to identify what learns at each step, following the
[Noul walkthrough](noul-training-walkthrough.md). The shared backbone W stays
frozen; R is the trainable native adapter; D is the decision adapter initialized
from Qev and aligned to the latent protocol. Native mode is W+R and decision
mode is W+D, with mutually exclusive adapters on one shared backbone.

This walkthrough owns its example-specific teaching targets. The
[implementation specification](architecture-and-training-plan.md) owns dispatch
timing/shared contracts; the [handoff](implementation-handoff.md) lists remaining
codec/loss choices. Pretrained W is frozen; reserved scalar rows are explicit
learned extensions with validated gradient/tying handling.

## 1. Ordinary human request

> Tell the packing team which printer to use this afternoon. We need an
> available printer that produces self-adhesive shipping labels at a rate of
> at least 100 labels per hour. Cedar prints those labels at 80 per hour and is
> available. Birch prints them at 140 per hour but is unavailable this
> afternoon. Pine prints them at 120 per hour and is available.

Only this request enters as the human prompt. Candidate annotations, the
correct selection and the reference answer stay in the separate training
record. No operation hint, correct pointer or gold feasibility label is added
to the inference input.

## 2. Single prefill and continuous startup budget

W+D prefills the prompt once and produces the fixed initial continuous
thinking-budget score. The draft effort target is s=0.05, mapping through the
recorded S-curve to B_pred=950 reasoning positions, subject to context admission.
The example configuration reserves A=160 visible-answer positions and H=64
restricted control positions.

What learns: the startup budget readout and its trainable decision path learn
to predict the annotated effort preference. This remains a review placeholder;
it does not mean Choice always receives 950 positions or that this task needs
that amount. The runtime performs the mapping and limit enforcement.

Select native W+R while retaining the same causal state. The fixed startup
budget port remains distinct from general task Score requests.

## 3. Identify the task and its actual constraints

Native Qwen recognizes that the user wants a recommendation among the listed
alternatives. The supplied criterion requires all of:

- Availability this afternoon.
- Self-adhesive shipping-label output.
- At least 100 labels per hour.

What learns: native R learns to extract the actual criterion. At least 100
includes exactly 100. Fastest printer would be a different criterion, which
the user did not request. Do not invent a date or timezone for this afternoon.

The selected supervised trajectory uses Choice. Dispatch occurs only after
the candidate bank and question are complete. A correct direct answer remains
acceptable outside this selected trajectory; do not impose domain-to-operation
routing at inference.

## 4. Form alternatives and the latent question

The learned candidate writer constructs:

| Stable candidate | Label format | Rate | Availability |
|---|---|---:|---|
| Cedar | Self-adhesive shipping labels | 80/hour | Available |
| Birch | Self-adhesive shipping labels | 140/hour | Unavailable |
| Pine | Self-adhesive shipping labels | 120/hour | Available |

Each candidate has four learned 1024-wide vectors, recoverable exact attributes
and source references. Bind each attribute to the correct supplied clause.
Stable object identities are runtime bookkeeping; Qwen learns to form the
contents and select relevant references, rather than invent arbitrary UUIDs.

The intended question is:

> Which listed printer is available this afternoon and meets the supplied
> label-format and minimum-throughput requirements?

That sentence is an offline annotation. At inference, the learned question
writer expresses its meaning through eight 1024-wide vectors, without generating
an English tool question or JSON. It must not simply encode choose Pine.

What learns: native R, candidate formation, reference selection and question
formation learn to include all supplied alternatives, keep attributes attached
to the correct printer, preserve exact numbers and availability, stop after
the supplied alternatives, and express the criterion faithfully.

Feasible and correct winner are separate training labels, not candidate facts
supplied to Qev. The gate receives supervision for Choice intent, starting a
native draft. Learned writer completion triggers runtime seal/admission; there
is no second semantic gate at dispatch. Free evaluation removes supplied
actions/boundaries.
Semantic encodings and alignment objectives still require implementation and
validation; this document does not invent vector coordinates or claim learned
compatibility already exists.

## 5. Freeze the bank and enable Qev

Use the ordered bank:

    Position 0 -> Cedar
    Position 1 -> Birch
    Position 2 -> Pine

The runtime seals that order, candidate versions, question and evidence into
an immutable request snapshot. It assigns request identity, checks structural
validity and admits the bounded completion against the remaining allowance.
Then it selects W+D on the same attention/convolution/recurrent state lineage.

What learns: nothing from assigning identities, freezing order or enforcing
limits. Those are deterministic runtime operations. Qwen learns the semantic
contents and bindings; the runtime does not decide printer suitability.

There is no independent question prefill, cache reset or second full model.
Charge formation, receiver, return and consumption under the
[budget contract](position-limits-and-budget-accounting.md). The alternating
history remains subject to the unverified
[cache/gradient feasibility gate](cache-and-gradient-strategy.md).

## 6. Evaluate feasibility and select a candidate

Reference assessment:

| Candidate | Reference assessment |
|---|---|
| Cedar | Fails minimum throughput |
| Birch | Fails availability |
| Pine | Meets all supplied requirements |

Reference return:

    Semantic status: Assessed
    Selected candidate: Pine
    Pointer training target: 2 (zero-based in this frozen bank)

What learns: aligned Qev receiver/readouts learn to understand the latent
criterion, evaluate the actual alternatives and select a feasible one. Candidate
scores and a selection loss target index 2 in this particular bank. This is a
Choice pointer, not a continuous generic Score output.

Feasibility and semantic status need separate supervision. A largest pointer
score does not establish that any candidate is acceptable. No feasible choice
requires a supported absence of feasible entries, with no selection. Missing
evidence must not become known infeasibility. See [result statuses](result-statuses.md).

The actual model result remains observable. A wrong but structurally valid
selection is a semantic failure; the runtime does not substitute the gold winner.
Supplied gold returns used in isolated reader exercises are explicitly marked
as supervised rather than autonomous whole-flow execution.

## 7. Resolve the selection and interpret it in native mode

The runtime maps index 2 through the sealed bank to Pine's stable entry identity.
It validates the request binding, retrieves the selected entry representation
and recoverable contents, selects W+R and delivers the return once.

Human-readable rendering of the intended return contents:

    Selected alternative: Pine
    Format: Self-adhesive shipping labels
    Rate: 120/hour
    Availability: Available this afternoon

This is an explanation for review, not generated inference-time tool text.

What learns: result encoding and native R learn to interpret the selected
alternative and use its actual attributes. Qwen must not memorize index 2 as
Pine; the index means whichever entry occupies that position in the sealed bank.

The selection supports a recommendation, not a claim that booking occurred.
Explaining why other printers failed requires their evidence; the selected
identity alone does not contain those reasons.

## 8. Produce the normal answer

Acceptable continuation:

> Use Pine this afternoon. It's available and prints the required self-adhesive
> shipping labels at 120 per hour, exceeding the 100-per-hour minimum.

What learns: native R turns the bound selection into a useful normal answer,
preserving the selected name, attributes and actual user criterion. It should
not call Pine the fastest listed printer; Birch's stated rate is higher.

Visible text uses the answer allowance. Any further latent call while answering
would use the remaining reasoning allowance, which does not reset between phases.

## 9. Update the connected training flow

Combine supervision for:

- Candidate completeness, attributes, source bindings and formation stopping.
- Question/criterion fidelity and Choice invocation.
- Feasibility, semantic status and selected index.
- Correct native interpretation and final continuation.

The backbone W stays frozen; adapters and protocol components learn. Native
result consumption must actually update R, rather than only a bridge around
an entirely frozen native path. Use the recorded complete differentiable
training unroll and preserve required state/activation gradients.

Ordinary answer-loss backpropagation does not pass through the hard selected
index to teach which candidate should win. Selection needs its explicit loss.
Connected continuous representations and causal state can carry gradients
where preserved; this does not imply a derivative through integer lookup,
argmax or hard status decisions. Do not claim answer loss alone teaches every
switch and selection.

## Companion checks and interpretation

| Change | Expected behavior |
|---|---|
| Reorder bank to Pine, Cedar, Birch | Select index 0; selected identity stays Pine |
| Make Birch available | Birch or Pine is acceptable under the original criterion |
| Ask for the fastest available qualifying printer, with Birch available | Select Birch |
| Reduce Pine below 100/hour while retaining other facts | No feasible choice |
| Omit Pine's availability while retaining other facts | Insufficient evidence |
| Change candidate names and contents | Follow the actual entries rather than memorized names or indices |

Keep related variants in the same split. An isolated reader exercise supplies
the selected entry without the original bank and checks accurate interpretation
of those returned contents. It does not establish selection ability by itself.

Free whole-flow evaluation must show Qwen forming a bank, invoking Choice,
receiving Qev's actual selection and using it without supplied actions or
boundaries, within the same inference run. A guided route or correct final
answer alone does not establish autonomous invocation and result use. Follow
the [training and readiness plan](training-data-and-readiness.md).
