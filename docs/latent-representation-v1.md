# Version-one latent representation proposal

The opt-in [direct-role-states-v1 contract](direct-role-states-v1.md) versions
variable request-role lengths, native emission capacity and complete return
reservations together. The rules below remain the legacy default; the new
profile is experimental and its GPU assertions are pending RD-07.

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

Owner: semantic field layout. The
[implementation specification](architecture-and-training-plan.md) owns the
cross-component lifecycle. The [packet codec](latent-packet-codec.md) records
request/result framing and typed-value presentation. The
[final audit](final-decision-audit.md) records A1-A4 and
[B1's exact neural feature map](neural-input-encoding-v1.md), while keeping
implementation and GPU evidence distinct from recorded design.

The [writer schedule](latent-writer-schedule.md) now records how native steps
emit parallel field slots, references and completion. The
[semantic method](semantic-supervision.md) and
[loss configuration](loss-configuration-and-training-schedule.md) are recorded;
the [typed ABI](typed-value-abi.md) fixes value-head/body bounds. The
[model profile](model-configuration.md) records dimensions and normalization;
implementation and GPU evidence remain unverified.

## Core representation

Use typed latent records: short sequences of learned continuous vectors carry
meaning, while exact references and values preserve identity and precision.
Do not assume an undifferentiated current hidden state is a specific question,
or hide exact times and identifiers solely inside approximate vectors.

Each soft input vector has 1024 values for the pinned Qwen3.5-0.8B model.

| Object | Semantic representation | Exact accompanying data |
|---|---|---|
| Question | Eight vectors | Operation type, target and evidence references |
| Alternative | Four vectors per entry | Stable entry identity and recoverable attributes/references |
| Scoring rubric | Five semantic fields of two vectors each | Rubric identity/version, numeric range, direction and optional-field flags |

Field roles are fixed interface syntax. Questions, alternatives and criteria
inside those fields are learned content. Fixed syntax does not authorize
handwritten semantic routing or domain-specific answer rules.

Textless inference uses vectors, indices, scalars and typed metadata, without
generating a tool-question string or JSON. This does not eliminate runtime
memory, require all data to live in weights, or make latent computation free.

## Question record

The eight vectors collectively represent the question, qualifications and
negation. Individual dimensions do not receive assigned English meanings.
The native learned writer produces these vectors at inference.

For the refund flow the intended meaning is “Does this customer qualify under
the supplied refund policy?” The case and policy remain separately referenced
evidence. The question must not merely encode the expected answer “yes.”

Checks must vary evidence while retaining the question, and vary the question
while retaining evidence. For example, changing delivery age must permit a new
answer to the same eligibility question; asking whether the refund window has
expired is a different proposition about that same case.

## Alternative record

Four vectors describe one alternative. A stable identifier and exact attributes
or source references keep its contents recoverable through selection.

An illustrative meeting entry retains:

- A reference to the supplied Tuesday.
- Start time: 600 minutes after midnight.
- Duration: 30 minutes.
- Timezone: unspecified.
- Stable candidate identity.

Qwen produces bindings and values. The runtime stores and validates their
structure, without deciding participant availability. Copied alternatives can
refer to source contents; constructed alternatives must retain derived values
too. References to approximate semantic similarity are not a substitute for
recoverable exact attributes.

A return retrieves the selected entry and its contents. Do not deliver only an
arbitrary identifier or normalized array position and expect Qwen to infer the
meaning. The [reference and immutable snapshot design](memory-and-request-snapshots.md)
records the next layer of this contract; it is not an implemented capability.

## Rubric record

Use two vectors for each field:

1. Dimension being measured.
2. Conditions under which it is measured.
3. Zero anchor.
4. One anchor.
5. Optional midpoint reference.

Exact metadata carries the normalized range, scale direction, rubric version
and whether the midpoint is present. Anchors describe a continuous scale;
they do not create score categories or output bins.

For the demo flow: dimension is disruption severity; condition is encountering
the issue live; zero means no material interruption; one means an essential
demo step cannot continue; the optional midpoint describes a noticeable but
recoverable interruption. Higher values mean more disruptive under this rubric.

Reuse one rubric when comparing targets under the same criterion. A semantic
change creates a new rubric version. Returns identify both target and rubric.
The number does not become a probability, confidence estimate or thinking-token
allocation merely because it is between zero and one.

## Establishing shared meaning through training

The [semantic-supervision design](semantic-supervision.md) now owns the exact
allowed inputs and training sequence: answer-blind reference encoding, semantic
reconstruction/contrast tests, frozen targets, writer alignment and receiver
grounding. The summary below does not define a separate competing method.

1. Create inspectable annotations with explicit questions, alternative contents
   and rubric fields. Exclude verdicts from question inputs and desired scores
   from rubric descriptions.
2. Train semantic encodings and the receiver using separately supplied evidence
   and known operation outcomes. Reconstruction and similarity objectives can
   assist, but are not sufficient evidence of the intended semantics.
3. Train Qwen's native LoRA and writer to form these records from ordinary task
   context. Operation hints remain absent from the human prompt.
4. Train Qwen's native result reader to interpret returns using the question and
   retrieved memory contents, rather than fitting only a bridge around a frozen
   native execution path.

Both adapter modes use shared field-role and input-projection components. There
is no second deployed language model serving as translator. Shared base weights
or vector dimensions do not establish shared understanding without training.

Source evidence or exact attributes may be retrieved through the versioned
codec, but must not become an independent full-prompt replay. Any bounded
presentation into the existing state is charged; references alone do not prove
that the receiver obtained their contents. Freeze evidence/attribute encoding
and return retrieval with the packet codec in G0.

Readiness requires question/evidence interventions, candidate permutations,
exact selected-attribute recovery, rubric changes and result-dependent native
continuations. Eight/four/ten vectors must be revised if those capacities prove
insufficient; fitting the tensor shapes is not success.

## Scope boundaries

The initial thinking-budget port remains separate and retains its fixed startup
meaning. This representation concerns later Noul, Choice and general Score
requests. The [memory/reference and snapshot contract](memory-and-request-snapshots.md)
is recorded separately, as is the [typed result-status contract](result-statuses.md).
The [position and budget contract](position-limits-and-budget-accounting.md)
records compute-position accounting and initial limits. The
[cache and gradient strategy](cache-and-gradient-strategy.md) records the
proposed training path; mixed-adapter GPU feasibility remains unverified.

Learned continuous input embeddings have precedent in
[prompt tuning](https://arxiv.org/abs/2104.08691) and hidden-state inputs in
[Coconut](https://arxiv.org/abs/2412.06769). Neither establishes the proposed
record layout, slot counts or Qwen–Qev interface.
