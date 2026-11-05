# Meeting invitation: proposed learned Choice flow

Status: design discussion, not an implemented or validated complete flow. This
document authorizes no code changes or training. Invocation remains optional.

This original semantic example is an appendix. Use the
[implementation specification](architecture-and-training-plan.md) and
[documentation map](README.md) for current protocol decisions.

## Human request

> Draft an invitation for a thirty-minute project review. Maya is available Tuesday at 10 or Wednesday at 2. Leon is available Tuesday at 10 or Thursday at 11. I’m available Tuesday morning or Wednesday afternoon.

For the walkthrough, interpret the times as thirty-minute slots in a shared
timezone. A real calendar event still needs calendar dates and a timezone;
neither is supplied here. Drafting an invitation does not create or send one.

## Identify the question and alternatives

Qwen identifies the need for a slot consistent with everyone's stated
availability. Human analysis of the supplied evidence is:

| Proposed slot | Maya | Leon | User |
|---|---|---|---|
| Tuesday 10:00–10:30 | Available | Available | Available |
| Wednesday 14:00–14:30 | Available | Not listed | Available |
| Thursday 11:00–11:30 | Not listed | Available | Not listed |

“Not listed” is not a factual assertion of unavailability. Tuesday is the only
slot confirmed by all three people's supplied availability.

A learned candidate producer would identify the proposed times, merge duplicate
mentions, and form a candidate bank. Each entry needs a latent representation
and a recoverable binding to its contents, such as weekday, start time and
duration. For copied alternatives, source references can bind entries to prompt
mentions; constructed alternatives need their derived attributes retained too.

Illustrative arbitrary identifiers are:

| Identifier | Retained meaning |
|---|---|
| c17 | Tuesday, 10:00, thirty minutes |
| c42 | Wednesday, 14:00, thirty minutes |
| c8 | Thursday, 11:00, thirty minutes |

Qwen produces entries and a learned stopping signal. The runtime stores entries
and assigns identifiers; it must not extract alternatives through handwritten
scheduling rules or decide which slot works. Candidate formation must not
precompute a winner. The availability table above is an explanation, not a set
of validity labels supplied by the caller to Qev.

## Form and evaluate a latent Choice request

The intended question is: “Which proposed thirty-minute slot is confirmed to
work for all three attendees?” The sentence is an explanatory gloss or training
annotation, not generated call text.

The request binds the latent question, candidate bank and availability evidence.
Freeze the bank for that request. Later additions or edits must not silently
change what an earlier return refers to.

With the decision adapter enabled, Qev evaluates the candidates against the
question and evidence. Its Choice readout produces scores and selects an entry.
In this example the intended selection is c17. Qev needs alignment training to
interpret the proposed latent questions and candidate representations; this
capability is not established by the existing checkpoint.

The readout may select an array position internally. The runtime mechanically
maps that position to the bank entry's stable identifier. It does not supply the
semantic scheduling decision.

## Interpret the return and continue

The typed return identifies the pending request and selected entry. An opaque
identifier has no inherent meaning: the return path must retrieve that entry
and supply its representation, contents, operation type and request binding to
Qwen.

Train Qwen's native LoRA to recover the selected entry's attributes and continue
the requested task. An illustrative draft is:

> Subject: Project review
>
> Let’s meet Tuesday from 10:00 to 10:30 for our thirty-minute project review.

The contract is not “option 2 means Tuesday.” It is that the selected entry's
meaning remains intact through formation, selection, retrieval and use.

## Training proposal

**Candidate formation:** use annotated alternatives and attributes to supervise
extraction, deduplication and construction. Vary counts, ordering, expressions
of time and which candidate is selected. Preserve exact attribute bindings.

**Question and selection semantics:** train the latent receiver on questions,
candidate banks, evidence and valid selections. Use the same bank with different
criteria so an entry does not always win. Question text can appear in training
annotations while deployed requests remain textless.

**Result interpretation:** update Qwen's native LoRA on complete request/return
continuations. Include controlled reader exercises with a bank and selected
entry but no availability evidence, so the invitation must follow the returned
entry. Then train complete natural flows with the evidence present.

**Optional invocation:** retain successful direct answers. The human prompt
must not automatically require Choice or contain operation hints.

## Checks and recorded protocol decisions

- Shuffle bank order and identifiers: the selected meeting time should stay the
  same. Stable identity must not collapse into array position.
- Change Leon's availability to Wednesday at 2 or Thursday at 11: the common
  slot should become Wednesday at 2.
- Give the isolated reader different selected entries: its invitation should
  express the corresponding entry's attributes.
- A conflicting return should affect the continuation appropriately, rather
  than being blindly followed or ignored.
- Missing availability must not be silently converted into a known fact.
- Handle multiple feasible alternatives without inventing user preferences or
  treating one arbitrary valid alternative as the sole correct selection.

The later [result-status contract](result-statuses.md) defines No feasible choice
separately from Insufficient evidence. Missing availability alone does not prove
all alternatives unsuitable. The availability evidence above supports Tuesday;
always taking the largest candidate score is not sufficient for other cases.
These statuses require training and are not assumed existing Qev behavior.

The [latent layout](latent-representation-v1.md),
[memory/snapshots](memory-and-request-snapshots.md) and
[data/readiness plan](training-data-and-readiness.md) record later design choices.
Exact generator/codec/alignment implementation and mixed-state GPU feasibility
remain in the [handoff](implementation-handoff.md). Shared weights do not prove
semantic compatibility.
