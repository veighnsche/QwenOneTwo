# Independent architecture proposal — 2026-11-09

The user requested another agent to design the architecture, implementation plan
and training route from GIVEN / WHEN / THEN requirements, without knowledge of
the existing implementation. This delivery is a proposal, not an adopted design
or authorization to run its experiments.

- [Exact requirements brief](blind-architecture-requirements-2026-11-09.md)
- [Agent's original architecture and implementation/training plan](blind-architecture-proposal-2026-11-09.md)
- [Agent's concrete latent-target and training addendum](blind-architecture-training-addendum-2026-11-09.md)
- [Provenance and document hashes](evidence/blind-architecture-provenance-2026-11-09.json)

## Independence method

One agent was started with `fork_turns: none`, using `gpt-6.1-sol` at High effort
under the model-selection skill's major-architecture rule. No chat history was
forked. The task prohibited tools, repository/file access, browsing, other agents
and requests for implementation context. The agent explicitly reports no such
access in both outputs; this is its provenance declaration, not an independently
instrumented tool audit.

The coordinating agent distilled the brief from recorded target requirements and
the user's constraints. Fixed product constraints include the Qwen/Qev mode
requirement, one prefill/history, textless internal interface, operation semantics,
LM-head budget mapping, GPU execution and throughput requirements. Existing
tensor layouts, packet formats/sizes, projection choices, loss weights, training
artifacts, experiment results and performance measurements were not supplied.
Independence concerns the implementation design; the brief itself remains
reviewable for requirement selection and framing.

There were two follow-ups. One requested an honest provenance declaration. The
other asked, in GIVEN / WHEN / THEN form, how text annotations yield latent
targets, how owners/losses/gradient boundaries work, and how native request
formation learns that space. Neither supplied implementation or experiment facts.
Both follow-ups are recorded in the provenance file. The proposal and addendum
are preserved as received; the observations below are the coordinating agent's
separate review.

## What it proposes

The proposal uses one chronological model state with explicit Qev-active and
Qev-inactive positions. It treats correctness of that switching, the native
answer-entry transition, and full-context memory as feasibility gates.

It proposes source-grounded variable-length latent requests, immutable bindings,
typed results and native result consumption. A frozen offline foundation encoder
defines latent target coordinates from reviewed textual fields; human review
concerns the meaning and evidence, not vector coordinates. Reconstruction and
semantic contrast tests have separate roles.

Its training route first qualifies the interface and decision semantics, trains
the continuous budget readout separately, then trains native formation and
consumption against the frozen accepted decision system, followed by finite
autonomous trajectories. It explicitly stops gradients at that decision boundary
and uses direct request/pointer/semantic losses to train formation. This is a
design choice, not a claim that answer loss alone trains all components.

## Review points before adoption

1. **Put performance gates before costly fitting.** Milestone D calls for this,
   but the ordered final experiment list places throughput seventh, after learning
   experiments. Resolve that inconsistency in any executable plan. Representative
   complete training steps must demonstrate that host work keeps up with the GPU;
   a utilization snapshot or CPU cap does not meet the requirement.
2. **Keep proposed acceptance numbers provisional.** The agent suggests 95%,
   90%, MAE 0.10 and a two-point answering margin. They are design suggestions,
   with no supporting measurements or statistical justification yet. They do not
   replace existing acceptance criteria or authorize weakening retention checks.
3. **Do not treat a frozen encoder as semantic ground truth.** The addendum names
   the required reviewed fields, independent labels, leakage exclusions and
   contrast tests. Their artifacts, coverage and encoder usefulness still need
   verification before dependent fitting.
4. **Resolve the trainable shapes and loss contract before implementation.** The
   plan declares owners and loss families, but dimensions, adapter placements,
   codec normalization, loss coefficients and exact finite exposure counts remain
   future design choices. No claim is made that this is a ready-to-run trainer.
5. **Retain the explicit feasibility conditions.** Mode/state correctness,
   zero-extra-thinking closure, full required context on the selected hardware,
   reviewed supervision and autonomous generalization remain unproven. The
   proposal does not establish a probability of success.

## Scope and validation

Only planning documents and provenance were written. No code, runtime, existing
training plans, learned weights or acceptance definitions were changed. No local
model workload, new training data, native training, deployment or service change
was performed. RD-08's recorded outcome and RD-09's unstarted state remain.

Document checks cover the 22 GIVEN clauses, matching WHEN/THEN structure,
provenance hashes, local links and whitespace. GPU tests were not run for this
design-only task. Commit/push delivery is recorded in the current handoff.
