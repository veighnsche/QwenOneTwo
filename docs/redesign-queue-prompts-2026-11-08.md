# Redesign queue prompts — 2026-11-08

Copy each numbered prompt into a separate sequential queue item. This file is
instructions for future tasks, not authorization to execute them while preparing
the prompts. RD-01..07 implement and qualify without training. **Submitting
RD-08 or RD-09 authorizes the training explicitly described in that prompt.**
For a queue with no training, enqueue RD-01..07 only.

The application may advance its queue after an interrupted turn. These prompts
cannot change that scheduler. Instead, every item must recover its earliest
unfinished prerequisite before starting its own target. A genuine unresolved
blocker leaves dependent work unstarted, even if more queued items arrive.

## Required queue protocol

Applies whenever one of the prompts below is submitted:

1. Work in `/home/vince/QwenOneTwo` on `main`. Read `AGENTS.md`,
   `docs/next-agent-handoff.md`, `docs/learning-redesign-plan-2026-11-08.md`,
   this protocol and `docs/redesign-progress.md`. Inspect current code, git
   changes, receipts and relevant running jobs before acting.
2. Treat your requested RD number as the **last permitted slice**, not a command
   to jump to that slice. Inspect RD-01 through that target in order. Resume and
   finish the earliest undelivered prerequisite, then continue toward the target
   within the submitted prompt's authorization. Do not proceed beyond the target.
   The prompts explicitly authorize this predecessor recovery; do not ask routine
   permission to finish it.
3. The progress ledger is an index, not proof. Validate a delivered row against
   actual implementation, checks, evidence and pushed commits. Repair missing,
   stale or contradicted evidence before relying on that row. Reuse valid
   completed results when dependencies are unchanged. Do not rerun an expensive
   check just to reconfirm its checker.
4. If previous work is uncommitted, inspect and resume it without discarding
   unrelated edits. If a matching run is still active, reconnect to its logs and
   monitor it; never launch a second copy. Identify it by run manifest and actual
   process command, not an assumed PID. Never kill or modify existing services.
   For an interrupted fit, use a compatible complete-window checkpoint and exact
   optimizer/RNG/sampler state; retain completed valid arms and exposures. Missing
   recovery state is a blocker to report, not permission to fabricate progress.
5. Keep one small progress ledger. Mark a slice `in_progress` while working.
   Mark it `delivered` only when its currently due acceptance checks and evidence
   are present, the handoff is updated, and the work is committed and pushed to
   `main`. Record code/evidence paths, exact commands/results and implementation
   commits; verify push. Do not infer completion from a final chat message.
6. RD-03..06 deliberately defer their real-GPU assertions to RD-07. Their delivery
   requires implemented checks, passing applicable static tests and explicitly
   listed pending GPU check IDs. This permits implementation to advance, but is
   **not GPU qualification**. RD-07 must resolve every required deferred check
   before RD-08 can run. Do not introduce a dependency cycle that requires RD-07
   to be delivered before RD-03 can be implemented.
7. Track implementation delivery separately from qualification and learning
   outcome. RD-08 can be delivered with a fully recorded failed experiment.
   RD-09 additionally requires a specific receiver artifact that actually passes
   RD-08's useful-improvement acceptance and all technical prerequisites. A
   finished failed fit does not satisfy that condition.
   Likewise, RD-01/RD-06 can deliver complete audits that identify missing data
   or unsupported native branches. Record those gaps as limits on the affected
   later experiment, not as missing audit work. Required implementation/technical
   checks still must pass; a report of absent labels is not a waiver to train on
   invented labels.
8. When a prerequisite is genuinely blocked, first fix routine implementation
   issues within scope. If the blocker requires new data, broader training,
   unavailable hardware, changed acceptance or missing authorization, record the
   exact blocker, preserved progress and next recovery action in the ledger and
   handoff. Commit/push reviewable changes. Leave dependent work unstarted.
   Do not invent a waiver, relabel failure as success or start an automatic search.
   Independent work inside the blocked slice may continue; later dependent
   slices may not. A later queued item rechecks the same earliest blocker.
9. Existing data only; no agents, new generation, new semantic review calls,
   locked-set access, deployment or changes to The Operating Table/live services.
   Keep W frozen and internal call/return latent-only. Preserve one prompt,
   one prefill, one causal cache, bindings, accounting, continuous budget and
   existing numeric/semantic tolerances. Model execution is exclusively on
   RTX3090 UUID `GPU-225509c3-d702-6c27-de03-76b5e99b33b1`, never CPU/3060.
   Keep the reference immutable. Respect any newer explicit user restrictions.
10. RD-01..06 authorize implementation and appropriate model-free checks only;
    no model workload, fitting or optimizer updates. RD-07 additionally authorizes
    its specified GPU forward/backward qualification without optimizer updates.
    RD-08/09 permit only their explicitly submitted finite fits. No time/token/cost
    cutoff, automatic extra passes or capability claim from diagnostic fixtures.
11. Keep work small and coherent. Use existing runner/optimizer/checkpoint
    machinery; do not implement a queue manager, worker framework or repeated
    semantic approval layer. Update the current slice/evidence/blocker during
    work. If interrupted, the next item recovers from repository evidence.
12. At the end report: slices actually delivered, technical checks still pending,
    actual learning outcome if any, commit/push status, and earliest unfinished
    slice/blocker. Commit and push each coherent completed slice before moving
    to the next. Do not wait for permission already supplied by the prompt.

## Prompt 1 — RD-01: data coverage and supervision

```text
Work in /home/vince/QwenOneTwo. Implement RD-01 from
docs/learning-redesign-plan-2026-11-08.md. Apply the Required queue protocol in
docs/redesign-queue-prompts-2026-11-08.md, including predecessor recovery and
evidence-backed completion in docs/redesign-progress.md. Finish through RD-01 only.

Extend the existing metadata audit/report to create deterministic coverage,
comparison-panel and native-supervision manifests from the existing reviewed
training/development releases. Preserve splits/families, account for every
receiver row, separate assessed behavior from abstentions, and report familiar
operations versus unseen operator/range and wording/binding changes without
claiming confounded cohorts isolate one cause. Inventory writer/reader/interleaved
target eligibility and missing bindings. Never invent annotations or move dev
data into training. Reuse valid recorded audits.

Meet RD-01 acceptance, run relevant model-free tests, record exact evidence and
known gaps, update the ledger/handoff, commit directly to main and push. No model
execution, agents, new data or training.
```

## Prompt 2 — RD-02: one runner and resumable identities

```text
Work in /home/vince/QwenOneTwo. Implement through RD-02 from
docs/learning-redesign-plan-2026-11-08.md. Apply the Required queue protocol in
docs/redesign-queue-prompts-2026-11-08.md. Verify RD-01 against implementation,
checks, evidence and pushed commits; recover and finish it first if incomplete.

Extend the existing receiver runner with named input profiles and a model-free
prepare mode. Bind panels, dependencies, owners, objectives, initial state and
exposures in deterministic manifests. Reuse Window and checkpoint/resume; reject
incompatible profile/release/cache/checkpoint identities and ambiguous execution.
Distinguish plans, offline encodings, Sessions and actual updates. Preparation
must never launch a model or fit. Use small construction functions rather than
a new framework or another copied training loop.

Meet RD-02 acceptance and rejection tests. Document exact evidence, update
docs/redesign-progress.md and the handoff, commit each finished slice to main and
push. Stop after RD-02. No model execution, agents, new data or training.
```

## Prompt 3 — RD-03: stable frozen-W component features

```text
Work in /home/vince/QwenOneTwo. Implement through RD-03 from
docs/learning-redesign-plan-2026-11-08.md using the Required queue protocol in
docs/redesign-queue-prompts-2026-11-08.md. Verify all earlier deliveries and finish
the earliest incomplete predecessor before RD-03.

Add the explicit frozen-w-source-v1 offline source policy with zero adapter
contribution, qualified v3 arithmetic, complete permitted bytes and unchanged
source coordinates. Keep the old D-source policy and pooled role payloads as
named controls. Fingerprint every frozen cache dependency; never merge adapters,
replace weights or toggle shared training ownership to encode inputs. Native
states must remain their actual one-prefill states and graph.

Run relevant model-free policy/provenance/cache tests. Implement the 3090 tests
for D-independence, cache parity and consumer gradients, but record them as
pending RD-07 without running model workloads now. Meet the queue's staged
delivery criteria, update ledger/handoff, commit and push. Stop after RD-03.
No agents, new data, model execution or training.
```

## Prompt 4 — RD-04: richer latent packet

```text
Work in /home/vince/QwenOneTwo. Implement through RD-04 from
docs/learning-redesign-plan-2026-11-08.md. Follow the Required queue protocol in
docs/redesign-queue-prompts-2026-11-08.md; verify and finish incomplete predecessors
in order. RD-03 GPU assertions may remain explicitly pending RD-07.

First version the direct-role-states-v1 layout/admission contracts, then implement
the opt-in richer ordered role states and identity-initialized FP32 residual
projection. Reuse the runtime/registry/renderer. Define lengths, role order, rubric
subfields, writer capacity, candidate ends, return echoes and worst-case return
reservations together. Preserve exact attributes/provenance, old-profile behavior
and existing envelopes. Reject overlength requests before execution; report
coverage loss and never crop, hide excluded rows or silently raise allowances.
Runtime requests/results must contain latent packets, never annotation/tool text.

Run applicable model-free contract/rejection tests, implement and register pending
RD-07 GPU checks, record evidence, update owners/ledger/handoff, commit in small
slices and push. Stop after RD-04. No agents, new data, model execution or training.
```

## Prompt 5 — RD-05: joint objectives and binding

```text
Work in /home/vince/QwenOneTwo. Implement through RD-05 from
docs/learning-redesign-plan-2026-11-08.md. Apply the Required queue protocol in
docs/redesign-queue-prompts-2026-11-08.md; inspect evidence and finish all incomplete
predecessor deliveries first, retaining their explicit pending RD-07 checks.

Integrate existing status/Noul/Choice/Score and applicable feasibility/Reason
objectives for the new profile, using existing coefficients and eligible-example
accumulation. Implement honest experimental component admission and explicit
owner manifests without fabricating reference qualification or disabling guards.
Keep W, R, budget row and reference frozen in the proposed receiver fit. Preserve
actual-snapshot binding, outcome blindness, the documented post-forward Reason
exception and masks on unsupported writer semantics. Report joint delivered
behavior, per-cell statuses, eligible counts and startup retention; do not add
auxiliary count regression or tune on development.

Run relevant model-free leakage/binding/mask/ownership tests and register pending
RD-07 gradient/resume checks. Record evidence, update ledger/handoff, commit and
push. Stop after RD-05. No agents, new data, model execution or training.
```

## Prompt 6 — RD-06: connected native path and eligibility

```text
Work in /home/vince/QwenOneTwo. Implement through RD-06 from
docs/learning-redesign-plan-2026-11-08.md using the Required queue protocol in
docs/redesign-queue-prompts-2026-11-08.md. Verify earlier deliveries and complete
the earliest unfinished predecessor before starting RD-06.

Wire the new profile into native writer emission, target/binding contracts and
actual-return continuation using the existing EpisodeProducer. Use original
prefill states and one connected cache. Keep offline annotations/encoders and
gold returns out of the live invocation path. Preserve masks for arbitrary
unverified requests and incompatible continuations. Inventory eligible existing
formation/continuation/reader targets by operation and explain every unsupported
branch. Separate prepared components, supplied-return reader exercises and actual
native episodes. Prepare connected-gradient and real-return fixtures plus final
value/identity and change-pair evaluation, recording all forcing explicitly.

Run model-free checks only; register the GPU dry-run/gradient assertions for
RD-07. Missing valid supervision is a visible blocker, never a reason to invent
labels. Update ledger/handoff, commit and push. Stop after RD-06. No agents,
new data, model execution or training.
```

## Prompt 7 — RD-07: run technical qualification without training

```text
Work in /home/vince/QwenOneTwo. Implement and run through RD-07 from
docs/learning-redesign-plan-2026-11-08.md. Apply the Required queue protocol in
docs/redesign-queue-prompts-2026-11-08.md. Recover unfinished RD-01..06 deliveries
first, then resolve every required deferred GPU check; never start training.

Use only RTX3090 UUID GPU-225509c3-d702-6c27-de03-76b5e99b33b1, frozen W/reference,
existing data and disclosed fixtures. Verify stable source/cache behavior,
new-profile forward/VJP parity, actual native continuous gradients, all-status
admission/returns, cancellation, one prefill/cache, exact accounting, all 320 W
tensors/tying and export/resume/RNG/owner integrity. Exercise affected long traces
through 1024 positions. Preserve tolerances and qualified v3 computation. Reuse
unchanged numeric matrices; run changed-path coverage rather than certifying the
new interface from old results alone. No optimizer steps or CPU model fallback.

Record commands, executed source/profile hashes, every pass/failure/unsupported
case and counts. Repair routine failures within scope; a real unresolved technical
failure blocks RD-08. Update progress/handoff, commit and push. Stop after RD-07.
No agents, new data, training or service changes.
```

## Prompt 8 — RD-08: authorize and run the matched receiver comparison

**Training prompt: submit only when receiver training should be authorized.**

```text
Work in /home/vince/QwenOneTwo. This message authorizes only the finite RD-08
receiver comparison in docs/learning-redesign-plan-2026-11-08.md, including
recovering unfinished RD-01..07 work. Apply the Required queue protocol in
docs/redesign-queue-prompts-2026-11-08.md. Do not fit until every required RD-07
check has actually passed and its evidence is committed/pushed.

Run the two matched arms: identical frozen-W source policy, old pooled roles
versus direct role states, matched shared initialization/moments/seed/objectives/
exposure and disclosed profile-only initialization. Use existing receiver data
only: four passes over all 1,500 supported records per arm, 6,000 exposures/750
eight-example windows. If complete shared support is unavailable, record the
coverage blocker and do not silently substitute a smaller experiment. Keep W,
R, startup row and reference frozen. Only declared receiver owners may update,
on RTX3090 UUID GPU-225509c3-d702-6c27-de03-76b5e99b33b1. No CPU fallback.

Reconnect to a matching active run or resume compatible checkpoints instead of
duplicating work. Finish both fixed fits before development evaluation. Evaluate
joint status/value behavior, interior Score, Noul baselines, Choice retention,
Reason support and all declared E/X cohorts. No dev fitting/checkpoint selection,
new data/agents, extra passes or time cap. Record actual results even if acceptance
fails. Mark execution delivery separately from useful-improvement acceptance;
failure must block dependent native training. Update ledger/handoff, commit and
push. Stop after RD-08; do not launch RD-09 or promote/deploy weights.
```

## Prompt 9 — RD-09: authorize the conditional native learning demonstration

**Training prompt: submit only when native training should be authorized.**

```text
Work in /home/vince/QwenOneTwo. This message authorizes the finite RD-09 native
learning demonstration in docs/learning-redesign-plan-2026-11-08.md, conditional
on actual prerequisite acceptance. Apply the Required queue protocol in
docs/redesign-queue-prompts-2026-11-08.md. Recover incomplete predecessors within
their fixed scope, including RD-08 if it has not run. A delivered but unsuccessful
RD-08 is a blocker: report it and do not rerun/search for a better receiver or
start native training. Verify the accepted receiver artifact and all RD-07 checks.

Use only the existing, validly bound writer/reader/interleaved/direct/replay data
identified by RD-01/06. Before execution, fix and commit the exact deterministic
eligibility, owners, coefficients, guided-to-unforced exposure schedule, finite
pass/update counts and evaluation manifest based on training metadata, not dev
results. Choose routine schedule details autonomously from the existing training
profile; do not create an open-ended fit. Unsupported or zero eligible paths
remain blockers. Keep W, accepted D/receiver and reference frozen; train R and
declared writer/reader owners, freezing every parameter shared with the receiver.

Run on RTX3090 UUID GPU-225509c3-d702-6c27-de03-76b5e99b33b1, never CPU. Reuse active
runs/checkpoints. Preserve actual latent calls/returns, continuous gradients and
one prefill/cache. Evaluate held-out native result use, Noul true/false, Choice
binding/order, interior Score, available abstentions/change pairs and no-call
behavior. Disclose forcing; learned invocation requires unforced evaluation.
Retain full dev failures and unsupported coverage. No agents, new data, new
review calls, automatic extra passes, tolerance changes or deployment. Record
actual outcome, verify frozen-receiver retention, update ledger/handoff, commit
and push. Stop after RD-09 even if the demonstration fails.
```
