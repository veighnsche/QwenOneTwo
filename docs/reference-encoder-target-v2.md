# Target-private reference pooling revision v2 — 2026-11-06

The user authorizes fixing encoder/pooling learning and proving recovery with
existing data on the RTX 3090. No new source data, agents, capability training
or live-service changes. No qualification is claimed before measurement.

## Measured cause and smallest architectural change

Five mixed-role six-case probes leave Target 0001 incorrect: ordinary CE 5/6,
dense latent MSE 4/6, field-value memory 5/6, 5x LR 5/6 and focal CE 5/6. None
received a full block. All sources/results are retained; rejected candidates
are removed from active fitting code.

The separately bounded Target-only probe **recovers both Targets at final
window 64**, while Question/Key behavior regresses (3/6 total). It uses the same
parent, frozen decoder, ordinary CE, LR/schedule and 64 updates. This supports
cross-role interference in shared pooling for these training cases; it does not
prove a globally unique root cause or held-out competence.

Start with the demonstrated interference only: **give Target a private small
pooler; retain a shared pooler for the other nine roles**. Both begin as exact
copies of the original shared module's memory projection, two cross-attention
blocks and output lift. Split its 30 learned slot queries into 28 non-Target
queries and two private Target queries, preserving each role's original order.
There are still exactly 30 active query vectors. Executing one role still uses
one 1024->256 memory projection, two eight-head/32-wide blocks, one 256->1024 lift
and canonical N. Extra small projection/block weights are explicit; W, T and
wire codec are not duplicated. No additional positions, prompt, runtime channel,
raw magnitude or current outcome field is introduced.

`Reference.split_pooler()` adapts a verified old tensor artifact as initialization,
not as qualified new behavior. `target-v2` has distinct registered tensor keys
and fingerprints; strict loading rejects the wrong structure. Frozen exports
record `pooler_kind`; qualification constructs the matching class. Full fitter
checkpoints use actual partitioned ownership/state and pin the fitting profile
in source identities. Default/historical constructors remain shared-v1.

## Implementation and fixed six-case acceptance

Require a GPU check that all ten role outputs are **exactly equal at migration**,
30 total queries remain, parameter storage is disjoint, Target losses have no
non-Target pooler gradients, decoder/live gradients remain absent and frozen
export/load restores the exact outputs/fingerprint. Validate the reverse
non-Target/Target ownership path during the actual learning probe as well.

Reset to original parent tensor fingerprint
`d0d4aef44b4415beb19311350eb24533190e1a27406c5d6f59a15a5a57402dec`, split pooling,
freeze T/codec, and run the same **64 Window updates / 512 exposures**, six train
IDs, exposure order, ordinary CE, seed 17, 2e-4 peak LR, A4 schedule, clipping,
decay, eligibility and 0/16/32/64 observations as the mixed-role control.
Require final **6/6 exact**, finite eligible updates, unchanged decoder/live/base
and verified role-local gradients. Do not increase steps, lower acceptance,
select an earlier observation or continue failed probe weights. A fitted six-case
result cannot qualify development or the learned flow.

## Conditional full reference qualification

Only after acceptance, discard probe weights and reset to the exact old parent,
repeat the deterministic split, and start fresh pooler-only AdamW. Explicit
`--profile encoder-target-v2` runs one fixed **1,000-window / 8,000-exposure**
block over the same 3,376 reviewed training views with the existing exact
four-diversity/four-role-balanced sampler, seed 17. Ordinary reconstruction CE,
coefficient 1.0, masked absent explicit contrasts, per-example/eligible-example
normalization, LR/decay, finite guards and clipping remain unchanged. T/codec
stay fixed, live W/R/D/codec unchanged; F7/Reason and CUDA feature caching apply.
No development/locked fitting, new source annotations or model-assisted reviews.

Save/restore checkpoint zero, checkpoint/evaluate every 100 windows, and require
**>=95/100 original development exact recovery and >=90% contrasts at BOTH
900 and 1,000**. No earlier-checkpoint substitution, gold repair, tolerance,
threshold, denominator or case changes. Verify all exposures, role-local active
owners, exact decoder/live hashes, all 320 frozen base tensors/tying and original
release/source hashes. Retain actual small-weight parameter/memory measurements;
this is not qualification of a full participating capability trainer. Record
failure honestly; no automatic extra epochs or capability training after failure.
