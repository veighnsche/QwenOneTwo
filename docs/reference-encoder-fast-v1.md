# Stationary encoder update-size revision v1 — 2026-11-06

This is a separately bounded reference encoder-learning revision under the
user's authorization. Existing source data only; RTX 3090 only; no agents,
capability training or live-service changes. It does not relax qualification.

Three completed 64-window controls are retained: frozen-decoder ordinary CE
finished 5/6; adding dense latent MSE finished 4/6; field-value-only memory
finished 5/6 with the same Target confusion. All failed their six-case training
acceptance, so none received its conditional full block. The field-value candidate
is removed from active code; its archived source, patch, tests and results remain.

The ordinary-CE control still had decreasing CE and unclipped gradient norm
0.20057842135429382 at window 64. It used 50 warmup windows of the 1,000-window
reference schedule, leaving only 14 windows at near-peak LR. These observations
justify testing update size; they do not establish a disconnected encoder,
scaffold-only root cause, need for new data or mathematical impossibility.

## Fixed causal comparison

Reset to the original parent reference
`d0d4aef44b4415beb19311350eb24533190e1a27406c5d6f59a15a5a57402dec`.
Keep full-v1 annotation memory, original pooler architecture, frozen T/codec,
ordinary causal reconstruction CE, seed 17 and the exact same six train IDs,
64 Window updates / 512 exposures, exposure order, schedule shape, eligibility,
clipping/decay and 0/16/32/64 observations as the ordinary-CE control.
**Change only the pooler peak LR from 2e-4 to 1e-3.** T/codec/live parameters
are not optimizer owners. No dense MSE, field-value mask, extra probe windows,
random restart, threshold change or continuation from a failed probe.

Require final **6/6 exact training recovery**, finite eligible gradients and
unchanged T/codec/live/base before a full block. Retain actual outcomes and
separate physical update counts even on failure. No full block follows failure.

## Conditional existing-data qualification

If this acceptance passes, discard probe weights, reset again to the exact
parent and use fresh pooler-only AdamW. Explicit `--profile encoder-fast-v1`
runs one fixed **1,000-window / 8,000-exposure** block on the same 3,376 reviewed
training views and exact existing sampler. Only pooler peak LR differs from the
stationary-decoder recipe; ordinary CE, normalization, reconstruction coefficient
1.0, masked absent contrasts, seed, role allocation, schedule shape, finite
guards, global clipping and frozen decoder remain as documented. Preserve F7
and the Reason exception. No development or locked fitting; no new annotations.

Save/restore checkpoint zero; checkpoint/evaluate every 100 windows. Require
**>=95/100 exact original development recovery and >=90% contrasts at both
900 and 1,000**. No earlier-checkpoint replacement, case substitution, threshold,
denominator or tolerance changes. Verify all 8,000 exposures, pooler-only owners,
exact decoder/live hashes, 320 frozen base tensors/tying and original releases.
The six-case probe does not qualify generalization. Record failure honestly;
do not extend automatically or launch capability training after failure.
