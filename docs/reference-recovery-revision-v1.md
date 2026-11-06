# Reference recovery development revision v1 — 2026-11-06

This is a preregistered reference-only development revision under the user's
request to diagnose/fix failed recovery using existing data. It does not grant
qualification, authorize capability training or change the recovery thresholds.
No new source annotations, model-generated records, reviews or approvals are
created. Existing training/development/locked releases remain byte-identical.

## Measured cause and implementation gaps

The previous 1,000-update artifact recovered 0/100 development RF records. New
read-only RTX 3090 diagnostics deduplicate these to 50 canonical meanings and
measure **0/50 development versus 31/50 training** exact recovery. Training's
500 selected reference rows contain only five distinct meanings per role; the
RF development meanings are all different. Every recorded development mismatch
occurs after matching the JSON scaffold, at content tokens.

The previous mean training CE (last 100 windows: 0.046998847847844444) therefore
did not establish usable reconstruction. Correct-versus-swapped same-role
prefixes barely change teacher-forced CE for several roles: mean training loss
differences are 0.0003760494 alternative, 0.0012271263 target and 0.0025907476
Reason. Those roles recover 0/5, 1/5 and 2/5 training meanings respectively.
Other roles do depend on their prefixes and recover all five training meanings.
This is partial conditional/template overfitting and failed transfer, not proof
of zero information in every latent field or an off-by-one decoder bug.

A mechanical inventory finds **3,376 unique permitted annotation views** already
available in the selected reviewed training release: question 1,114; target 869;
alternative 1,055; dimension 7; conditions 102; zero 7; one 7; midpoint 5; key 5;
Reason 205. The fitter previously ignored those additional existing role fields.
Byte uniqueness is not a claim of independent semantic scenarios. Sparse roles
remain sparse; broader reuse is a hypothesis to test, not proof of generalization.

## Fixed revision

1. `reference_data.catalogue` reads only the loaded, pinned training release.
   Reuse standalone reference annotations, existing supervised request fields,
   existing per-call request targets and explicit supervised Reason. Every view
   binds its original row hash and field address. Canonical equal views deduplicate
   while retaining their sources. Request builders never traverse result labels,
   chosen winners, scores, native continuations or effort. Outcome-bearing content
   remains confined to the documented non-assessed Reason exception.
2. Each eight-exposure window contains four deterministic diversity draws and
   four role-balanced draws. Each diversity cycle covers every distinct view;
   the balance prevents sparse roles from disappearing. Seed is 17. Exact sampler
   resume pins catalogue/release hashes and replays the exposure cursors.
3. Alternate ordinary next-token teacher forcing and **target-history ablation**:
   zero the preceding-target input vectors for four of eight exposures. Target
   labels, normalization, positions and latent role/slot prefix remain unchanged.
   This deterministic fitting exposure forces a stronger latent-conditioned
   reconstruction task without source text in the decoder prefix. The other
   four exposures retain ordinary teacher forcing. There is no random model
   dropout, future-target input, new token vocabulary or production protocol field.
   Decoder/evaluation behavior remains normal causal autoregressive recovery.
4. Warm-start only the previous unqualified reference tensor fingerprint
   `fb38a31359a4ff367780bb8729695894e663810efd124c016bc2d5ba3fdbf7de`.
   Begin a **fresh reference-only AdamW schedule/moments** because both exposure
   distribution and objective conditioning change. This is a separately recorded
   development revision, not an exact continuation or automatic extra epoch.
   The old artifact's 1,000 updates and discarded 16-update setup attempt retain
   their original evidence. Do not initialize or optimize live R/D/codec/W.
5. Run one fixed **1,000-window / 8,000-exposure** revision, microbatch one, using
   the existing ownership, eligible-component averaging, coefficients 1.0
   reconstruction / masked absent contrasts, clipping and LR schedule. Cache only
   frozen-W annotation states on the prescribed RTX 3090; pooling/T/wire stay
   connected and recomputed. Explicit reviewed training-reference contrast
   relationships are still absent; do not invent batch negatives.

This explicitly revises the reference fitting exposure recipe; the original
module shapes, slot allocations, loss coefficients, precision and numeric
tolerances remain unchanged. `--profile legacy-v1` reproduces the old recipe;
the default fitter profile is now `recovery-v1`. `--initial-reference` is an
explicit warm start, distinct from exact `--resume`.

## Acceptance, recording and stopping

Before fitting, require role/provenance/split/sampler checks and serial/parallel
GPU loss/input/T-gradient checks for both history exposures. Verify the ablated
path never embeds target tokens and retains finite nonzero prefix derivatives.
Save/restore the actual zero-step checkpoint before the first update.

Evaluate/checkpoint every 100 windows on all **100 unchanged development RF
records plus all 100 existing contrast groups**. The promotion requirement stays
**>=95/100 exact reviewed recoveries and >=90% contrasts at both 900 and 1,000**.
Nonexact outputs remain unresolved. No development or locked annotations enter
fitting, no easier cases replace failures, no post-hoc threshold or denominator
change, and no earlier favorable checkpoint can substitute for the final two.
Live parameter hashes, 320 frozen W tensors/tying, reference provenance, exact
8,000 exposures and original release hashes must verify.

Finish this bounded revision and record actual outcomes, including failure. Do
not automatically extend the block, tune on locked tests, author more data or
start capability training if it fails. Any subsequent revision needs a separate
documented causal change, not repeated fitting in search of a passing checkpoint.
Passing would qualify the reference gate only; actual learned invocation,
predicted-packet semantic binding and full-trainer memory remain separate work.
