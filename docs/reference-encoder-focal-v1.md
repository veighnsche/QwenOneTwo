# Focal reconstruction encoder revision v1 — 2026-11-06

The user authorizes fixing reference encoder/pooling learning on existing data
using the RTX 3090. No new source data, agents, capability fitting or services.
This is a new causal objective revision; it does not relax qualification.

## Evidence and exact change

Four completed, separately bounded 64-window probes failed six-case training
acceptance: stationary decoder CE 5/6; CE plus dense latent-target MSE 4/6;
field-value-only memory 5/6; five-times pooler peak LR 5/6. None received a full
block. Dense-target, field-value and high-rate candidates are not active fitter
profiles; their actual sources/patches/results remain archived.

A read-only training-pair probe finds the Target payload cosine increases from
0.9623206257820129 in the parent to 0.9933992624282837 in the high-rate control,
while the first Target still substitutes arrival lateness for work completion.
Attention/input derivatives are not confined to pair-blind prefix positions.
Low mean token CE and increasingly similar outputs have not recovered the
material distinction. This supports testing the reconstruction token objective,
not declaring source data absent or a unique architectural root cause.

Keep the original parent
`d0d4aef44b4415beb19311350eb24533190e1a27406c5d6f59a15a5a57402dec`, full annotation
memory, original pooler shapes, fixed T/codec, ordinary causal teacher-forced
inputs and pooler peak LR **2e-4**. Change only token loss to **focal CE gamma 2**:

    mean_i [(1 - exp(-CE_i))^2 * CE_i]

Use `-expm1(-CE_i)` for the factor. Average all eligible target tokens within
each example, then the same eight eligible examples per Window. No token is
classified by its wording, no new negatives/labels are invented and source
annotation is not a decoder prefix. Correctly predicted scaffold tokens receive
less gradient relative to remaining errors. Gamma zero preserves original CE
arithmetic and remains the default for historical/other callers. Inference and
qualification stay normal greedy autoregressive recovery, with no focal rule.
This explicitly versions the training objective; it changes no numeric tolerance,
precision, slot count, runtime field, scalar or S-curve.

## Fixed training-only acceptance

First require serial/parallel GPU focal loss and input/T-gradient agreement at
the original tolerances. Run the same six training IDs, exposure order, seed,
64 Window updates / 512 exposures and 0/16/32/64 observations as the ordinary-CE
control. Require **6/6 final exact recovery**, finite eligible gradients, only
pooler optimizer owners and unchanged decoder/live/base. Do not increase steps,
select an earlier observation, lower acceptance or continue failed probe weights.
Record all real update counts separately. No full block follows failure.

## Conditional full block

Only after acceptance, discard probe weights and reset to the exact parent and
fresh pooler-only AdamW. Explicit `--profile encoder-focal-v1` runs one fixed
**1,000-window / 8,000-exposure** block over the same 3,376 reviewed training views
and exact four-diversity/four-role-balanced sampler, seed 17. No development or
locked fitting. Freeze T and codec; preserve F7/Reason, clipping, decay, schedule,
reference coefficient 1.0, masked absent explicit contrasts, cache continuity,
frozen-W CUDA feature caching and connected pooler recomputation. Loss reports
must identify focal loss rather than call it ordinary CE.

Save/restore checkpoint zero; checkpoint/evaluate every 100 windows using all
100 original RF development records and all 100 existing contrast groups. The
unchanged final-two gate is **>=95/100 exact recovery and >=90% contrasts at both
900 and 1,000**. Verify exact accounting, decoder/live hashes, 320 frozen base
tensors/tying, source/release hashes and reference-only ownership. No earlier
checkpoint substitution, threshold/denominator changes, gold repair, automatic
extra epochs, new source data or capability training. Record actual failure as
failure; a six-case fitted control never qualifies held-out recovery.
