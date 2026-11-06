# Reference encoder latent distillation v1 — 2026-11-06

This is a separate causal encoder-learning revision, authorized by the user's
request to fix pooling with existing data on the RTX 3090. It creates no source
annotations, reviews or new scenarios and trains no live capability weights.
The stationary-decoder/ordinary-CE control completed its fixed 64 windows and
improved 1/6 to 5/6 training recoveries, failing its six-case acceptance. It will
not receive the conditional full block. Its failed artifacts remain evidence.

The remaining Target confuses two meanings despite average CE 0.1254262. The
existing decoder expresses all six annotations after 16 direct latent steps.
The new hypothesis is that pooling needs a dense, stationary representation
learning target instead of only the decoder's diluted/conflicting token loss.

## Fixed objective and training-only acceptance

Freeze the same parent T and codec, keep the same pooler architecture and initial
weights, and use the same six train IDs / 64 Window updates / 512 exposures,
optimizer schedule, clipping, seed and checkpoint observations as the stationary
control. Do not continue the control's weights. For each existing training view,
start at the original parent pooled payload and optimize **only its disposable
latent vector for 16 Adam steps** (LR .05, betas .9/.999, eps 1e-8), ordinary
reconstruction CE through the frozen decoder/wire. This is the already measured
isolation recipe at its recorded step 16, not an unbounded search. Normalize and
detach the resulting payload with the existing canonical N.

Learn the pooler with **ordinary reconstruction CE + mean squared distance to
that fixed canonical latent target**, coefficient 1.0 for each term. Both terms
average within each example; the existing eligible-example accumulation then
averages eight examples. Log terms separately; no contrast labels are invented.
Teacher targets use only the view's own permitted annotation, never assessed
outcomes (except the documented non-assessed Reason content), development or
locked records. Do not differentiate through target preparation or supply source
annotation text as a decoder prefix. Runtime/qualification uses the learned
pooler alone, without target fitting, annotation lookup or latent repair.

Require all six exact final training recoveries, finite eligible exposures,
unchanged decoder/live/base tensors and correct ownership before a full block.
Record 96 disposable-latent steps and 64 reference pooler updates separately.
Discard fitted probe weights. No increased probe steps or threshold changes.

## Conditional full existing-data block

If the training-only acceptance passes, prepare fixed numeric targets for all
3,376 existing reviewed training views with the same parent and 16-step recipe.
These are training intermediates, not new annotations or additional independent
cases. Pin release/catalogue/parent/source/recipe hashes and cache chunk hashes;
cache frozen-W states too so they need not be re-extracted for fitting. Save
completed chunks to avoid repeating expensive preparation after interruption.
Use CUDA for extraction, target fitting and pooler learning. No CPU fallback.

Reset the pooler to the original parent, start fresh pooler AdamW, and run exactly
one **1,000-window / 8,000-exposure** block with the existing four-diversity/four-
role-balanced catalogue sampler, seed 17, and the objective above. T and codec
stay frozen; live W/R/D/codec stay unchanged. Preserve ordinary causal cache,
slot allocations, BF16 storage / FP32 transformer / compensated BF16 vocabulary,
all tolerances, reference eligibility averaging, LR schedule and finite guards.
Keep target-preparation steps distinct from reference model optimizer updates.

Save/restore checkpoint zero; checkpoint/evaluate every 100 windows using all
100 original development RF records and 100 contrast groups. The same final-two
gate applies: **>=95/100 exact recovery and >=90% contrasts at both 900/1,000**.
No development/locked fitting, earlier-checkpoint substitution, relaxed gate,
new-data generation or capability training. Failed teacher targets or gates are
recorded honestly, without automatic retry/extra-epoch loops. Passing the six
controls or reducing training loss does not qualify held-out recovery.
