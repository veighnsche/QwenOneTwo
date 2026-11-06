# Complete-coverage reference experiment — 2026-11-06

The user authorizes implementing shuffled complete passes and running a
learning-curve-driven reference comparison on existing data. This versioned
development revision supersedes the fixed 1,000-window stopping rule for this
experiment only. No elapsed-time, token, spending or maximum-pass cap applies.

Continue the completed exposure-treatment checkpoint at update 1,000, preserving
its exact model weights, optimizer moments/steps and RNG. W, reference T/codec
and live weights stay frozen. Only the existing target-v2 pooler learns, on RTX
3090 UUID GPU-225509c3-d702-6c27-de03-76b5e99b33b1. No agents, new data/reviews,
capability training, promotion or service changes.

Each pass presents every one of the same 3,376 unique reviewed annotations
exactly once: 422 eight-example optimizer windows. Shuffle each pass with the
fixed seed-17 protocol. Preserve per-example CE, eligibility normalization,
clipping, AdamW moments/settings and v3/graph-v1 arithmetic. The explicit
reference-only continuation schedule holds the existing peak pooler LR 2e-4
constant; the old cosine had finished decaying. Record this schedule change:
this is a continuation/coverage experiment, not a matched sampler-only causal
comparison. Default A4/capability schedules remain bounded and unchanged.
Checkpoint manifests pin the schedule profile and exact sampler cursor.

Before fitting and after every complete pass, measure a fixed training panel:
all prior 40 diagnostic annotations plus deterministic outcome-independent
selection up to 32 distinct views per role (all views for smaller roles), 191
total. Report exact recovery, each role, original 40-case gains/losses, and
pass-average CE over ALL 3,376 training presentations. The panel is a learning
curve, not certification of the full training catalogue. Reuse existing final
development baseline and prior 40-case outcomes at the identical starting
fingerprint, with bound unchanged model sources. Full-catalogue checks reuse the
same-checkpoint panel recoveries; do not repeat those greedy episodes.

At 100 updates before the end of each pass and at its end, run the original
100-case development recovery and 100-group contrasts. Preserve >=95% recovery
and >=90% contrasts at BOTH final checkpoints, exactly 100 updates apart. No
locked labels, threshold relaxation, earlier-checkpoint selection or live use.

Continue complete passes while learning improves. A plateau is three consecutive
complete passes with neither a new best panel exact-recovery COUNT nor a new
best whole-pass mean CE. Three separate shuffles distinguish sustained lack of
progress from a single unfavorable presentation order; this is an observed
learning condition, not a runtime limit. Strict comparisons introduce no new
numeric tolerance. A new best in either metric resets the plateau counter.

When the panel reaches >=95%, evaluate the ENTIRE 3,376-view training catalogue.
If full recovery remains below 95% and there is no plateau, continue passes.
At a plateau also perform the full catalogue evaluation. Stop at a full-catalogue
>=95% training recovery or the defined plateau, report the final original paired
development gate and retention, and preserve the actual last checkpoint. Full
training success with failed development points to generalization; poor training
recovery after the observed plateau leaves pooling/decoder learning unresolved.
No automatic architecture change or capability training follows either outcome.

Save complete small-weight/optimizer/RNG/sampler checkpoints every 100 updates,
at both per-pass development observations and at pass boundaries. Resume uses
the saved coverage cursor and in-pass loss totals, preserving exposure continuity.
Record exact view counts, charged and executed forward positions, time/memory,
finite eligibility, role-local ownership, frozen identities and source/data hashes.
Do not rerun unchanged numeric matrices. Retain raw evidence locally; commit the
implementation, measured results and handoff, then push main.

## Explicit user conclusion: pass 12

During pass 12 the user instructed: "this is the last pass... do not start
pass 13". This overrides the automatic recovery/plateau stopping rule for this
run. Preserve progress at a checkpoint and resume with `--stop-after-pass 12`.
The explicit limit is stored in subsequent checkpoints. A loop guard rejects
any next-pass episode once pass 12 is complete. No pass-13 training is allowed.

Complete pass-12 panel/development measurements and frozen-state checks, then
stop with `USER_REQUESTED_FINAL_PASS`. Do not call this convergence or a plateau.
No full-catalogue sweep is required at this user stop; do not infer full-training
recovery from the panel. A narrowly validated source migration permits only the
runner/control-document changes needed for this stop; all model/optimizer/math
sources, saved weights, moments, RNG and sampler continuity must remain unchanged.
