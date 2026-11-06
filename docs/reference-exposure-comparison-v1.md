# Exposure-controlled reference comparison — 2026-11-06

The user authorizes this reference comparison on the RTX 3090, with existing
data, W frozen, no agents or capability training, and no arbitrary time cap.
Only the existing target-v2 reference pooler is fitted. W, T, codec and live
parameters remain frozen. Arithmetic, reconstruction and acceptance are unchanged.

## Matched intervention fixed before execution

Reuse the completed encoder-target-v2 1,000-window control and its saved initial
checkpoint, including exact optimizer state and RNG. All original source files
must still match the control's source snapshot. Keep 1,000 updates, eight examples
per update, ordinary teacher-forced CE, A4 learning schedule, AdamW settings,
clipping and eligibility normalization. There is no elapsed-time stopping rule.

Replay the recorded control schedule with exactly 336 same-role substitutions:
Question train-reference-0000/0010 increase from 1 each to 86 each; Target
train-reference-0001/0011 increase from 2 each to 85 each. These final counts match
the successful six-case probe. Key 0008/0018 remain at 81 each. Preserve all
original anchor presentations, the first occurrence of every other view, all
3,376 distinct views, all 8,000 slots and each individual slot's role. Replace
only repeat occurrences of non-anchor views of the same role. Choose integer
midpoint quantiles of eligible chronological slots, alternating the two anchors
within each role. Save every substitution and before/after exposure count.

This is one targeted exposure reallocation within the original broad mixture,
not equal exposure for all views. Total positions may change with example length;
record them instead of claiming token/compute matching. The control is reused,
not rerun. Check exact initial fingerprint and restored optimizer ownership;
compare loss and gradient norm against the old control until the first changed
window. No change to training math or extra candidate/epoch is authorized here.

## Measurements and interpretation

At windows 900 and 1,000 evaluate exact greedy recovery on the same 40 training
cases saved by the completed block search. Reuse its original final baseline
(21/40 overall; 2/6 controls), report individual gains/losses and unchanged-case
retention, and distinguish that final baseline from treatment window 900.
All four targeted cases and the two Key controls belong to the first 20 cases;
the other 20 provide separate retention observations, not unseen training data.

Run the original development 100-case recovery and 100 explicit contrast suite
at BOTH 900 and 1,000, with unchanged >=95% recovery and >=90% contrast thresholds.
Use the existing qualification evaluator and evidence loader. No locked labels,
gold repair, alternative metric acceptance, earlier checkpoint selection or live
promotion. Preserve checkpoints, frozen exports and per-case reports.

Improved targeted recovery is evidence that exposure allocation contributes to
those failures. Persistent development failure means this does not solve reference
generalization. No targeted improvement rules out this particular allocation as
a sufficient repair, not every possible exposure schedule. Retention regressions
must be reported. One deterministic comparison cannot establish a universal or
unique root cause. Do not call the old broad run catastrophic forgetting: it
restarted from the common initial weights, not from the six-case probe.

Verify actual eligibility, active pooler ownership, unchanged 483 T/codec tensors,
all 320 frozen W tensors and tying, live identities, data and source hashes.
Record measured time and memory without making them stopping conditions. No
new generation, annotation/review, external model calls or capability updates.
