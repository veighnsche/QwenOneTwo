# Reference cross-role learning isolation v1 — 2026-11-06

The user authorizes encoder learning on existing data using the RTX 3090, with
no new source data, agents or capability training. Five completed 64-window
six-case controls leave Target 0001 incorrect (dense MSE also loses a Question).
Focal reconstruction finished 5/6. No conditional full blocks were launched.

This diagnostic tests whether shared pooler updates from other roles interfere
with learning the two Target meanings. It is not qualification or a new annotation
collection. Reset to original parent
`d0d4aef44b4415beb19311350eb24533190e1a27406c5d6f59a15a5a57402dec`, freeze T/codec,
keep the original shared pooler, full annotation memory, ordinary causal CE,
seed 17, peak LR 2e-4, A4 schedule/eligibility/clipping/decay. Run exactly
**64 Window updates / 512 exposures**, alternating only existing training IDs
0001 and 0011, four presentations of each per window. No other record is fitted.

At 0/16/32/64 evaluate the original six training controls (Question 0000/0010,
Target 0001/0011, Key 0008/0018) with an independently frozen copy. Record all
results, including any regressions in roles not fitted. Require **both Targets
exact at final window 64**, finite eligible updates and unchanged decoder/live/
base for positive evidence of Target-only learnability. No additional windows,
restart or threshold change. This result cannot qualify held-out recovery or
trigger a full fitting block by itself.

If the Targets learn separately, a separately documented role-specific pooler
revision may then test whether preventing cross-role gradient sharing preserves
all six recoveries. That still needs its own six-case acceptance and unchanged
100-case final-two development gate. If they do not learn, record failure and
investigate encoder conditioning/derivatives instead of claiming sharing is the
cause. No development/locked fitting, label generation or capability updates.
