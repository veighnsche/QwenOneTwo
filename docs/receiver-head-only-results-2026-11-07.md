# Frozen-representation Noul/Score heads — 2026-11-07

The authorized 3090 experiment completed. **Score can closely approximate these
training values with fixed representations, but this fit does not generalize.
Noul still fails to separate the balanced training panel. Choice is retained
exactly.** This is a completed diagnostic with no promoted checkpoint.

Read the [predeclared protocol](receiver-head-only-v1.md) and
[portable evidence](evidence/receiver-head-only-v1.json).

## Scope and machinery

Only the existing **Noul prototypes (2×1024)** and **Score row (1024)** train:
**3,072 FP32 master parameters**. W, R/D, all input encoders, shared Qev q/k,
status/Choice/feasibility/Reason heads, budget and reference stay frozen. No new
head, bias, threshold, target, label or widened interval. Score retains the
BF16-effective row, FP32 dot product and original raw regression loss.

The runner restores checkpoint-0188 under its original strict identities, then
projects the two heads' AdamW moments into unchanged head groups/settings.
Moments match exactly; the A4 schedule continues from 188, with clip 1. The existing
32-record panel/order is unchanged and family-disjoint from development. Saved
joint-probe receipts provide initial parity and a 32-update comparison; parent
checkpoint-0188 receipts provide the held-out starting baseline. Neither control
is rerun. This does not start from either later checkpoint-0220.

Training features are extracted through actual latent packets on the 3090, before
label access. Detached CUDA features are reused for **1,812 head updates**, A4
windows **189..2000**, **453 repeats /14,496 cached visits**. Eligible totals are
**Noul 3,624 and Score 4,530**; Choice/non-assessed visits have no head loss. Every
window steps, with zero failed window exposures. These are cached readout visits,
with **zero new causal positions or prefills**, not 14,496 latent episodes.

There are 32 initial and 32 final fresh training extractions. Both sets charge
**3,332 positions** independently, with one prefill/lineage per case, exact
P+B+A+H, zero held positions/invocation text tokens and no gold returns. Offline
reference encoding is outside those session counts. Fresh final hidden states,
candidates, source tensor hashes, head results and head losses match the cached
path exactly. No stale representation is passed off as connected joint learning.

## Training fit

Status remains fixed at initial Noul 10/11, Choice 6/8 and Score 12/13. Choice
assessed selections remain 4/4 at every measurement. Value heads are measured
independently of status, without claiming feedback delivery on cached visits.

| Head updates | Noul truth /8 | Noul Brier | Score interval hits /10 | Score mean interval distance |
|---|---:|---:|---:|---:|
| 0 | 4 | 0.256381 | 0 | 0.411528 |
| 32 | 4 | 0.256086 | 0 | 0.300245 |
| 128 | 4 | 0.248804 | 2 | 0.140865 |
| 256 | 4 | 0.246434 | 3 | 0.026252 |
| 512 | 4 | 0.242616 | 3 | 0.019915 |
| 1024 | 4 | 0.235968 | 3 | 0.010840 |
| 1812 | 4 | 0.232680 | 3 | 0.000959 |

Noul BCE decreases **0.706077 → 0.658275**, but training truth accuracy remains
4/8. No claim of convergence or absence of recoverable information follows from
that failure. At the matched 32-update point, the joint probe has Noul Brier
0.247584 and Score distance 0.166759, versus head-only 0.256086 and 0.300245. The
later 1,812-update endpoint has much greater exposure and is not a fair equal-step
improvement claim over that joint probe.

Score raw regression MSE falls **0.206481 → 0.000001851882**. **All ten training
intervals have zero width**: these are exact FP32 point comparisons. The final
values are:

| Target | Delivered prediction |
|---:|---:|
| 0.99 | 0.988551021 |
| 0.50 | 0.498289049 |
| 0.06 | 0.058646873 |
| 0.42 | 0.418458551 |
| 0 | 0 |
| 1 | 0.998903453 |
| 1 | 0.999121070 |
| 0 | 0 |
| 1 | 0.998438179 |
| 0 | 0 |

The three exact hits are zero targets reached through the existing clamp; their
raw predictions are −0.001239, −0.001133 and −0.001433. The seven misses have delivered
errors 0.000879..0.001711. These small residuals explain the low strict hit count
on this training panel, but no tolerance is added or point target relaxed. Their
optimization/quantization cause has not been isolated. Close training fit also
does not establish semantic generalization.

## Held-out head accuracy, behavior and Choice retention

All 130 E + 180 applicable X cases execute after training with actual decoded,
bound and delivered latent results. All statuses match parent checkpoint-0188
exactly. **All 90 Choice cases retain actual results, predicted statuses, source
views, grades and failure fields exactly**. All 310 transports pass; no evaluation
case fails. Locked/native-reader cases stay outside this experiment.

| Assessed independent head measure | Parent 0188 | Head-only final |
|---|---:|---:|
| Noul truth correctness | 25/80 | 24/80 |
| Noul Brier | 0.284675 | 0.261361 |
| Score interval hits | 1/50 | 0/50 |
| Score mean interval distance | 0.271739 | 0.319199 |
| Choice delivered correct selection | 47/50 | 47/50 |

E Noul raw truth is 5 → 4/10; X stays 20/70. Score has 0/10 E and 0/40 X head hits.
Score E mean distance worsens 0.423136 → 0.458949 and X 0.233890 → 0.284261. Training
point fitting did not transfer to these cases.

Delivered assessed success remains **Noul 0/80, Choice 47/50, Score 0/50**. Frozen
parent status accepts zero assessed Noul cases and 45/50 assessed Score cases,
so independent head results must be separated from delivery. Total behavior
stays **83/310**: E Noul 13/40, Choice 11/50, Score 10/40; X Noul 9/100, Choice 40/40,
Score 0/40. The ten E Score successes are non-assessed, not numeric successes.
Status stays 130/310; X paired success stays 23/90 and its existing semantic/
behavior qualifier remains BLOCKED. The diagnostic itself completed successfully.

## Checks, limits and next slice

**26 CPU tests pass in 29.597 s**, with model imports blocked. Independent audit
passes for release/control/source identities, all 1,812 ordered windows and
eligible denominators, cached/fresh receipts, reports and all 12 checkpoint file
sets. Minimal-head versus actual Codec Noul/Score VJPs match exactly before
updates; forward readouts match at all measurements. The first-update restore
reproduces receiver, optimizer tensors and CPU/CUDA/Python RNG exactly.

All 320 W tensors/physical tying pass before/after. Frozen representations,
reference, cache tensors and boundary grad=None checks pass; no other owners
update. No backbone/loss arithmetic changes or numeric tolerance changes;
existing numeric matrices are reused. No agents, data generation, reviews,
service/deployment changes or Operating Table modifications. After exit only
the original GPU service PID 2094 remains, 3852 MiB on the 3090.

The result narrows the problem: approximate Score training readout learning is
possible here, but held-out transfer is poor. Noul remains unresolved even on
eight balanced training examples with representations fixed. **Next unfinished
slice:** measure frozen-feature information/separability and conditioning on the
3090 to distinguish weak/ambiguous features from readout optimization problems;
inspect the Score transfer gap separately from its small training residuals.
Do not infer an architectural impossibility, declare MVP success, widen the
grader or launch another fit automatically. Full learned invocation, native
answer use, Reason meaning and general receiver training remain unqualified.

Raw plan, executed sources, feature tensors/receipts, eleven panels, head-update
ledger, fresh-final receipts, E/X evaluations, checkpoints, test/audit logs and
commands/exits: `data/receiver-head-only-20261107/` (ignored, preserved).
Whole-run wall **525.633195 s**, peak CUDA allocation **3,622,856,704 bytes**,
including parent optimizer loading; no isolated throughput benchmark is claimed.

Final checkpoint `run/checkpoint-2000`, manifest
`7f05e09f5c7a7b58a9dc69556ace3a38d1e9c4f71f46173c716b597e236991e2`;
receiver `618412959e691888215f208f75e1bad8c6d4d0d32bac18dfaf37cf5776191751`.
Frozen representation fingerprint
`94153808343ca8b2bb3b5763d00db94eac5c2c06a205d188d00018208726e64e`.
Reference remains
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.
Head bundles depend on the pinned parent for frozen adapters. No standalone
model export or promotion: **COMPLETED_HEAD_ONLY_NOT_PROMOTED**.
