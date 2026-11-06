# Bounded reference weight repair results — 2026-11-06

**The single repair failed the recovery criterion.** Both repaired checkpoints
still recover **0/100** existing development annotations, with **100/100**
contrasts, matching their source-bound parents. The production qualification
loader rejects both with `reference qualification did not pass`. The experiment
completed; this is a measured semantic failure, not a hardware blocker.
See the [portable receipt](evidence/reference-weight-repair-v1.json).

## Completed implementation and acceptance

Implementation/protocol commit `54a171a` preceded application and all semantic
feedback. The [fixed protocol](reference-weight-repair-v1.md) tests activation
common-mode removal: estimate one unit output direction per physical pooler
branch, then apply `L' = L - u(u^T L)` and `b' = b - u(u^T b)` once at full
strength. The same rule applies independently to the saved 900/1,000 parents.
There was **one candidate rule, two applications, no search or outcome-based
retry**. Both repaired artifacts were exported before development evaluation.
This is a hypothesis test, not a demonstrated causal explanation of recovery.

The runner reuses the reviewed training catalogue and its original hashes/field
addresses. Deterministic hash ordering selects **71 distinct existing views**:
Question 8, Target 8, Alternative 8, Dimension 7, Conditions 8, Zero 7, One 7,
Midpoint 5, Key 5, Reason 8. Calibration retains answer blindness and the
explicit Reason exception; it uses no decoder target, loss or development
feedback to construct the edit. Development is evaluation-only; locked labels
were not loaded. No annotations, reviews or source scenarios were generated.

Construction uses CUDA FP64 and stores FP32 lift weights. Normal forwards retain
v3 BF16 frozen storage, FP32 transformer computation, compensated BF16 vocabulary
projection and graph-v1 recurrence. The original BF16 transformer profile still
fails; its historical failure and completed numeric qualifications are unchanged.
Runtime channels, latent tags/normalization, one-prompt cache handling, budget
mapping, tolerances, vocabulary and architecture did not change.

**Two metadata tests passed in 0.001 s; two synthetic RTX 3090 tests passed in
0.347 s; 19 related metadata regressions passed in 0.501 s.** Checks cover
selection determinism/coverage, invalid-input rejection, affine projection,
orthogonal-complement preservation, nonmutation and detached repair outputs.
Compilation and `git diff --check` passed. These are engineering checks, not
semantic qualification. No completed expensive numeric matrix was rerun.

## Measured semantic outcome

| Parent window | Parent exact recovery | Repaired exact recovery | Parent/repaired contrasts | Production loader |
|---|---:|---:|---:|---|
| 900 | 0/100 | **0/100** | 100/100 each | Rejected |
| 1,000 | 0/100 | **0/100** | 100/100 each | Rejected |

Each repaired artifact executed all **100 original RF cases and all 100 original
contrast groups**, using normal learned pooling and greedy causal-cache recovery.
An independent audit verifies identical case IDs/annotation hashes and unchanged
qualification source/profile. Exact recovery stops at the first mismatch; no
gold token is supplied before its prediction. Original parent reports were
hash-verified and reused, avoiding duplicate baseline evaluation. The unchanged
>=95/100 recovery and >=90% contrast criterion must hold at both endpoints;
it failed. Existing contrasts include identical positives and already passed
before this repair; their continued success does not establish semantic recovery.

## Weight and workload accounting

Only `pooler.{shared,target}.lift.{weight,bias}` changed: **4 of 541 reference
tensors per artifact**, with the other **537 byte-identical**, including all
**483 T/codec tensors**. Changed FP32 entries number **526,334 / 526,336** for
900/1,000. Rank-one does not mean few scalar entries. Weight delta Frobenius
norms relative to the original lift are:

| Window | Shared lift | Target lift |
|---|---:|---:|
| 900 | 0.12480747408851024 | 0.1266667012322839 |
| 1,000 | 0.12391001442304958 | 0.12594138164831667 |

Before editing, mean absolute payload alignment with the estimated unit direction
was 0.6966770152560244 / 0.8329057132117538 for shared/Target at 900, and
0.665990696463984 / 0.816948837630829 at 1,000. Four post-edit checks on cached
calibration states give output orthogonality residuals from
**1.899765103711248e-9 to 5.061970714233198e-9**. The programmed projection
worked mechanically; this did not improve exact recovery.

All model/tensor workloads ran on **RTX 3090 UUID
GPU-225509c3-d702-6c27-de03-76b5e99b33b1**. Calibration used exactly **71 frozen-W
annotation prefills**, shared across the parents, **142 pooler forwards**, and
four post-edit checks on cached states. Evaluation adds its original RF/contrast
workloads. There were **zero gradient updates, optimizer steps, capability
updates, agents or external model calls**. Base W passes all **320 original
tensor comparisons and physical tying**. Live identity before/after is
`c56c8f45d11f4eca02ba4bbc1d1024c50d1b74cceeee8e1d3a5c8e7378485450`;
all owner gradients remain `None`. Parents and all three release manifests are
unchanged. No service or Operating Table artifact was modified.

Runtime was **284.00155400899894 s**, including metadata validation and both
qualifications; peak allocated CUDA memory **3,622,856,704 bytes**. Runner exit
was **2**, recording the failed paired criterion. Independent file-byte/hash
checks pass for all 54 archived source/protocol/test files, both exports, all
537 nonedited tensors per export, cases/annotations and release manifests.
File-byte/metadata auditing uses CPU; it executes no model or tensor arithmetic.

Repaired frozen fingerprints (unqualified, never promoted):

- 900: `41bb9e035d9036cb182d16f965039d64a720c9cc85e0f6dbd7cf61235a1c0f29`
- 1,000: `e40c025b621a4c587f93811f4e22aa66c25a1fb9b0b06874212a7d2ca6c66ada`

## Evidence and next unfinished slice

Raw evidence and both repaired exports are preserved under
`data/reference-weight-repair-20261106/`, with the run in `run/`. The portable
receipt binds the actual report, qualifications, calibration provenance,
directions, edit accounting, export manifests/weight hashes, source snapshot,
test logs, independent audit and parent report/export identities. Model/data
artifacts remain ignored by git. The completed command was:

```sh
CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 .venv/bin/python -m scripts.repair_reference --snapshot <pinned-base-snapshot> --output data/reference-weight-repair-20261106/run
```

No job remains active. The next unfinished slice remains full-mixture reference
recovery/generalization and retention; this bounded common-mode intervention is
retired. Its failure does not prove every weight-space repair impossible or
establish that new data is necessary. There is no further edit, fine-tuning,
data generation or capability training queued. No repaired artifact initialized
live weights. Semantic binding, observed correction materialization and full
participating-trainer memory remain separately unfinished; passing mechanical
projection does not certify them or the learned flow.
