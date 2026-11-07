# Direct-readout diagnostic: training signal, weak development transfer

The authorized [direct affine-readout diagnostic](direct-latent-readout-v1.md)
completed on the RTX 3090 using saved features. **It does not establish a working
Noul/Score receiver or a reliable representation.** It learns useful training
signal, but held-out Noul performance is weak and Score error worsens.
No production installation, new data, additional fit or promotion followed.

[Portable evidence](evidence/direct-latent-readout-v1.json) pins the data, source,
feature caches, probes, cross-validation predictions and independent receipt audit.

## Scope and fitting

Training uses the existing **256 assessed Noul /254 assessed Score** records from
the 1,003-feature cache, **96 families per operation**. No non-assessed value
targets are invented. Saved development h exists for the X panel: evaluate its
**70 assessed Noul /40 assessed Score** cases, **20 families per operation**.
E h was not saved, so E is excluded without re-extraction. These denominators
differ from the previous 80/50 assessed cases across all E+X.

Each probe is a direct affine map over all 1,024 h coordinates with a bias,
bypassing the fixed scalar heads/prototypes and residual bridge. Fit by CUDA FP64
ridge solves, with training-subset mean/scale. Three-fold family-grouped training
validation chooses regularization; development never enters fitting or selection.
All training permitted inputs are unique and training/development have no family
or permitted-input overlap. Both probes select **lambda 0.001**.

| Lambda | Noul training OOF raw MSE | Score training OOF raw MSE |
|---|---:|---:|
| 1e-9 | 12.224421 | 3.103114 |
| 1e-6 | 3.313404 | 2.036469 |
| 0.001, selected | 0.167704 | 0.143547 |
| 1 | 0.190249 | 0.188098 |

Completed **24 CV solves +two final fits**. The two selected probes and all
training selection evidence were exported before opening development tensors.
No iterative optimizer, convergence cutoff, threshold calibration or search
after seeing development. Weakly regularized candidates generalize poorly even
within grouped training validation; their selection was not rescued using X.

## Scalar results

These are readout-only predictions with reviewed Assessed eligibility. They do
not predict status or demonstrate delivery, learned invocation or Reason meaning.
Values use the existing FP32 scalar ABI and [0,1] clamp; Noul ties fail at 0.5,
and exact Score point intervals are not widened.

| Measurement | Training fit | Training OOF | Development X |
|---|---:|---:|---:|
| Noul correct | 197/256 | 177/256 | 48/70 |
| Noul balanced accuracy | 75.14% | 66.21% | 55.50% |
| Noul Brier | 0.154134 | 0.166796 | 0.226305 |
| Noul AUC | 0.866071 | 0.785038 | 0.558000 |
| Score exact hits | 1/254 | 8/254 | 0/40 |
| Score mean interval distance | 0.263342 | 0.310479 | 0.336694 |
| Score clipped MSE | 0.087785 | 0.143412 | 0.168792 |

OOF rows come from three distinct training-subset fits, not the final probe.
Score clipping occurs on one training prediction and eight OOF predictions;
the exact-hit counts do not establish reliable interior numeric decoding. Six development Score
predictions clip at one, with raw range **0.052935..1.202283**. Development raw
Score MSE is **0.200271**, worse than its clipped MSE; raw errors remain visible.

Development Noul has **50 true /20 false** labels. The probe gets **43/50 true
and 5/20 false** correct. Thus its 48/70 headline is below the **50/70 majority-label
reference**, and balanced accuracy is only 55.5%. This reference describes the
label counts; no development-selected majority model or cutoff was fitted.
Noul probabilities range 0.414057..0.684599, without clipping or ties.

Compare the exact same X scalar subset against saved prior predictions:

| Measurement | Parent 0188 | Joint residual | Value-only residual | Direct probe |
|---|---:|---:|---:|---:|
| Noul correct /70 | 20 | 20 | 20 | 48 |
| Noul balanced accuracy | 50% | 50% | 50% | 55.5% |
| Noul Brier | 0.289148 | 0.307112 | 0.307054 | 0.226305 |
| Noul AUC | 0.600 | 0.621 | 0.618 | 0.558 |
| Score exact hits /40 | 0 | 0 | 0 | 0 |
| Score mean distance | 0.233890 | 0.250675 | 0.254892 | 0.336694 |

The Noul prediction distribution changes substantially; the larger correct count
is not evidence of strong semantic transfer. Ranking is worse than all three
saved baselines. Score error worsens in both direction groups: higher-scale
**0.247125 parent →0.289264 direct** on 30 cases; lower-scale
**0.194187 →0.478984** on ten. The training set has no assessed reversed-scale
Score examples, but higher-scale results fail too.

Of 90 existing intervention pairs, **50 have two assessed scalar targets**.
The probe gets both values correct on 12/50: **4/20 changed-truth Noul pairs**
(one negation, three criterion-change), plus eight of ten unchanged-target Noul
pairs. Score gets zero of its 20 pairs correct. Correct direction on changed
targets is **20/40**, versus parent 22/40, joint 21/40, value-only 18/40.
The 12 scalar-correct pairs are not complete runtime/behavior passes.

## Validation and artifacts

**26 CPU tests pass in 14.118 s**, with no CPU model execution; compilation and
whitespace checks pass. CUDA analytic affine and independent primal/dual checks
pass (maximum absolute difference **1.387778781e-17**); constant features are
rejected. Maximum relative equation residual across all 26 fitted systems is
**1.547262027e-11**, below the declared 1e-8 solver tolerance. This is a separate
FP64 diagnostic profile, not a modification of v3 or its tolerances.

Export/reload predictions match exactly on all **510 training +110 development**
records. Saved feature tensor/file hashes and parent manifest remain unchanged.
Independent CPU audit recomputes training-only selection, family isolation,
OOF risks, all scalar metrics, FP32 conversion, direction groups, paired results,
prediction coverage, provenance and zero-position accounting. Existing production
numeric qualifications are reused. There is no new backpropagation path to qualify.

Raw: `data/direct-readout-20261107/run/`, with logs, `audit.py`, `audit.json`,
`audit.log` and CPU test evidence in its parent. `prepared/` is model-free preflight.
The export is `run/readouts.safetensors` with its hashed `export.json`, explicitly
marked incompatible with production installation. No failed/retried GPU attempt.

GPU fit/check/evaluation wall **2.247410 s**, peak CUDA **46,622,720 bytes**.
This excludes release/provenance preparation and imports, and is not a matched
transformer throughput benchmark. **Zero new Sessions, prefills, causal positions,
source encodings or reference encodings**. W/reference and the codec were never
loaded into model instances; Choice was not modified or freshly evaluated.
Historical Choice/transport receipts remain historical. After exit only existing
service PID 2094 remains, using 3,852 MiB. No 3060, agents, new annotations/reviews,
locked-data access, native training, deployment or Operating Table/service changes.

## Interpretation and next unfinished work

A simple decoder can exploit some information in training h, including grouped
training validation, so the previous failures are not evidence that h is wholly
uninformative. However, this diagnostic does **not** supply strong held-out truth
or numeric decoding. It does not isolate architecture, optimization, representation
fidelity or train-to-development distribution differences as the remaining cause.
A failed affine probe cannot rule out useful nonlinear decoding.

Next unfinished: audit the existing training-to-X task/input mismatch and the
semantic/numeric information preserved by the latent path before another receiver
fit. Reuse these cached features and results. Do not automatically enlarge the
training run, repeat a loss sweep, promote the probe or generate new examples.
Reference meaning, status/value integration, Reason meaning, learned invocation
and full native use remain unqualified. Further model experiments need new scope.
