# Slice 4 result: Choice retained; value repair unsuccessful

The authorized [residual repair](receiver-residual-repair-v1.md) completed on the
RTX 3090. It improved status handling but **did not meet the declared value-learning
acceptance condition**. No checkpoint was promoted and no additional fit was run.
The [portable evidence](evidence/receiver-residual-repair-v1.json) pins the source,
data, checkpoint, raw receipts, checks and independent audit.

## Development results

All 130 E and 180 X cases were evaluated using actual latent requests and returned
results. Failures remain in denominators; there were zero execution failures.

| Measurement | Parent 0188 | Final repair |
|---|---:|---:|
| Overall behavior | 83/310 | 123/310 |
| Correct status | 130/310 | 182/310 |
| Noul assessed truth, independent of status | 25/80 | 25/80 |
| Noul assessed delivered success | 0/80 | 9/80 |
| Noul Brier score | 0.284675 | 0.299799 |
| Choice assessed delivered success | 47/50 | 47/50 |
| Score assessed exact delivered success | 0/50 | 0/50 |
| Score exact head interval hits | 1/50 | 0/50 |
| Score mean interval distance | 0.271739 | 0.285469 |
| Complete intervention pairs | 23/90 | 30/90 |
| Valid transport | 310/310 | 310/310 |

**All 90 Choice cases exactly retain** actual results, predicted status/value,
grades, input hashes, transport flags and complete canonical runtime ledgers.
That includes 40/40 successful X Choice cases; unchanged Choice is not universal
Choice qualification (overall Choice behavior remains 51/90).

Noul predicts false for every assessed development case. Its ranking improves:
descriptive AUC **0.600727 → 0.675636**, but every probability remains below 0.5.
Final positive range is 0.382124..0.421472 and negative range 0.354589..0.420482;
these overlap. The improved ranking is a signal worth preserving, not proof that
changing a threshold would solve generalization. Brier worsens. No threshold was
fit or changed. Overall behavior gains predominantly reflect status/abstention
handling, not more correct truth or numeric decisions.

Score error worsens in both direction groups: higher-scale **0.291127 →
0.304833** on 40 assessed cases; lower-scale **0.194187 → 0.208016** on ten.
Exact delivered success remains zero in both. All 254 assessed training Score
targets are increasing-scale points; the reversed-scale coverage gap remains,
but it cannot alone explain the higher-scale failure.

## Completed fit and checks

Only the **133,184 new residual parameters** were optimized. W, R/D, the entire
existing codec/readouts and the versioned reference remained frozen. Original
Reason/structural features and Choice bypass the repair. The new module is
explicitly installed only inside the experiment, with no production default change.

All **1,003 existing Noul/Score receiver records** were used: 501 Noul and 502
Score. The 2,000-window fresh A4 schedule completed exactly **16,000 visits**:
955 records visited 16 times, 48 visited 15 times. Eligible counts are status
16,000, Noul 4,087, Score 4,049. Every window stepped, with no failed exposure.
Loss coefficients, masks, clipping, actual Score row, continuous initial budget,
S-curve and exact-value evaluation tolerances are unchanged.

| Training measurement | Initial | Final |
|---|---:|---:|
| Noul assessed truth | 168/256 | 168/256 |
| Noul status | 395/501 | 420/501 |
| Noul BCE | 0.657184 | 0.626934 |
| Noul Brier | 0.232134 | 0.217993 |
| Noul descriptive AUC | 0.359037 | 0.767519 |
| Score status | 411/502 | 502/502 |
| Score exact interval hits | 0/254 | 0/254 |
| Score interval-distance loss | 0.200711 | 0.184063 |
| Score mean interval distance | 0.405995 | 0.393886 |

Thus the new branches learn status strongly and some scalar ranking/loss changes,
but do not solve the training value decisions either. This experiment does not
separate loss interference, optimization, remaining readout limitations and
upstream semantic fidelity. It is a complete repair-package comparison against
the frozen parent, not a matched trained ablation of adapter capacity alone.

Validation:

- **19 CPU tests pass in 14.162 s**, with no CPU model execution. Byte compilation
  and whitespace checks pass.
- CUDA FP64 reference checks of the new residual pass the existing 1e-5 relative
  RMS / 1e-11 near-zero absolute tolerance. Maximum forward relative RMS is
  **2.712716892e-8**, maximum VJP relative RMS **1.867542228e-7**. Nonzero VJPs
  reach both projection layers, biases and h. Full/minimal head outputs and
  gradients agree exactly for both operations.
- All **1,003 initial readouts exactly match** the actual parent receipts.
  Four fresh final training Sessions reproduce cached h and repaired heads,
  with identical full training ledgers.
- The first-update resume test deliberately perturbs a weight and optimizer
  moment, advances RNG, and restores weights, moments, counters and RNG exactly.
  Exported bundles contain only repair weights and optimizer/RNG state and
  require the pinned parent; they are not standalone model checkpoints.
- Both complete W checks pass: **320 tensors**, physical embedding/LM-head tying,
  unchanged frozen receiver/reference fingerprints, cached feature fingerprint,
  and no outstanding gradients outside the permitted optimizer ownership.
- Independent CPU receipt audit recomputes all losses' eligibility counts,
  exposures, summaries/grades, source audits, hashes, ledgers, Choice retention
  and the improvement decision. This is a mechanical audit, not a human review.
  Existing transformer numeric matrices are reused because that code is unchanged.

## Artifacts and accounting

Raw files: `data/receiver-repair-20261107/run/`, with `run.log`, `audit.py`,
`audit.json`, `audit.log` and `cpu-tests.json` in its parent directory. No failed
or retried GPU attempt. The run emits a harmless PyTorch warning when converting
validation-only scalar statistics to Python; it does not detach the training path.

Final bundle `run/checkpoint-2000`, manifest SHA256:
`abc321bb1535656a931e7e5926e01def2417a3d90931235ae996e0a43feb6a8f`.
Repair fingerprint:
`12262d41a02ef9d148a68f0888f038f2a4c67fa56c598121417774d61ff0ceae`.
Parent manifest remains
`0cd1d4bc55341e78339c1b0b05dec0b3650573d7501fb3f3d2fc6d2f649b6e58`;
frozen reference remains
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.

Reused 24 compatible training features. Newly executed:

| Work | Sessions | Causal positions |
|---|---:|---:|
| Missing training features | 979 | 93,964 |
| Fresh final training checks | 4 | 399 |
| Development | 310 | 38,319 |
| Total | 1,293 | 132,682 |

One prefill/lineage per Session, zero held positions and invocation text. The
16,000 cached fit visits and readout measurements add **zero** causal positions
or prefills. Separately: 102,528 offline source tokens and 42,925 reference
annotation tokens across 1,202 new reference encodings. Model-run wall time
**1,619.862349 s**, peak CUDA allocation **3,622,856,704 bytes**; this is not a
matched throughput benchmark. After exit only existing service PID 2094 remains,
using 3,852 MiB. No 3060 work, agents, new annotations/reviews, locked-data access,
native capability training, deployment or Operating Table/service changes.

## Next unfinished work

Keep this failed repair and its now-complete frozen feature cache as evidence;
do not promote or extend it. Value learning/calibration that transfers is still
unfinished. A focused next comparison would remove status gradients from the
same residual fit, holding parent, feature corpus, initialization and exposure
fixed. That would test whether sharing each branch between status and value is
limiting scalar learning; this run does not establish that explanation. Reuse
the saved 1,003 features rather than repeating extraction. Any additional fit
requires new authorization. Reference meaning, learned invocation, Reason
meaning and complete native use remain unqualified.
