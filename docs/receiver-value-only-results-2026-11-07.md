# Matched value-only comparison: no development value improvement

The authorized [comparison](receiver-value-only-comparison-v1.md) completed on
the 3090. Removing status loss improved Noul's training probability loss, but
**did not improve development truth or numeric accuracy**. Delivered behavior
regressed relative to the joint status/value control. Choice stayed exactly
unchanged. No checkpoint was promoted or training extended.

[Portable evidence](evidence/receiver-value-only-comparison-v1.json) contains
the pinned control, source/data hashes, complete metrics and receipt audit.

## What was matched

Used the same 1,003 saved feature tensors, frozen parent-0188 and reference, and
the same 133,184-parameter residual architecture. Original initial **weights,
optimizer and RNG files are byte-identical** in the new checkpoint-0000.
All initial heads, losses and metrics match for all 1,003 records. All model,
loss, optimizer, schedule and ownership source files match the archived control;
only orchestration gains the explicit comparison mode.

The only training intervention was removing status loss before eligible-count
accumulation. The complete **2,000-window /16,000-visit** order is identical:
955 records have 16 visits, 48 have 15. Noul eligibility remains **4,087**, Score
**4,049**, status zero. Ten predetermined windows have no scalar targets and
correctly perform no optimizer step, yielding **1,990 updates**. Inactive branches
have grad=None and receive no moment/decay update under the existing rules.
No replacement exposures, schedule changes, resampling or changed loss weights.

The same forward path still feeds repaired h to both status and scalar heads;
removing status supervision does not freeze predicted statuses. Reason/structure
use their original features and Choice bypasses the repair. W, R/D, all existing
codec weights and reference remain frozen. Exact values, bindings, Noul threshold,
Score row/arithmetic, budget/S-curve and tolerances are unchanged.

## Development: parent, joint control, value-only

All 130 E and 180 X cases execute actual latent requests and returned results,
with zero execution failures. Only the final scheduled checkpoint is evaluated.

| Measurement | Parent 0188 | Joint control | Value-only |
|---|---:|---:|---:|
| Overall behavior | 83/310 | 123/310 | 78/310 |
| Correct status | 130/310 | 182/310 | 129/310 |
| Noul assessed truth, independent of status | 25/80 | 25/80 | 25/80 |
| Noul assessed delivered success | 0/80 | 9/80 | 4/80 |
| Noul Brier | 0.284675 | 0.299799 | 0.302809 |
| Noul descriptive AUC | 0.600727 | 0.675636 | 0.638545 |
| Choice assessed delivered success | 47/50 | 47/50 | 47/50 |
| Score assessed exact delivered success | 0/50 | 0/50 | 0/50 |
| Score raw interval hits | 1/50 | 0/50 | 0/50 |
| Score mean interval distance | 0.271739 | 0.285469 | 0.288103 |
| Complete intervention pairs | 23/90 | 30/90 | 20/90 |
| Valid transport | 310/310 | 310/310 | 310/310 |

All **90 Choice cases exactly match both controls**, including actual outputs,
status/value telemetry, input hashes, grading, transport flags and full canonical
ledgers. This preserves existing Choice performance, not universal qualification:
overall Choice behavior remains 51/90, with 40/40 X successes.

Value-only Noul predicts false on all assessed development cases: **0/55 true
positives, 25/25 true negatives**, balanced accuracy 0.5, no probability ties.
Positive probabilities span 0.213738..0.437073 and negative probabilities
0.089459..0.426196. The ranges overlap, and neither threshold accuracy nor
calibration improves. No cutoff was fitted on development or changed.

Score distance worsens versus the joint control in both direction groups:
higher **0.304833 →0.306424** (40 cases), lower **0.208016 →0.214822** (ten).
Exact delivered success stays zero. Missing reversed training examples cannot
alone account for failure on higher-scale cases. All assessed training Score
targets remain the original increasing-scale point annotations.

## Training comparison

| Measurement | Joint control | Value-only |
|---|---:|---:|
| Noul assessed truth | 168/256 | 169/256 |
| Noul BCE | 0.626934 | 0.507407 |
| Noul Brier | 0.217993 | 0.174319 |
| Noul descriptive AUC | 0.767519 | 0.763934 |
| Noul status | 420/501 | 358/501 |
| Score exact interval hits | 0/254 | 0/254 |
| Score interval-distance loss | 0.184063 | 0.183422 |
| Score mean interval distance | 0.393886 | 0.393377 |
| Score status | 502/502 | 417/502 |

The extra correct Noul training decision is a net result: eight true positives
gained and seven true negatives lost (8/88 and 161/168 correct respectively).
Better training probability loss is real, but it does not establish transferable
truth learning. Score training changes little even without status loss.

**Interpretation:** removing status supervision changes the learned probabilities
and hurts status handling, but does not solve either operation's development
values in this setup. Status competition alone is not a sufficient explanation
or fix. This single matched comparison is not an impossibility or convergence
result and does not locate the remaining limitation between frozen representations,
readout structure, optimization and data coverage. Do not promote or extend it.

## Validation, artifacts and accounting

**20 CPU tests pass in 13.860 s**, without CPU model execution. Byte compilation
and whitespace checks pass. Targeted GPU scalar loss/VJP checks match the real
full codec exactly for both operations, with nonzero gradients in the selected
branch and no gradients in the other. Reuse the unchanged residual FP64 and
transformer numeric qualifications; no repeated matrix or pooling comparison.

Initial state/readout matching, deliberate first-update weight/moment/RNG
perturbation and exact resume, four fresh final h/readout/ledger checks, frozen
receiver/reference and feature fingerprints, both complete **320-tensor W**
checks/physical tying and gradient ownership all pass. Independent CPU audit
recomputes every exposure and eligibility count, active optimizer owners, all
ten empty windows, summaries/grades, provenance, accounting and Choice retention.
This is a mechanical receipt audit.

Raw artifacts: `data/receiver-value-only-20261107/run/`; logs, `audit.py`,
`audit.json`, `audit.log` and CPU test evidence in its parent. The model-free
`prepared/` directory records preflight only, not another fit. No failed/retried
GPU attempt. The launcher retains an `extraction` progress label while checking
cached features; **no bulk extraction actually executes** in comparison mode.

Final `run/checkpoint-2000` manifest:
`7d719824ec545cd9f12489cf61a9c231bc79085111a49501d65b591e82b32ffc`.
Repair fingerprint:
`f98e0f742d105e553666e02a045ae1a251f0890a1b3145e3c0e24632cfba7245`.
It requires the exact parent checkpoint and explicit experimental installation;
it is not a standalone or default model.

New execution: **4 final training-check Sessions /399 positions**, plus
**310 development Sessions /38,211 positions**, total **314 /38,610**. Every
Session retains one prefill/cache lineage and exact accounting, no held positions
or invocation text. All 1,003 training features are reused byte-for-byte. Cached
fit/measurement adds zero prefills or causal positions. Separate offline work:
**26,551 source tokens**, **16,316 annotation tokens /483 reference encodings**.
Model-run wall **479.425140 s**, peak CUDA **3,622,856,704 bytes**. This elapsed time
includes evaluation and differs in extraction work from the control; it is not
a matched throughput benchmark. After exit only original service PID 2094 remains,
using 3,852 MiB. No 3060, agents, new data/reviews, locked-data access, reference or
native capability training, deployment or Operating Table/service changes.

**Next unfinished work:** locate the remaining scalar information/transfer versus
readout-optimization limitation using the saved feature corpus. A separately
specified direct-readout diagnostic could distinguish those factors before
another coupled receiver fit; any additional fitting needs new authorization.
Do not repeat this status-loss comparison or bulk extraction by default. Reference
meaning, Reason meaning, learned invocation and complete native use remain unqualified.
