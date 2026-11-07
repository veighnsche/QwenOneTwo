# Receiver training-panel probe — 2026-11-07

The authorized probe completed **32 optimizer updates** on the RTX 3090, using
existing data and frozen W/reference. It improves status handling, preserves
Choice assessed success, and reduces training Score error, but **does not solve
Noul truth discrimination or Score interval agreement**. No weights were promoted.
Read the [fixed protocol](receiver-training-probe-v1.md) and
[portable evidence](evidence/receiver-training-probe-v1.json).

## Implemented and executed

The learning report separates assessed value accuracy, status, joint success,
calibration, eligible loss counts and actual delivered behavior. Detached head
telemetry is recorded after the real latent request and before reading labels.
An AST comparison against starting commit `6e2e96f` confirms that removing this
telemetry leaves the producer objective unchanged. No input, packet, label, loss,
coefficient, optimizer, numeric tolerance or accounting rule was changed.

The deterministic training-only panel contains 32 distinct families: eleven
Noul, eight Choice and thirteen Score records, covering all thirteen operation/
status cells. Its assessed Noul labels are balanced four true/four false.
Development E/X families are disjoint. Eight complete repetitions give **256
presentations**, eight exposures per record, **32 eight-example windows at
cumulative positions 189..220**, and **zero failed training exposures**.

Eligible counts: status 256, Noul 64, Choice 32, Score 80, feasibility 64,
Reason alignment 74 and structure 74. The 26,656 session positions obey exact
P+B+A+H accounting, one prefill/cache lineage and zero held positions. This count
excludes offline clean-reference encoding and backward rematerialization.
Every recorded invocation uses zero text tokens and no gold return.

Checkpoint-0188 weights, optimizer moments, RNG and schedule are restored under
their original identities, then forked into an explicitly new probe protocol and
source identity. There is no warmup reset or claim that the previous sampler
already traversed this panel. Saves exist at 188, 189, 192, 196, 204 and 220.
The first-update restore exactly reproduces receiver weights, optimizer tensors
and CPU/CUDA/Python RNG. All checkpoint file hashes pass independent audit.

## Training panel: head measurements only

These measurements use the same panel and source views at every checkpoint.
They end before feedback delivery; they are not behavior qualification.

| Repetition | Noul status /11 | Noul truth /8 | Choice status /8 | Choice selection /4 | Score status /13 | Score interval /10 | Score mean interval distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 4 | 6 | 4 | 12 | 0 | 0.411528 |
| 1 | 11 | 4 | 6 | 4 | 12 | 0 | 0.399025 |
| 2 | 11 | 4 | 8 | 4 | 13 | 0 | 0.394757 |
| 4 | 11 | 4 | 8 | 4 | 13 | 0 | 0.316655 |
| 8 | 11 | 4 | 8 | 4 | 12 | 0 | 0.166759 |

Noul Brier improves **0.256381 → 0.247584** and BCE **0.706077 → 0.688231**,
but every final panel Noul raw logit remains negative (−0.249054..−0.166390).
It still predicts false for every assessed example. Score interval loss falls
**0.206481 → 0.053827**, with zero assessed clipping at all five measurements,
but still no interval hits. The observed reduction in error is evidence of
connected learning, not successful task learning or convergence. Choice already
gets all four assessed selections correct before updates.

## Held-out actual decoded and delivered behavior

All 130 E and 180 applicable X cases run after the final update, with no further
steps or receiver changes. Baseline numbers below reuse the hash-verified saved
checkpoint-0188 receipts; the parent was **not newly evaluated**. The twenty
native-reader X cases stay outside this receiver experiment; locked labels stay
unopened. Intervals and exact identity binding use the existing grader unchanged.

| Panel | Noul passed | Choice passed | Score passed |
|---|---:|---:|---:|
| E, baseline → probe | 13 → 18 /40 | 11 → 14 /50 | 10 → 5 /40 |
| X, baseline → probe | 9 → 50 /100 | 40 → 40 /40 | 0 → 0 /40 |
| Combined assessed only | 0 → 23 /80 | 47 → 47 /50 | 0 → 0 /50 |

Overall behavior is **83 → 127 /310** and status is **130 → 230 /310**.
All **310/310** transports pass; there are no failed evaluation cases.
Non-assessed successes are reported separately in the evidence and are never
counted as assessed value learning.

The Noul gain comes from better status: assessed status improves **0 → 76 /80**,
while independent raw truth correctness stays **25/80**. Brier improves
**0.284675 → 0.273479**. This does not show better binary truth discrimination.
Choice assesses 49/50 eligible cases and delivers 47 correct selections in both
runs; selection availability is explicit, not unconditional raw pointer accuracy.
Score assessed status improves **45 → 48 /50**, but assessed behavior remains
0/50. Combined Score mean interval distance improves **0.271739 → 0.261081**;
E worsens **0.423136 → 0.447060**, while X improves **0.233890 → 0.214586**.
The regression in E Score abstention handling reduces its total success to 5/40.

X paired success improves **23 → 30 /90**, with 180/180 statuses correct after
the probe. The existing semantic/behavior qualification still reports **BLOCKED**:
overall/per-operation correctness, Score agreement and intervention pairs fail
their existing thresholds. Transport success and status fitting do not override
these measured behavior failures.

## Gradient observations and next unfinished slice

Existing VJP buffers are observed in the first and last four windows without
changing combination or optimizer math. In the first repeat, coefficient-weighted
eligible-mean D norms are status **9.692..17.558**, Noul **0.176..0.923** and Score
**1.400..2.217**; feasibility reaches **24.001**. In the last repeat, status falls
to **0.236..3.265**, Noul is **0.413..1.263**, and Score rises to **6.452..7.922**.
All eight sampled combined norms exceed the unchanged clip of 1
(**28.406..58.193**). Noul/status D cosines are predominantly small/negative;
Score/status cosines remain near zero. These are window aggregates, not proof
that one loss causes the failures; Adam moments and shared representations matter.

**Next unfinished slice:** a controlled value-learning comparison on this same
panel, testing objective balance against the unchanged baseline from the same
parent and optimizer state, with fixed exposures and all-operation retention
measurements. Specify that comparison before execution; this run does not
authorize extra updates or prescribe a coefficient as a proven fix. The immediate
question is whether Noul can separate true/false and Score can enter the reviewed
intervals without losing status/Choice behavior. Full-catalogue training, learned
invocation, native answer use and Reason meaning remain unqualified.

## Integrity and reproducibility

- GPU: `GPU-225509c3-d702-6c27-de03-76b5e99b33b1`; established training backend,
  FP32 transformer and compensated BF16 projection, frozen BF16 W storage.
- All **320 W tensors and physical tying** pass before/after. Reference, R,
  budget/nonreceiver codec and grad=None checks pass. The final receiver remains
  identical throughout held-out evaluation. No base updates or reference fitting;
  checkpoint bundles retain the frozen reference and omit W/R base owners.
- **21 CPU metadata/report/receiver/behavior tests pass in 29.697 s**, with model
  runtime imports blocked. Independent receipt/source/checkpoint/report audit
  passes. Existing numeric/gradient qualifications are reused: no model arithmetic
  or objective math changed, and no expensive matrix is rerun.
- Model run: **1856.238311 s**; peak CUDA allocation **3,622,856,704 bytes**.
  This is whole-probe wall time, not an isolated throughput benchmark.
- Final process inventory contains only the existing 3090 service PID **2094**,
  **3852 MiB**. No service, deployment or Operating Table changes; no agents,
  generated data or new content reviews.

Raw plans, executed sources, training receipts, five panel files, E/X receipts,
checkpoints, audit, test logs and command/exit records:
`data/receiver-probe-20261107/` (ignored; preserved locally).

Final checkpoint: `data/receiver-probe-20261107/run/checkpoint-0220`, manifest
`ce1ab2bbcc946846a7b052c7f32cdc0a3e208dc6698e135cac7aa994bad15a41`.
Receiver: `5eabb8e580144bf0829a5311528a307671bc6568141e489d7277ae5e7fcd329d`.
Frozen reference: `5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.
The portable evidence binds original releases, parent, sources, raw receipts and
all checkpoint manifests. The checkpoint is **COMPLETED_PROBE_NOT_PROMOTED**.
