# RD-08 matched receiver comparison — completed 2026-11-09

The finite comparison completed. **Useful-improvement acceptance failed.** No
receiver is accepted; RD-09 and deployment remain unstarted. Both arms used all
1,500 existing training examples four times: 6,000 effective presentations and
750 eight-example updates per arm. No training exposure was rejected. Initial
and final full 1,500-case head and actual-delivery panels are preserved per arm.

The direct-state candidate has 28 wins/106 losses/35 both-pass/141 both-fail
against pooled control across all 310 development cases. Both-correct X pairs:
control 30/90, candidate 0/90. Candidate loses all 47 historically passing Choice
cases from the fixed 50-case retention panel (control currently retains 46/47).

[Portable results and all cohort metrics](evidence/receiver-matched-v2-results.json),
[all 310 case outcomes](evidence/receiver-matched-v2-cases.json),
[all 90 intervention pairs](evidence/receiver-matched-v2-pairs.json). Raw reports,
rows, archived sources, plans and checkpoints: `data/receiver-performance-20261109/`.

## Training measurements

Heads and decoded returns are separate. Training input support does not imply
that a predicted return is valid or correct. Denominators include failures.

| Arm / panel | Noul truth head /256 | Noul assessed delivery /256 | Choice head and assessed delivery /158 | Score head and assessed delivery /254 | Score interval distance |
|---|---:|---:|---:|---:|---:|
| control / 0 | 168 | 110 | 158 | 0 | 0.404325 |
| control / 4 | 256 | 256 | 158 | 85 | 0.046883 |
| candidate / 0 | 168 | 168 | 87 | 0 | 0.408468 |
| candidate / 4 | 179 | 170 | 158 | 3 | 0.266343 |

Final whole-panel delivery, including non-assessed cases:

| Arm | Noul /501 | Choice /497 | Score /502 |
|---|---:|---:|---:|
| control | 393 | 414 | 246 |
| candidate | 344 | 409 | 251 |

Actual eligible training counts per arm: status 6000, Noul 1024, Choice 632,
Score 1016, feasibility 1988. New-profile Reason semantics stay unsupported/masked;
structure/alignment/native supervision is not invented. No Reason semantic
learning claim follows from structural return grading. All 1,500 paired startup
readouts and mapped budgets are unchanged in both arms; no missing pairs.

## Held-out actual delivery

Both fits and final measurements finished before either development evaluation.
Each evaluation made exactly 0 optimizer updates and kept its receiver unchanged.

| Arm / panel | Noul | Choice | Score | Total |
|---|---:|---:|---:|---:|
| control / E | 16/40 | 22/50 | 13/40 | 51/130 |
| control / X | 50/100 | 40/40 | 0/40 | 90/180 |
| candidate / E | 13/40 | 16/50 | 14/40 | 43/130 |
| candidate / X | 20/100 | 0/40 | 0/40 | 20/180 |

The predeclared familiar Noul cohorts regress: E control 4/4 versus candidate 3/4,
X 50/50 versus10/50. Candidate truth equals the training-defined constant baseline
(3/4 and 10/50), rather than beating it. The baseline is false, from 168 false / 88
true assessed training labels (mean 0.34375). Familiar Score cohorts contain 0
assessed records, so their required improvement cannot be established. Across
all assessed Score cases, interval distance worsens: E 0.312811→0.492258,
X 0.238843→0.517047. Correct interior delivered Score remains 0/3 E and 0/40 X
for both arms. Several operation/status cells regress; all cells and 27 declared
cohort reports are retained in portable evidence. None of these failures is waived.

| Reference operation/status | Cases | Control status correct | Candidate status correct |
|---|---:|---:|---:|
| Choice/Assessed | 50 | 47 | 1 |
| Choice/Conflicting | 10 | 1 | 6 |
| Choice/Insufficient | 10 | 8 | 5 |
| Choice/NoFeasibleChoice | 10 | 3 | 5 |
| Choice/Unresolved | 10 | 4 | 0 |
| Noul/Assessed | 80 | 76 | 79 |
| Noul/Conflicting | 10 | 3 | 2 |
| Noul/Insufficient | 40 | 4 | 0 |
| Noul/Unresolved | 10 | 4 | 5 |
| Score/Assessed | 50 | 45 | 47 |
| Score/Conflicting | 10 | 5 | 4 |
| Score/Insufficient | 10 | 7 | 7 |
| Score/Unresolved | 10 | 1 | 3 |

Return failures remain in denominators:

- control: E {"no structurally compatible pointer entries": 3, "numeric literal": 2, "reason aggregate bound": 12}; X {"reason aggregate bound": 30}.
- candidate: E {"direct request exceeds 256 positions; never crop": 3, "no structurally compatible pointer entries": 4}; X {}.

The 3 oversized candidate E requests (270/276/295 positions) remain explicit
failures under request256. Other failures are actual predicted-return decoding
failures, including Reason bounds, numeric literals and incompatible pointers;
no gold repair or post-fit decoder change was applied.

## Qualification, matching and recovery

The v2 format represents all 1,500 training records without cropping or new data.
Every original input/field mapping/record/family hash remains bound. Complete
roles use explicit 63-state fields; compact bound returns cost 3 positions for
Assessed, at most 39 with Reason. Request 256 / return 64 and all other bounds stay.
[Coverage](evidence/compact-role-layout-coverage-v2.json),
[format contract](compact-role-states-v2.md) and
[current GPU qualification](evidence/compact-role-qualification-v2.json) remain.

All 19 affected-path 3090 checks pass with 0 optimizer updates; 35 model-free checks
pass. The six paired performance windows and exact arithmetic/resume tests pass.
Qualification commit 6793f252b4259ff2a19bc0ffb54e349a68374d2c was pushed and its
exact remote SHA verified before recovery fitting. No required GPU check remains
pending. Historical failed qualification attempts and their bounded repairs are
preserved in prior reports/raw logs; they are not silently relabeled.

Shared initial masters, Adam moments, CUDA/Python RNG match exactly. Candidate
profile-only projection starts zero-delta identity with fresh absent moments.
W/R/startup/reference stay frozen; all 320 W tensors, physical tying, reference
and nonreceiver owners and grad=None pass after fitting and evaluation. All 95
bound sources match current/archived bytes; no source/plan/qualification edits
occurred during fitting. Initial/final/training input hashes and masks match,
and every current-process episode matches the fixed planned order.

Only 500 of 599 previously logged control updates were recoverable. The compatible
checkpoint migration changed metadata only: all weight/Adam/RNG bytes and
sampler state were preserved. Remaining 250 effective windows completed; 99 lost
windows / 792 presentations were replayed with exact original input/head/loss/
eligibility/update receipts. Recorded physical control calls 849 versus 750
retained are disclosed; unflushed work immediately before process loss is unknown.
The 12 discarded performance updates are separate. Original baseline 1500 rows
were copied unchanged with original provenance, without a model rerun.

| Arm | Final checkpoint | Manifest SHA256 |
|---|---|---|
| control | `data/receiver-performance-20261109/control/checkpoint-0938` | `f616f4c9154ff2a4c4b7d5b58b5f8d67b9afd55baf80d758e5e234a0b3073452` |
| candidate | `data/receiver-performance-20261109/candidate/checkpoint-0938` | `7b12b33db61563456ac0b203d9851c9b5a7e512aa337e77418106818397f617e` |

Both final manifests store historical counters 938 and sampler cursor 6000;
all four files verify. RD08 contributes 750 retained updates, not 938 new updates.

## Performance limitation and reproducibility

The measured window speedup is 24–26%, with 35–37% lower CPU time. It does **not**
establish removal of the CPU bottleneck. A short candidate 459 sample used 1.025
CPU-core equivalents; whole-host busy 9.30% on 12 logical CPUs. An instantaneous
GPU reading 100% is not a utilization distribution. The user requires that CPU
keep up with GPU; no CPU cap was requested/applied. No further rewrite, training
extension or promotion is implied. See [performance report](receiver-performance-v1.md).

Candidate fit setup+750 windows+both full measurements:26,688.315s; peak allocation
4,902,695,424 bytes. Recovered control setup+250 windows+final measurement:
11,777.648s, peak3,685,016,064 bytes. These scopes differ and are not a matched
whole-run speed comparison. Read-only E/X evaluations took 734.973s control and
594.103s candidate. Reproduction argv/environment and source/plan/report/checkpoint/row
hashes and all before/after metrics are in portable evidence. Reduction command:

```sh
PYTHONPATH=. python data/receiver-performance-20261109/analyze.py
```

This reduction blocks model imports and regrades every case using immutable
released annotations. Final metadata verification also checks exact source bytes,
planned order, inputs/masks, unchanged evaluated receiver and final checkpoint
file hashes. No new data, semantic review, locked access, RD09, deployment,
service/fan change or CPU model execution occurred. Execution is complete;
commit/push delivery status is tracked in docs/redesign-progress.md.
