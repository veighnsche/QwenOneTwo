# Architecture slices 1–2: measured information diagnostics — 2026-11-07

**Completed, read-only, no training or promotion.** The observations do not
support identical-vector collapse as a general explanation of these failures.
They expose a training-threshold problem alongside failed Noul generalization,
and concrete limitations in Score/reference training coverage. Semantic fidelity
of the reference and complete learned invocation remain unqualified.

Read the [architecture/data baseline](architecture-baseline-2026-11-07.md),
[predeclared protocol](information-loss-v1.md), and
[portable evidence](evidence/information-loss-v1.json).

## What ran

On the explicitly selected RTX 3090, restore parent checkpoint-0188 receiver
`94218aae6ec62a7745c349af8e9975fd4987520cf9f1345f38e44586e5e7742c` and reference
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`. Freeze W,
D/R, codec, heads and reference. No backward, optimizer, classifier, threshold
or regression probe was fitted. The saved head-only final heads are read as
immutable tensors for replay; they are never installed into the live receiver.

- Reuse **32 hash-bound training feature records**, with zero new training
  Sessions/positions. Parent and final scalar readouts reproduce saved values
  and raw logits exactly. Features and their frozen representation fingerprint
  remain unchanged.
- Capture **180 X development cases /90 pairs** through original clean inputs,
  actual latent packets and actual returned results. Every case's outcomes,
  permitted view, transport checks and full canonical ledger **exactly match**
  saved parent receipts. Saved final scalar-head values also match on all **140**
  Noul/Score cases, without another backbone pass.
- Capture full pre-pool annotation states, pooled slots, actual wire vectors,
  EVALUATE h, q(h), status logits and candidate states/raw scalar outputs.
- **160 unique reference encodings /480 cache hits**. Offline reference tokens
  **5,894**, offline source tokens **14,212**, kept separate from Session charges.
- Successful run: **180 new Sessions /21,244 positions**, each one prefill and
  lineage, zero held positions after completion, exact P+B+A+H conservation and
  **zero invocation text tokens**. Cached replay is not an additional episode.
- Model-run wall **215.737326 s**, peak CUDA allocation **3,622,856,704 bytes**.
  These include loading/checks/analysis within the runner's measured region;
  they exclude earlier CPU plan preparation and the failed first attempt.

## Noul: the training head separates classes but misses the fixed threshold

The eight balanced training examples reuse the exact features from the prior
head-only experiment; no new learning occurred here.

| Readout / samples | AUC | Correct at unchanged threshold |
|---|---:|---:|
| Parent / eight training examples | 0.500 | 4/8 |
| Saved final head / same eight | **1.000** | **4/8** |
| Parent / 70 assessed X cases | 0.600 | 20/70 |
| Saved final head / same 70 | **0.373** | **20/70** |

Final training false logits span **−0.318092..−0.134084**; true logits span
**−0.080566..−0.013014**. Every true example ranks above every false example, but
**all logits are negative**. The previous unchanged 4/8 accuracy therefore does
not mean the head learned no distinction. It failed the fixed zero-logit/0.5
probability decision threshold. No replacement threshold was fitted or applied.

The development true/false ranges overlap, and ranking worsens after the saved
head-only fit. A training-panel threshold adjustment would not establish a
generalization fix. AUC describes these correlated samples; it is not a
population estimate or independent significance test.

## Input changes survive the measured boundaries

Of 90 intervention pairs, **20** have identical permitted inputs (withheld facts
or misleading supplied returns). Their wire vectors, h and q are **bit-identical**.
All **70** changed-input pairs have different final h. Among **150 changed
corresponding role-input comparisons**, **zero** produce identical pooled slots.

| Intervention, ten pairs each | Median pooled-question relative L2 | Median final-h relative L2 |
|---|---:|---:|
| Negated proposition | 0.230495 | 0.070715 |
| Changed eligibility rule | 0.361328 | 0.115924 |
| Reversed rubric/direction | 0.665006 | 0.500107 |
| Changed numeric scale/anchors | 0.668087 | 0.451042 |

Relative L2 is ||a−b|| / max(||a||,||b||). Full variable-length pre-pool states
are retained; unequal shapes are not falsely compared as aligned sequences.
Mean/last-state comparisons are explicitly lossy summaries. Candidate fields are
compared by original ordinal/role; permutations are not mislabeled corruption.

These observations reject **exact collision** on these pairs. They do **not**
prove that the correct meaning survives compression, that all useful distinctions
survive, or that the receiver can decode them. Larger distances across different
representations are not a measure of greater semantic information.

## Conditioning and behavioral direction

Centered sample ranks at the declared FP32-scale cutoff:

| Features | h rank | q rank |
|---|---:|---:|
| Eight assessed training Noul | 7/7 possible | 7/7 |
| Ten assessed training Score | 9/9 possible | 9/9 |
| Forty assessed X Choice | 39/39 possible | 39/39 |
| Forty assessed X Score | 39/39 possible | 39/39 |
| Seventy assessed X Noul | 49 | 49 |

Noul X has **50 unique permitted inputs** among its 70 assessed cases; its rank
49 is maximal for 50 centered unique rows and must not be interpreted as 70
independent observations. Training retained condition numbers are about
2.32 (Noul h) and 3.99 (Score h). Both q/k weight matrices have full row rank
**256/256**, with conditions **2.958 /2.960**. This does not suggest a grossly
singular projection or collapsed training panel; it is not a separability or
semantic certificate. Centered spectra are sample-limited; full singular values,
cutoffs and stable ranks are saved.

Correct direction of scalar change is weaker than correct values/status:

| Existing intervention | Parent raw change | Saved final head-value change |
|---|---:|---:|
| Noul negation | 9/10 | 2/10 |
| Noul eligibility rule | 8/10 | 7/10 |
| Score reversed rubric | 5/10 | 5/10 |
| Score numeric scale | 0/10 | 10/10 |

The final-head direction column is a post-hoc descriptive aggregation of existing
saved values, all reproduced by the new frozen-feature replay. It involves no
additional model execution or fitting. The Score head can improve scale-change
direction while its previously measured development absolute error worsens.
Neither column changes any original success criterion.

Parent behavioral results remain **49/180**, status **89/180**, pairs **23/90**:
Noul 9/100, Choice 40/40, Score 0/40. All **180/180 transports** pass. These are
the same old failures with new intermediate evidence, not a capability gain.

## Validation, failed attempt and reproducibility

- **14 model-free tests** pass: ten baseline/report/ownership tests in 14.330 s,
  four diagnostic tests in 0.095 s. Diagnostic imports are tested with model
  libraries blocked. Byte compilation and `git diff --check` pass.
- CUDA analytic checks validate rank-two/zero matrices and exact differences.
  Every production forward is unchanged and compared with saved receipts.
  No backward implementation changed; completed numeric/gradient matrices were
  reused rather than rerun.
- Before/after checks pass for **all 320 W tensors and physical tying**,
  receiver/reference fingerprints, cached features, frozen ownership and
  `grad is None`. Independent CPU receipt/source/accounting/report audit passes.
- Initial attempt stopped after its first request on a new comparator bug:
  live ledgers used tuples, saved JSON used lists. Canonical serialization fixes
  this without dropping fields or loosening tolerances. Failed log/report and
  partial reference captures are retained. It has zero validated cases and no
  retained case ledger; **do not claim an exact combined physical-position total
  across attempts**. It is separate from the successful run's 180 Sessions.
- After exit, only original service PID **2094**, **3,852 MiB**, remains on the
  3090. The 3060 was unused. No new data/reviews, agents, service or Operating
  Table changes, deployment, model promotion or locked-data access occurred.

Raw root: `data/information-loss-20261107/`; successful run: `run-v2/`.
`captures.safetensors` and `capture-index.json` retain tensor/role lineage;
`cases.jsonl`, `pairs.json`, `spectra.json`, source snapshot, plan and report
retain exact inputs, measurements and identities. The existing runner and
model-free `audit.py` reproduce capture and report validation respectively.

## Next unfinished slice

Slice 3 remains a **matched comparison with one simpler source-grounded latent
input profile**, not an automatic pooling rewrite. These results require its
hypothesis to distinguish useful semantic interpretation and transfer from mere
vector differentiation. Keep the current representation as the control.

Address the observed reference/Score coverage limitations in that comparison's
design, using existing data first. Inspect Noul threshold behavior separately
from development ranking, and Score direction separately from calibration and
exact target fidelity. The current evidence does not isolate which combination
of representation, receiver learning, readout constraints or training exposure
causes the remaining failures. No subsequent fitting or architecture change
was run or is authorized by this completed diagnostic.
