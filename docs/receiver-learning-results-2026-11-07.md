# Frozen-reference receiver learning results — 2026-11-07

**The experiment completed. Receiver training paths work and some Choice behavior
transfers; Noul and numeric Score remain unqualified.** No job remains active.
See the [fixed protocol](receiver-learning-v1.md),
[portable evidence](evidence/receiver-learning-v1.json),
[training-path checks](evidence/receiver-training-paths-v1.json),
[exact continuation](evidence/receiver-learning-resume-v1.json) and
[user-requested Astra performance audit](receiver-performance-audit-2026-11-07.md).

## Experiment and implementation

The RTX 3090 completed **188 optimizer windows / 1,504 presentations**: all
**1,500 existing receiver records once**, plus four declared real-record repeats.
There were **zero failed training exposures**. W, R, the budget row, reference
pooler/T/codec and all nonreceiver codec parameters stayed frozen. D and the
specified live receiver/protocol/readout parameters learned; no reference fitting,
writer/gate/reader training, promotion, deployment or service change occurred.
No scenarios, annotations or reviews were generated. One later user-requested
Astra audit was read-only and launched no model workload.

The new isolated component ends after the actual latent request reaches receiver
heads and valid reviewed losses are constructed. This fixes a learning dead end:
an untrained Reason decoder formerly discarded valid status/value supervision.
These are complete **request-to-heads objectives**, with unused return holds
released. They do not supply gold feedback, claim consumed returns or pretend to
complete a native answer. Actual construction/forward/resource failures remain
ineligible. Behavioral evaluation always decodes and consumes actual model output.

All request inputs stay answer-blind; outcome labels are accessed after the
request forward. Reason targets use the separate approved status/support
exception. Reason alignment/count supervision is masked to matching non-assessed
carriers; generic unknown feasibility labels are not invented subclasses.
Frozen reference annotation caching is bit-identical to uncached CUDA outputs;
D-dependent source states are recomputed at current weights for every example.

This is the first complete receiver-learning pass, using the first 188 windows
of the existing A4 component schedule and its 100-window warmup. It is not the
full 2,000-window component phase or a convergence claim. Initial live weights
are the existing diagnostic checkpoint; the final pass-12 reference is retained
unchanged. No alternative codec initialization/control was tested.

## Development outcomes

Every scheduled case stays in its denominator. E is the existing 130-case
receiver allocation across all five statuses; X is the existing 180 applicable
receiver intervention cases / 90 pairs. The 20 category-9 native-reader cases
remain explicitly outside this receiver experiment. Locked labels stayed unopened.
The initial X results were reused after weight/source/view/cache compatibility
checks; E was measured before training. Training/development selected families
are disjoint: **288 receiver-training / 104 E/X development families**.

| Operation | E baseline → final | X baseline → final |
|---|---:|---:|
| Noul | **4 → 13 / 40** | **13 → 9 / 100** |
| Choice | **8 → 11 / 50** | **31 → 40 / 40** |
| Score | **0 → 10 / 40** | **0 → 0 / 40** |

These are complete scoped behavioral successes, including appropriate
non-assessed statuses. **All ten final E Score successes are non-assessed
responses; numeric Score interval agreement is 0/50 assessable cases across
E+X.** Values and intervals were not rounded, widened or relabeled.

The assessed-only results expose the limits clearly:

| Reviewed Assessed operation | Baseline → final successes |
|---|---:|
| Noul | **11 → 0 / 80** |
| Choice | **38 → 47 / 50** |
| Score | **0 → 0 / 50** |

Final Noul returned no correct Assessed status for any of its 80 assessable
cases. Across all 140 Noul development cases, its head predicted Unresolved 121
times, Conflicting four, Assessed five and Insufficient ten. Its five Assessed
predictions occurred on reviewed non-assessable cases. Its E aggregate gain
mainly reflects correct Unresolved responses, not learned held-out truth decisions.

Choice passes **all 40 controlled X cases and all 20 Choice intervention pairs**,
including candidate renaming/reordering and changed acceptable tie sets. This
supports transfer for those tasks. E still has 0/10 Insufficient, 0/10 Conflicting
and 0/10 NoFeasibleChoice successes; broader Choice status/constraint handling is
not qualified. Assessed E Choice selection remains 7/10, while Unresolved rises
from 1/10 to 4/10.

Overall, scoped successes rise **56 → 83 / 310**, returned-status correctness
**115 → 130 / 310**, and complete intervention pairs **16 → 23 / 90**. The
intervention qualification remains **BLOCKED**. Correctly typed exchanges rise
**280 → 310 / 310**: the 30 previous invalid Reason decodes disappear. All final
cases have one prefill, one cache lineage and exact accounting. A valid exchange
does not establish correct meaning. Unsupported Assessed returns decrease
**95 → 46 / 130** reviewed non-assessable cases; this remains a serious failure.
Reason support meaning and native consumption are not evaluated here.

Continuous readout diagnostics also remain weak. These include all reviewed
Assessed cases with a head value, regardless of the returned status:

| Diagnostic | Baseline → final |
|---|---:|
| E Score mean absolute distance to interval, n=10 | **0.439048 → 0.423136** |
| X Score mean absolute distance to interval, n=40 | **0.239921 → 0.233890** |
| E Noul Brier score, n=10 | **0.251636 → 0.253361** |
| X Noul Brier score, n=70 | **0.247474 → 0.289148** |

Score errors remain large and Noul calibration worsens. The existing assessed
Noul training targets contain 168 false / 88 true cases, while X has 20 false /
50 true. This distribution difference is a possible factor to test, not an
established cause of the failure or authorization for automatic class weighting.

## Matched training-example probe

After training, a separate read-only 3090 process restored checkpoint 188 and
revisited exactly the 13 preflight operation/status examples. It performed
**zero optimizer updates** and verified unchanged model/reference/W. Status
accuracy on these fixed training examples rises **3 → 8 / 13**. Mean losses over
examples eligible both before and after are:

| Component | Matched examples | Before → after |
|---|---:|---:|
| Status | 13 | **1.975497 → 0.833762** |
| Noul | 1 | **0.693539 → 0.551364** |
| Choice | 1 | **0.251437 → 0.00000309944** |
| Score | 1 | **0.0230380 → 0.0125381** |
| Feasibility | 5 | **1.782734 → 0.953337** |
| Reason alignment | 1 | **1.974945 → 0.534769** |
| Reason structure | 1 | **5.040627 → 0.00538660** |

These measurements establish learning on the sampled training objectives. One
assessed example per operation is not a training-catalogue recovery score, and
Reason masks changing eligibility cannot be pooled into a flattering comparison.
The training/development gap does not isolate reference information loss,
receiver/codec interpretation, optimization or coverage as the unique cause.

## Integrity and provenance

- **All 320 frozen W tensors / physical tying pass before and after.** Reference
  fingerprint is unchanged; R and frozen codec checks pass at every checkpoint.
  All gradients finish None. Frozen owners are absent from optimizer ownership.
- The preflight checks 13 real operation/status strata with finite connected D
  gradients and no R/reference gradients. There are **25 unique model-free
  checks**, ten CUDA loss/eligibility/accumulation checks, cached/uncached payload
  equality, and exact model/optimizer restore after the first actual update.
- The full ledger audit verifies all **188 ordered windows, 1,504 record/view
  hashes, single-prefill/cache/accounting receipts and eligibility denominators**.
  **1,496 records occur once and four twice**. The charged Session-objective
  total is **160,853 positions**; offline encoding and backward rematerialization
  are separate and are not claimed as physical-compute totals.
- Eligible-example counts: status **1,504**, feasibility **497**, Noul **256**,
  Choice **158**, Score **257**, Reason alignment **248**, Reason structure **248**.
- All files in **ten checkpoints** (0, 1, 25, 50, 75, 100, 125, 150, 175, 188)
  pass hash checks. Every development grade, pair and denominator is recomputed
  from the original reviewed release, and final source snapshots match.

The original outer eligibility Counter incorrectly summed live loss tensors,
retaining graphs and preventing final serialization; Window's optimizer counts
were correct. At checkpoint 75 it was deliberately interrupted, **exit -2**.
The correction counts keys and seeds totals from the saved prefix. Resume checks
all learning-module sources, other identities and the update-loop AST excluding
only bookkeeping; weights, optimizer, RNG, order and cursor continue exactly.
The original log contains 1–75 and the continuation 76–188. **No completed
optimizer window was replayed**; possible in-flight preparation is not claimed
as zero repeated physical work. Continuation and matched probe both **exit 0**.

Raw evidence is `data/receiver-learning-20261106/`, with original/continued source
snapshots, logs, per-case outputs, audits, checkpoint states and probe. Final
checkpoint: `continued/checkpoint-0188`, manifest SHA256
`0cd1d4bc55341e78339c1b0b05dec0b3650573d7501fb3f3d2fc6d2f649b6e58`.
Receiver fingerprint changes from
`4b51fe078f54f7c2852bdd5fb450ba234134f50f69c4dafe365ff286e4d0cd08` to
`94218aae6ec62a7745c349af8e9975fd4987520cf9f1345f38e44586e5e7742c`.
Reference remains
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.

The original process had 3,022 elapsed seconds at the controlled signal; the
continuation reports **5,005.4104 seconds**, including its final evaluations.
Continuation peak Torch allocation is **3,622,856,704 bytes**; this is not a
whole-machine or exhaustive original-segment peak. Existing numeric matrices
were reused; no arithmetic or tolerance changed and no new trained-weight parity
certificate or full-flow promotion is claimed.

## Next unfinished slice

The requested experiment and audit are complete. The next engineering candidate
is fused finite checks at the same validation boundaries, with gradient/failure/
rollback/accounting equivalence, followed by measured checkpoint/cache-policy
comparisons. Astra found one trainer and low CPU contention; repeated VJPs and
synchronizations are the stronger leads. Suggested speedups remain unbenchmarked.

The next learning question is why Noul fits sampled training cases but abstains
on assessable development cases, and why Score remains numerically inaccurate.
Use the saved failures to distinguish input/reference/codec transfer from
optimization and coverage before choosing further training. Preserve the known
existing Reason annotation discrepancy in the protocol; alignment is not a
semantic review. Further fitting, generation or deployment is not queued.
Learned writing/invocation, Reason meaning, full status handling and native answer
use remain unqualified. No reference pass 13 was started.
