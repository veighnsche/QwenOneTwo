# Slice 3: simpler latent-input comparison — 2026-11-07

**Completed; reject this candidate as a replacement with the frozen receiver.**
Ordered full-width means reduce overall behavioral success **49/180 →31/180**,
Choice **40/40 →11/40** and complete intervention pairs **23/90 →0/90**. Score
remains 0/40. Higher status accuracy is dominated by returning Assessed and does
not establish better reasoning. No parameters were fitted, changed or promoted.

See the [predeclared protocol](simple-latent-comparison-v1.md) and
[portable evidence](evidence/simple-latent-comparison-v1.json).

## One controlled alternative

Profile `ordered-full-width-means-v1` averages disjoint, ordered bins of the same
frozen-W pre-pool states, then applies the existing FP32 normalization. It removes
the learned reference pooler's 256-wide projection, attention and lift, retaining
1024 width during pooling. It still compresses tokens to the existing role slots;
**this is not an uncompressed-input or increased-capacity test**. Template tokens
remain included. No token selection, new annotations or learned projection is added.

Both paths use parent checkpoint-0188 receiver
`94218aae6ec62a7745c349af8e9975fd4987520cf9f1345f38e44586e5e7742c` and reference
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.
W, D/R, codec, scalar rows, prototypes and reference remain frozen. Only the
candidate semantic payload basis changes. The prepared-draft R tick naturally
sees the changed semantic records too; no fixed hidden prefix at the D boundary
is claimed. No return/answer is substituted, and no threshold is changed.

All 160 pre-pool/pooled captures and all 180 control receipts are reused from
slice 2, after source, artifact, view, role and checkpoint checks. Three fresh
control anchors (one per operation) exactly reproduce outputs, full canonical
ledgers, wire vectors and final h. All 180 candidate cases execute once through
the original packet, runtime, decoder and result-consumption path.

Candidate provenance identifies the captured pre-pool tensor, new profile and
token bins, and sets `reference_target_compatible=false`. These tensors cannot
be presented as frozen-reference targets or a semantic qualification. The new
encoder rejects gradient-enabled use; production and training defaults are unchanged.

## Actual development results

These are the same **180 X cases /90 correlated intervention pairs**, not a new
independent evaluation sample. E, native-reader and locked cases were not run.

| Measure | Learned pooling control | Ordered means candidate |
|---|---:|---:|
| Complete behavior | 49/180 | **31/180** |
| Status correctness | 89/180 | 149/180 |
| Complete intervention pairs | 23/90 | **0/90** |
| Valid transport / execution failures | 180/180 /0 | 180/180 /0 |
| Noul complete behavior | 9/100 | 20/100 |
| Noul assessed delivered success | 0/70 | 20/70 |
| Noul independent truth head | 20/70 | **20/70** |
| Noul Brier, assessed cases | 0.289148 | 0.293918 |
| Choice delivered selection | 40/40 | **11/40** |
| Score original interval agreement | 0/40 | **0/40** |
| Score mean distance to original interval | 0.233890 | 0.241228 |

**Noul's apparent gain is status behavior, not a better truth head.** The candidate
predicts below 0.5 on all 100 Noul cases (range **0.413115..0.418366**) and calls
every case Assessed. It therefore delivers the 20 false assessed labels correctly,
while all 50 true labels remain wrong. It also incorrectly assesses **all 30**
Insufficient cases, versus three such unsupported assessments in the control.
The control's other non-assessed responses were not all correct either: only
nine correctly returned Insufficient. These nine successes are all lost.

Choice loses 29 previously correct cases and gains none. Of 40 Choice cases,
39 return Assessed and one Conflicting. No selection is delivered in that one
case, so the independent selection report has 39 measured values; it remains in
the 40-case denominator. Do not impute an unobserved selection as a raw-head result.

The candidate gains 20 Noul cases, loses nine Noul and 29 Choice cases, retains
11 successes and fails both paths on the remaining 111 cases. All failures remain
in denominators. Exact values, identities, bindings, thresholds and Score intervals
are unchanged.

| Existing intervention, ten pairs each | Control both-cases pass | Candidate both-cases pass |
|---|---:|---:|
| Negation / eligibility change / removed evidence | 0 /0 /0 | 0 /0 /0 |
| Withheld external fact | 3 | 0 |
| Candidate rename/permutation | 10 | 0 |
| Candidate property change | 10 | 0 |
| Reversed rubric / numeric scale / withheld misleading return | 0 /0 /0 | 0 /0 /0 |

Value-change direction is a weaker diagnostic: Noul negation **9/10 →3/10**,
eligibility **8/10 →1/10**, Score reversal **5/10 →4/10**, Score scale **0/10 →7/10**.
The scale-direction improvement does not yield interval correctness or lower
mean Score error. Neither behavioral configuration meets existing acceptance.

## Preserved mechanisms and numeric checks

- **180/180** cases retain identical permitted inputs, source byte/offset/tensor
  lineage, request role order/length, startup budget and pre-request accounting.
  Every nonsemantic wire row is bit-identical to the saved control.
- The **20 identical-permitted-input pairs** retain bit-identical candidate
  wire/h. Mean relative L2 displacement from the control is 0.451676 at the wire
  and 0.624408 at h; these are representation shifts, not semantic quality scores.
- All **160** role payloads cover every source state exactly once, retain slot
  shapes and match independent GPU FP64 means/normalization. Maximum relative
  RMS error **8.288918964e−8**, below the unchanged **1e−5** FP32-reference limit.
  Maximum squared slot-RMS deviation from one **2.384185791e−7**.
- All **320 W tensors and physical tying**, receiver/reference fingerprints,
  cached captures, frozen ownership and `grad=None` pass before/after checks.
  No backward, optimizer or fitted probe runs. V3 transformer arithmetic and
  production numeric/backward implementations are unchanged; prior matrices reused.
- **10 model-free tests pass in 0.189 s**, including complete bin coverage,
  forbidden model imports, replay lifecycle, input/accounting/source rejection
  and paired failure denominators. Compilation and whitespace checks pass.
- A separate model-free audit rechecks all cases/anchors, grades, paired outcomes,
  source snapshots, position sums and **160 actual saved payload byte hashes**
  against their construction receipts, without CPU tensor/model execution.

## Accounting, runtime and artifacts

Candidate **180 Sessions /20,856 positions**; control anchors **3 /374**. Total
new execution: **183 Sessions /21,230 positions**. Each has one prompt prefill,
one cache lineage, exact P+B+A+H conservation and zero remaining holds. Different
actual result lengths explain the changed total compared with the old control.
Reused control receipts incur zero new positions. No invocation text tokens.

Offline source encoding is separate: **14,212 candidate +253 anchor tokens**.
There are **zero new reference encodings and zero reference-pooler calls**;
all 160 state records are reused. Ordered means are tensor arithmetic, not an
additional causal episode. Nothing is replayed inside a live cache.

Model-run wall **195.108165 s**, peak CUDA allocation **3,622,856,704 bytes**.
This measured region includes loading, integrity checks, pooling, anchors,
candidate execution and summaries; it excludes initial CPU preparation and
final artifact serialization. It is not a matched throughput benchmark.

Raw root: `data/simple-latent-20261107/`. `run/` contains plan/source snapshots,
anchor and candidate receipts, per-case differences, candidate payload/feature
tensors and report. `audit.py`, `audit.log`, `cpu-tests.log`, `run.log` and portable
hashes preserve the actual validation. No failed/retried GPU attempt occurred.

After exit, only original service PID **2094**, **3,852 MiB**, remains on the
3090. The 3060 was unused. No agents, new training data/reviews, locked-data
access, service/deployment/Operating Table changes or training occurred.

## Decision and next unfinished slice

Keep the learned-pooling control. This one simpler fixed-capacity path is **not
a useful replacement with the existing receiver**. It changes status behavior
and breaks the previously useful Choice behavior; better aggregate status
accuracy must not be mistaken for semantic progress.

The receiver was trained with the original representation. This comparison
therefore leaves **representation information versus receiver adaptation**
unresolved. It does not establish that ordered means could never learn, that
the original reference is semantically faithful, or that more slots would help.

Slice 4 remains the receiver-learning design/repair: specify a controlled learning
comparison informed by the observed Noul threshold/status problems and Score
coverage/transfer gaps, preserving Choice. Any adaptation or fitting requires
new authorization; no extra candidate, training run or data generation follows
automatically from this completed slice.
