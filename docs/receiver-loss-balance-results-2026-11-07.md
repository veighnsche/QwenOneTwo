# Receiver loss-balance comparison — 2026-11-07

The controlled comparison completed on the RTX 3090. **The tenfold reduction of
status/feasibility weights is not an overall improvement under this protocol.**
Noul training probabilities separate more, but binary accuracy does not improve;
Score's early training advantage reverses by the final checkpoint. Held-out
success regresses **127 → 123 /310**. Keep production coefficient defaults and
live weights unchanged. No checkpoint was promoted.

Read the [predeclared contract](receiver-loss-balance-v1.md),
[control results](receiver-training-probe-results-2026-11-07.md) and
[portable evidence](evidence/receiver-loss-balance-v1.json).

## Matched experiment and implementation

The saved completed probe is the control; it was **not rerun**. The candidate
starts independently from the same checkpoint-0188, with byte-identical initial
weights, optimizer moments, CPU/CUDA/Python RNG and effective private rows.
Optimizer ownership/settings are identical. All executable source identities
match the control except the experiment runner. It adds an explicit opt-in
comparison and experiment-local Window subclass; production losses and Window
implementation remain unchanged.

Only the coefficient pair changes: **status 1 → 0.1**, **feasibility 0.5 → 0.05**.
Noul/Choice/Score/Reason losses, per-component VJPs, eligibility division, AdamW
moments/schedule, global clip 1, data order, seeds, backend and numeric rules are
unchanged. Both coefficients change together, so their individual effects are not
isolated. No adaptive weights, factor sweep, retries or additional fitting.

The same **32 training records from 32 distinct families**, disjoint from E/X,
complete **eight repeats, 256 presentations, 32 updates at windows 189..220**.
Every record has eight exposures, with **zero failures**. Initial panel heads,
losses and input hashes match the control exactly. In the first training window,
all component gradient norms match the control after the declared scaling,
and D-gradient cosines are exactly unchanged. This is norm/cosine evidence;
full gradient tensors were not saved for cross-arm equality comparisons.

Eligible totals match for status **256**, Noul **64**, Choice **32**, Score **80**,
and feasibility **64**. Reason alignment/structure change **74 → 59** because
eligibility depends on predicted status; their losses are not matched-example
comparisons. All training source-view hashes match across arms. The candidate's
**26,656 session positions** obey exact P+B+A+H accounting, one prefill/lineage,
zero held positions, zero invocation text tokens and no gold returns. This count
excludes offline reference encoding and backward rematerialization.

## Training-panel learning

These are independent head measurements before feedback delivery. Assessed
accuracy stays **Noul 4/8, Choice 4/4, Score 0/10** at all five measurements in
both arms. Correct abstentions are excluded from those denominators.

| Repeat | Control Noul Brier | Candidate Noul Brier | Control Score interval distance | Candidate Score interval distance |
|---|---:|---:|---:|---:|
| 0 | 0.256381 | 0.256381 | 0.411528 | 0.411528 |
| 1 | 0.257546 | 0.257493 | 0.399025 | 0.387392 |
| 2 | 0.257668 | 0.257777 | 0.394757 | 0.356627 |
| 4 | 0.254736 | 0.254113 | 0.316655 | 0.210957 |
| 8 | 0.247584 | 0.238634 | 0.166759 | 0.248728 |

Final Noul BCE improves **0.688231 → 0.669854**, but all candidate logits remain
negative (−0.383048..−0.167868 across eleven panel records). On the eight assessed
records, mean probabilities for false/true are **0.411566 / 0.445210**, versus
control **0.441664 / 0.452203**. The mean gap grows **0.010539 → 0.033644**, but
both classes remain below the unchanged 0.5 decision boundary. These are descriptive
training statistics, not a fitted threshold, held-out ranking proof or new labels.

Score interval loss is better at repeat four (**0.119649 → 0.067452**) but worse
at repeat eight (**0.053827 → 0.096911**). No assessed Score readouts clip at any
measurement. Its final interval distance is worse than the control and remains
above zero on all ten assessed records. Early loss reduction does not demonstrate
stable fit. Final all-case training status counts, control → candidate:
Noul **11 → 11 /11**, Choice **8 → 8 /8**, Score **12 → 11 /13**.

## Held-out actual returns

All **130 E + 180 applicable X** cases run after training, without additional
updates or receiver changes. They use real decoded/delivered latent results,
the existing strict Noul threshold, unchanged FP32 Score intervals and exact
Choice identity binding. The control receipts are hash-verified cached evidence.
Locked labels and the twenty native-reader cases remain outside the experiment.

| Measure | Completed control | Loss-balance candidate |
|---|---:|---:|
| Assessed Noul delivered success | 23/80 | 22/80 |
| Assessed Choice delivered success | 47/50 | 47/50 |
| Assessed Score delivered success | 0/50 | 0/50 |
| All-case Noul success | 68/140 | 67/140 |
| All-case Choice success | 54/90 | 51/90 |
| All-case Score success | 5/80 | 5/80 |
| Overall behavior | 127/310 | 123/310 |
| Status correctness | 230/310 | 226/310 |
| Valid transport | 310/310 | 310/310 |
| X intervention pairs | 30/90 | 30/90 |

E successes are Noul **18 → 17 /40**, Choice **14 → 11 /50**, Score **5 → 5 /40**.
X successes remain Noul **50/100**, Choice **40/40**, Score **0/40**. There are no
failed evaluation cases. All non-assessed successes stay separate in the evidence.

Independent Noul raw truth correctness remains **25/80**, while assessed status
correctness falls **76 → 75 /80**. Held-out Noul Brier worsens
**0.273479 → 0.284861**, despite the improved training Brier. Choice assesses
49/50 eligible cases and delivers 47 correct selections in both arms; its loss
of three successes is on non-assessed E cases. Score assessed status stays
48/50 and mean interval distance worsens **0.261081 → 0.272387**. E Score distance
slightly improves **0.447060 → 0.439102**; X worsens **0.214586 → 0.230709**.

The existing X qualifier remains **BLOCKED** on correctness, Score agreement and
intervention-pair thresholds. This is a measured behavior failure, not an execution
blocker: the experiment completed and all artifacts passed integrity checks.

## Gradient evidence and interpretation

First-window weighted D norms change exactly as declared: status
**17.557519 → 1.755752**, feasibility **13.039668 → 1.303967**, while Noul
**0.217256** and Score **1.902198** are unchanged. The combined unclipped norm
falls **54.512424 → 26.837349**, and both arms still clip at 1.

Across the candidate's first repeat, D norms are status **1.264..2.017**,
feasibility **1.304..2.737**, Noul **0.179..0.965**, Score **1.366..2.109**.
In the final repeat, status is **0.194..2.221**, Noul **0.970..4.317**, and Score
**4.856..6.859**. All eight sampled combined norms exceed 1
(**18.885..45.690**). Eligibility and representations evolve after updates, so
later gradients need not be scaled copies. Adam's inherited moments, shared
parameters and global clipping also prevent a norm ratio from being a causal
diagnosis by itself.

This rejects this coefficient pair as an overall replacement at these fixed
exposures. It does not rule out other optimization schedules, longer training or
the architecture's learnability. Neither a simple initial norm imbalance nor a
lower intermediate loss establishes the cause of the remaining value failures.

**Next unfinished slice:** isolate readout learning from representation changes.
Define a head-only comparison using fixed D/input encoders and existing train
features, fitting only existing Noul prototypes and the Score row on the 3090,
with exact values/bindings and independent held-out head measurements. Compare
with joint learning to test whether these representations support the required
readouts. Do not change the probability threshold or widen Score intervals.
This is a proposed diagnostic, not authorization for another run or a claim that
it will fix the problem. Full training, learned invocation, native answer use and
Reason meaning remain unqualified.

## Validation and artifacts

**24 CPU tests pass in 28.838 s**, with model-runtime imports blocked. Independent
source/control/checkpoint/receipt/report audit passes. All six checkpoint file
sets validate; the first-update restore exactly reproduces receiver, optimizer
tensors and CPU/CUDA/Python RNG. All **320 W tensors/physical tying** pass before
and after; reference, R, budget/nonreceiver codec and grad=None remain unchanged.
No model arithmetic changed; prior numeric matrices are reused rather than rerun.

Model run: **1851.655636 s**, peak CUDA allocation **3,622,856,704 bytes**. Whole-run
time is not a throughput benchmark. Only RTX 3090 UUID
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1` executes model work. After exit, the
only GPU process remains service PID **2094**, **3852 MiB**. No agents, new data,
reviews, service/deployment changes or Operating Table modifications.

Preserved raw commands/exits, plan, sources, five panel files, training and E/X
receipts, audit and checkpoints: `data/receiver-loss-balance-20261107/` (ignored).
Final checkpoint `run/checkpoint-0220`, manifest
`ee4421e60f0b788014ea733dd147c5c478d8d524f2327142a6ad37c890c22977`;
receiver `84ed1802cae0ecf011a3580baa22c7155409f7860e511a3e16d56df24338bab7`.
Reference remains
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.
Status: **COMPLETED_PROBE_NOT_PROMOTED**.
