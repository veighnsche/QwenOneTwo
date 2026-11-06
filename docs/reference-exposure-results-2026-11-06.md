# Exposure-controlled reference results — 2026-11-06

**Exposure allocation contributed to the four tested training failures.** A
matched replay recovers all four previously failing Question/Target controls,
preserves every baseline recovery in the existing 40-case diagnostic set, and
repeats that result at both 900 and 1,000 updates. **Development recovery remains
0/100**, so reference qualification and generalization are still unresolved.

The [protocol](reference-exposure-comparison-v1.md) and implementation `ade8246`
preceded execution. The [portable receipt](evidence/reference-exposure-comparison-v1.json)
binds the raw results, source snapshot, audits and checkpoints under
`data/reference-exposure-20261106/`. Runner exit **0/COMPLETED** means the
comparison finished; it does not mean the reference qualified. No job remains
active, and no additional fitting is queued.

## Controlled change

Reuse the completed target-v2 broad control, including its exact initial weights,
empty optimizer state and RNG. Keep the same 1,000 updates, 8,000 presentation
slots, 3,376 distinct reviewed views, per-slot roles, ordinary reconstruction
loss, learning schedule, clipping, AdamW and eligibility rules. Only the existing
4,213,760-parameter reference pooler learns. Base W, reference T/codec and live
parameters stay fixed on RTX 3090 UUID
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1`.

Replace **336 presentations (4.2%)**: 170 Question slots and 166 Target slots.
Only repeat occurrences of other same-role views are replaced. Every original
view's first occurrence remains, all anchor occurrences remain, and every slot
retains its role. This reduces repeat counts of 329 other views without removing
any view. The two Question controls increase from one presentation each to 86;
the two Targets increase from two each to 85. The two Key controls retain 81
each. This is targeted reallocation, not balanced exposure for the full catalogue.

All **49 original computation sources** match the control byte-for-byte. The
first changed presentation is in window 36; losses and gradient norms match the
saved control **exactly for all preceding 35 windows**. All 1,000 actual active
optimizer-owner lists match the corresponding control windows. The completed
control and its baseline evaluations were reused, not rerun.

## Recovery and retention

| Fixed cases | Original final control | Treatment 900 | Treatment 1,000 |
|---|---:|---:|---:|
| Two targeted Questions | 0/2 | 2/2 | 2/2 |
| Two targeted Targets | 0/2 | 2/2 | 2/2 |
| Two unchanged Key controls | 2/2 | 2/2 | 2/2 |
| All six controls | **2/6** | **6/6** | **6/6** |
| Original first diagnostic set | 11/20 | 15/20 | 15/20 |
| Original separate screening set | 10/20 | 10/20 | 10/20 |
| Both diagnostic sets | **21/40** | **25/40** | **25/40** |

The only gains at both observations are `train-reference-0000`, `0010`, `0001`
and `0011`. There are **zero losses**: all 21 original recoveries remain exact.
The other 36 cases remain at 21/36. These are existing training diagnostics, not
unseen test examples. The reused diagnostic baseline is the original **final
1,000-update control**, including when shown beside treatment window 900.

At window 900, the two Questions had received 61/60 presentations, the Targets
65/64, and Keys 73/73. Final counts are 86/86, 85/85 and 81/81. The experiment
does not establish a minimum sufficient exposure count.

| Original development qualification | Control 900 / 1,000 | Treatment 900 / 1,000 |
|---|---:|---:|
| Exact canonical recovery | 0/100 / 0/100 | **0/100 / 0/100** |
| Explicit contrast success | 100/100 / 100/100 | **100/100 / 100/100** |
| >=95% recovery and >=90% contrasts at both | Failed | **Failed** |

All 100 original recovery cases and 100 original contrast groups were evaluated
at each treatment checkpoint. Case identities, denominators, tolerances and
thresholds remain unchanged. No locked labels or earlier-checkpoint substitution
were used. Contrast success alone does not certify semantic reconstruction.

## Validation and accounting

- **Three metadata tests passed in 0.001 s**; all three new Python files compile.
  The real-data preflight confirms exact coverage, counts, role preservation,
  reproducible control schedule and bound baseline reports.
- All **1,000 updates / 8,000 presentations** were eligible, with finite gradients,
  no failed exposures, and only reference-pooler optimizer ownership.
- Initial serialized state matches all **1,396 tensors**, optimizer bytes and RNG
  bytes. Across all eleven checkpoints, **1,338 non-pooler tensors** remain
  byte-identical, including all **483 reference T/codec state tensors**. The
  update-time parameter version guard covers 482 frozen reference parameters;
  the serialized audit additionally covers the codec buffer.
- All **320 frozen base tensors and physical tying** pass the GPU comparison
  against the pinned source snapshot. Decoder/live fingerprints are unchanged,
  all gradients finish None, and both exports exactly match their checkpoints.
- Training features: **3,376 frozen-W prefills / 131,255 input positions**, cached
  on CUDA. Training accounting charges **246,970 positions**; the teacher-forced
  transformer inputs contain **238,970 positions**, excluding the final target
  token in each of 8,000 examples. This is forward-sequence accounting, not a
  count of backward recomputation. Example lengths differ after reallocation;
  equal updates/presentations do not imply equal token compute.
- Median update time: **2.550225070999659 s**. Measured run time after metadata
  preparation: **3,236.2155652950023 s**. Peak allocated CUDA memory:
  **4,193,888,256 bytes**. Time is an observation, never a stopping rule.
- Mean reconstruction CE, first/last 100 windows:
  **0.5455260183569044 / 0.40763906668158595**. Acceptance uses exact recovery,
  not this training-loss decrease.

Final reference fingerprint:
`d741b65f85ef4d381c74bf48b3bf2330de3009f8e561e20d9e3bdc389ffdeb42`.
Window-900 fingerprint:
`fa906e8f77bec91d6c43af8afc4a8c10e31f39281ab008b18719d526f4aa9969`.
Both remain unqualified. No new source records, annotations/reviews, agents,
external model calls, capability updates, live promotion, deployment or service
changes occurred. Existing numeric matrices were not rerun because their
computation sources are unchanged.

## What this establishes and what remains

The matched intervention supports a causal contribution from exposure allocation
for these four training cases: more repetitions in the broad mixture repaired
their recovery while retaining every tested baseline success. W and the decoder
already support recovering those meanings under this architecture. The earlier
six-case success versus broad-fit failure was therefore not evidence that these
four meanings were intrinsically unrecoverable.

This single deterministic comparison does not identify a unique global cause.
Reallocation increases anchor rehearsal and reduces some competing repetitions;
it does not separate those two effects. It does not equalize the thousands of
remaining low-exposure views, prove retention outside the fixed 40 cases, or
establish semantic competence. The old broad control restarted from the common
initial state, so this is not proof of catastrophic forgetting of the six-case
probe.

The next unfinished slice is **reference recovery/generalization under broad
training coverage**, with retention and the original paired development gate
still required. A broader exposure schedule is now an evidence-backed next
experiment to discuss. No further fitting or capability training is launched or
implicitly queued by this result.
