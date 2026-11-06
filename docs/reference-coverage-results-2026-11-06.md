# Complete-coverage reference results — 2026-11-06

**Stopped after pass 12 as explicitly requested. Pass 13 never started.**
Coverage improves the fixed training panel from **36/191 to 71/191** and reduces whole-pass reconstruction loss, while original development exact recovery stays **0/100**. This is useful learning evidence, not a solved generalization problem or a product-readiness claim.

The [protocol](reference-complete-coverage-v1.md) and implementation `caa1db5` preceded execution. The [portable receipt](evidence/reference-complete-coverage-v1.json) binds raw artifacts and the passing audit under `data/reference-coverage-20261106/`. No experiment job remains active.

## Scope and stopping

Continue the saved exposure-treatment checkpoint at update 1,000. Keep its weights, optimizer moments/steps and RNG; fit only the 4,213,760-parameter target-v2 reference pooler on the specified RTX 3090. W, reference T/codec and live parameters remain frozen. Use the existing 3,376 reviewed unique training views, exactly once per seeded shuffled pass: **422 updates / 3,376 presentations per pass**.

The continuation restores peak pooler LR **2e-4**, with unchanged ordinary teacher-forced CE, eligibility normalization, AdamW, clipping and v3/graph-v1 computation. Sampling and the schedule both change, so this is not a matched sampler-only causal comparison. Default A4/capability schedules retain their original bounds.

The initial loop had **no fixed pass count**. It automatically continued until a full-training recovery target or three passes without a new best panel count or whole-pass mean loss. Loss kept improving. During pass 12 the user explicitly instructed that it be the last pass. Implementation `c36d6ac` adds `--stop-after-pass 12`, persists it in checkpoints and checks the limit before any new episode. This user stop overrides the original automatic criterion; **no plateau or convergence is claimed**.

## Measured curve

The fixed, outcome-independent panel contains up to 32 unique views per role, all smaller roles, and all original 40 diagnostic annotations: **191 views total**. It is role-stratified training evidence, not an estimate or exhaustive measurement of all 3,376 training views.

| Completed pass | Panel exact /191 | Prior diagnostics exact /40 | Whole-pass mean CE |
|---:|---:|---:|---:|
| 0 | 36 | 25 | — |
| 1 | 32 | 20 | 0.683013 |
| 2 | 35 | 20 | 0.657778 |
| 3 | 39 | 18 | 0.627453 |
| 4 | 47 | 22 | 0.596811 |
| 5 | 55 | 24 | 0.567779 |
| 6 | 60 | 24 | 0.541821 |
| 7 | 61 | 23 | 0.515958 |
| 8 | 60 | 23 | 0.492751 |
| 9 | 67 | 25 | 0.467058 |
| 10 | 69 | 24 | 0.443685 |
| 11 | 73 | 27 | 0.422220 |
| 12 | 71 | 26 | 0.398414 |

Pass 11 has the highest panel observation, **73/191**. The final pass-12 weights are retained; no earlier favorable checkpoint substitutes for the requested endpoint. Mean CE declines from **0.6830126312468918** in pass 1 to **0.3984135712201553** in pass 12.

| Role | Starting exact | Final exact |
|---|---:|---:|
| Alternative | 0/32 | 25/32 |
| Conditions | 1/32 | 6/32 |
| Dimension | 7/7 | 7/7 |
| Key | 5/5 | 5/5 |
| Midpoint | 5/5 | 5/5 |
| One | 7/7 | 7/7 |
| Question | 2/32 | 4/32 |
| Reason | 0/32 | 2/32 |
| Target | 2/32 | 3/32 |
| Zero | 7/7 | 7/7 |

Final original diagnostics are **26/40**, versus 25/40 initially: three gains (`train-reference-0019`, `0041`, `0039`) and two losses (`0010`, `0001`). This is not complete retention. The six prior controls go from **6/6 to 4/6**. Unlike the earlier broad control that reset to common initial weights, this experiment actually continues the successful exposure-treatment weights; these are measured continuation regressions.

Across **all 24 development observations**, the unchanged original suite measures **0/100 exact canonical recovery and 100/100 contrast success**. Final observations at cumulative updates **5,964 / 6,064** are both failures under the original >=95% / >=90% paired certificate. No locked labels or earlier checkpoint selection were used.

Exact wording is a conservative diagnostic, not the complete product promise. Nonexact outputs remain semantically unresolved; 0/100 exact recovery alone does not prove 100 semantic failures or MVP failure. The [parent-weight meaning inspection](reference-meaning-inspection-2026-11-06.md) found clear material errors in 14 selected development readbacks, with six anchors unresolved. That inspection is not a final-weight semantic score. Connected, held-out use of meaning still needs behavioral evidence.

## Continuity, validation and accounting

- **12 complete passes / 5,064 new updates / 40,512 eligible presentations**: exactly 12 new presentations per unique view. The last cumulative optimizer update is **6,064**. Every role-local optimizer-owner check passes; failed exposures are zero.
- Two checkpointed interruptions preserve exact weights, optimizer, RNG, sampler cursor and in-pass loss totals. The first supports the meaning inspection; the second installs the explicit final-pass limit. Logs concatenate without repeated completed updates: prefix 1–100, continuation 101–4,800, final segment 4,801–5,064.
- The audit verifies **all 75 checkpoints**, exact initial **1,396 tensors** plus optimizer/RNG bytes, and **1,338 unchanged non-pooler tensors**, including all **483 reference T/codec state tensors**. Both final exports match their corresponding checkpoints.
- All **320 frozen base tensors and physical tying** pass the GPU comparison. Decoder and live fingerprints remain unchanged; all gradients finish None.
- **Three metadata tests passed in 0.005 s; eleven CUDA optimizer/checkpoint tests passed in 0.587 s**, including continuation beyond 1,000 and exact next-update resume. The user stop is verified by the actual last checkpoint/ledger and persisted limit. Completed numeric matrices were not rerun; model arithmetic is unchanged.
- Completed-window accounting: **1,401,924 charged positions / 1,361,412 teacher-forced forward input positions**. The difference is one final target token per presentation. This does not count backward recomputation.
- Recorded panel recovery adds **44,003 executed decoder input positions**, reusing the original 40 baseline recoveries. Development evaluations and the separate meaning inspection remain separate accounting categories.
- Frozen feature preparation was performed three times across process resumes: **10,128 W prefills / 393,765 input positions**, with an additional 20 annotation encodings for the meaning inspection. All model work was on RTX 3090 UUID `GPU-225509c3-d702-6c27-de03-76b5e99b33b1`.
- There may be unlogged in-flight work after checkpoint publication at the two process interruptions. Completed updates/presentations are exactly accounted; **no exhaustive physical-compute total is claimed**.
- Median completed optimizer-window time: **2.452649227998336 s**. Final resumed process time: **1,515.7522246669978 s**; its peak CUDA allocation: **4,194,480,128 bytes**. These are observations, not budget limits or a controlled performance comparison.

Final frozen reference fingerprint:
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.

The final process saved its checkpoint, evaluations, successful frozen checks and `status=COMPLETED` report, then exited **1** because its final console emitter supplied `elapsed_seconds` twice. The one-line logging fix was verified using the actual saved report; **no model work was rerun**. The original log/exit and executed source snapshot are retained. The two earlier intentional checkpoint interruptions exited 143. Do not rewrite those exits as success codes.

## What remains

Complete coverage produces broader training recovery, especially for Alternatives, but Question, Target and Reason recovery remains weak. Development canonical transfer has not improved. The experiment stopped at the user-selected endpoint while loss was still improving, so it cannot establish convergence, inability to learn with more exposure, or a unique encoder-versus-decoder cause. Full-catalogue recovery was not measured at this user stop.

The next unresolved work is reference meaning/recovery transfer and the weak roles, with actual semantic use distinguished from exact wording. No pass 13, additional fitting, new data generation, capability training, live promotion or deployment is queued. W, codec/live ownership, latent communication, budget arithmetic and runtime cache contracts remain unchanged; The Operating Table and live services were not modified.
