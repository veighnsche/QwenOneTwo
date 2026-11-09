# Experiment summary — through 9 November 2026

**We have demonstrated working mechanisms and some narrow learning, but not the
complete useful, autonomous latent-decision system.** The latest matched RD-08
candidate failed learning acceptance. CPU bottleneck removal remains unresolved.
RD-09 has not started. The independent architecture proposal is planning only.

This summarizes the recorded experiments, audits and engineering qualifications.
Related variants are grouped; linked reports preserve individual cases, rejected
attempts, source identities and raw evidence. No experiment was rerun to write
this document. Historical results describe their own checkpoints and interfaces;
they must not all be treated as evaluations of the current model.

## How to read the results

- **Engineering pass:** machinery, arithmetic, gradients, accounting or recovery
  behaved correctly in the measured scope. This does not establish task meaning.
- **Training gain:** fitted examples improved. This does not establish transfer.
- **Learning acceptance:** held-out behavior met the experiment's declared rules.
  Completing an experiment does not imply passing those rules.
- **E:** 130 receiver development cases. **X:** 180 receiver intervention cases,
  arranged as 90 pairs. The separate 20 native-reader X cases are excluded from
  these receiver counts. Some readout diagnostics use smaller assessed subsets.
- **Assessed value success** differs from a correct non-assessed status. Numeric
  Score was generally graded against unchanged FP32 point/interval annotations;
  a very small error can therefore still fail an exact point target.

## 1. Early prototype and engineering feasibility

| Experiment | What ran and what happened | Conclusion / evidence |
|---|---|---|
| Initial routing/latent prototype | 246 synthetic examples; recorded stages: 240 router, 800 latent and 600 head steps. Routing training accuracy reached 100%; router loss 0.829→0.0186, latent loss 4.106→0.800. | Narrow training success, not a held-out semantic benchmark. [Prototype description](../README.md); local `data/trained/one-two-v1/training.json`. |
| Routing refinement | 126 examples, 4,072 hidden-state samples, 1,000 steps; recorded training accuracy 100% for all four actions in thinking/answer phases. | Training-set routing proof. Local `data/trained/one-two-v1-routes-2/training.json`. |
| Feedback completion variants | Saved artifacts report: feedback-3, 240 steps/loss 18.686→10.491; feedback-4, 180 steps/0.00156→0.423; feedback-5, 160 Choice-focused steps/8.174→2.551; feedback-6, 120 Choice-index-focused steps/12.814→0.00166. | Mixed loss outcomes; no broad evaluation established. These are artifact-reported stage counts, not sums of inherited metadata. Local `data/trained/one-two-v1-feedback-{3,4,5,6}/training.json`. |
| Prototype GPU/browser demonstrations | Translation, Noul→“Yes”, Choice→“B: blue”, Score→0.7; one observed prefill and same cache per run. Budget-zero answers, bounded thinking, cancellation and browser checks also recorded. Answer-phase operation probes used declared teacher-forced prefixes. | Bounded demonstrations, not general invocation/answer-quality evidence. Local `artifacts/verification-summary.json`, `one-two-gpu.json`, `one-two-answer-phase.json`, `continuous*-gpu.json`, browser receipts; [scope](../README.md#verification-and-deployment). |
| Initial G1-core | Cache/accounting and connected gradient fixtures passed; overall numerical qualification failed. | Original BF16 path was inadequate under the required comparisons. [Results](g1-core-results-2026-11-05.md). |
| G1 compensated-precision revision | 72 numerical cases passed; released Noul continuations still selected unsupported Choice. | Numeric repair succeeded; overall G1 still failed the continuation gate. [Results](g1-precision-results-2026-11-05.md). |
| G1 gate repair | 19 saved features, 256 gate-only fitting steps. Then 72/72 numeric cases, 144/144 gate checks and 36/36 released continuations passed. | Bounded engineering pass; labels came from two diagnostic trajectories, not broad semantic training. [Results](g1-gate-results-2026-11-05.md). |
| Expanded original BF16 codec | Component tests and shorter probes passed; 512-position comparison failed: state RMS 0.234, gradient errors about 0.71. | Rejected arithmetic profile; exact cache continuity did not make numerics acceptable. [Results](codec-qualification-results-2026-11-05.md). |
| Feedback precision v3 | BF16 weight storage with FP32 transformer/adapter computation; 324/324 traces at 128/256/512 positions passed. Worst state RMS 0. | Numerical/mechanical qualification, not learned semantics. [Results](codec-feedback-results-2026-11-05.md). |
| 1,024-position extension | 108/108 traces passed; peak allocation 12.194 GiB; about 93.31 minutes. | Qualified these prescribed traces, not the full 262,144-position context or complete trainer. [Results](codec-1024-results-2026-11-05.md). |
| Backend, release and episode checks | Backend final GPU suite: 44 passes. Release/driver/sampler byte and resume checks passed. Episode integration: 57 model-free and 11 distinct GPU tests passed, including latent-only transport with tokenization/embedding lookup disabled during the tested boundary. | Implementation evidence; forced fixtures do not establish autonomous formation or native understanding. [Backend](backend-slices-2026-11-05.md), [release boundaries](backend-release-slices-2026-11-05.md), [episodes](episode-producer-results-2026-11-05.md). |

Early native routing/feedback fitting did occur. That is distinct from the later,
unstarted RD-09 demonstration of the redesigned complete learned flow.

## 2. Data preparation and coverage

| Work | Recorded result | Meaning / evidence |
|---|---|---|
| First Muse draft pilot | 12 proposals, two per domain; 7/13 operation/status cells; drafts had no approved semantic release. | Draft production, not model learning. [Report](muse-draft-pilot-2026-11-05.md). |
| Replacement economical batch | 12 new proposals in one call, 29.969 seconds; 12/12 passed draft validation and 12/12 correctly rejected at the approved-release gate. | Reduced drafting orchestration; proposals remained unapproved drafts. [Report](muse-economy-12.md). |
| Model-reviewed pilot | Existing 24 drafts plus 96 new cases formed 120 development cases, covering all 13 operation/status cells after documented repairs/reviews. | Model-assisted review, not human certification or learned capability. [Report](pilot-120-results-2026-11-05.md). |
| Full collections | 8,000 selected training, 1,000 development and 2,000 locked records. Training includes 1,500 receiver records. | Counts include correlated derivations/replays; they are not that many independent meanings. Later experiments left locked labels unopened. [Report](training-collections-results-2026-11-05.md). |

## 3. Reference representation/reconstruction experiments

The original recovery certificate required ≥95/100 exact canonical recoveries
and ≥90% contrast success at both final checkpoints. Exact wording is a
conservative diagnostic: failing it alone does not prove semantic failure.
The contrast suite already passed before learning and was not sufficient proof.

| Experiment | Recorded result | Conclusion / evidence |
|---|---|---|
| Untrained reference baseline | 0/100 exact recoveries; 100/100 contrast groups. Artifact audit found no compatible trained reference. | Not qualified. [Binding/baseline](reference-binding-results-2026-11-06.md). |
| First bounded reference fit | 1,000 updates/8,000 presentations over 500 IDs containing only 50 distinct meanings. Loss 2.659→0.047; final checkpoints both 0/100 recovery, 100/100 contrasts. | Strong training-loss reduction without held-out exact recovery. [Results](reference-fit-results-2026-11-06.md). |
| Expanded catalogue/history-ablation revision | 3,376 distinct views, 1,000 updates/8,000 presentations; alternated ordinary teacher forcing and preceding-target ablation. Loss 4.771→1.136. Both final gates stayed 0/100 recovery. | Broader exposure/conditioning change did not qualify. [Results](reference-recovery-results-2026-11-06.md). |
| Encoder ablations | Seven focused 64-window variants; detailed outcomes below. Target-private pooling alone recovered all six controls. | Local interference could be repaired, but this was not a generalization result. [Results](reference-encoder-results-2026-11-06.md). |
| Target-private full fit | Fresh 1,000-update/8,000-presentation broad-data run; final training controls only 2/6, development recovery 0/100 at both required checkpoints. | Six-case success did not survive the full mixture. [Results](reference-encoder-results-2026-11-06.md#completed-full-reference-block-failed-qualification). |
| Rank-one/common-direction weight repair | One fixed edit rule applied to the two final reference checkpoints; no gradient fit. Both remained 0/100 recovery and 100/100 contrasts. | Rejected; no recovery repair demonstrated. [Results](reference-weight-repair-results-2026-11-06.md). |
| Pooler block-replacement search | Initial run was incomplete at an imposed deadline. Continuation removed that cutoff and completed all 32 masks plus screening. Best first-set count 12/20 vs baseline 11/20; separate screening 7/20 vs baseline 10/20. | No candidate met improvement/retention rules. [Initial interruption](reference-block-search-results-2026-11-06.md), [completed search/all masks](reference-block-search-completion-results-2026-11-06.md). |
| Matched exposure reallocation | Same 1,000 updates/8,000 slots; changed 336 repeated presentations. Six controls improved 2/6→6/6; two diagnostic sets 21/40→25/40 with no losses. Development remained 0/100 recovery. | Exposure mattered for selected training failures; generalization remained unresolved. [Results](reference-exposure-results-2026-11-06.md). |
| Meaning inspection | Read-only full readbacks of 20 selected development annotations from the exposure-treatment parent. Fourteen had material substitutions/omissions; six anchor meanings remained unresolved. | Failure was not solely harmless wording variation. This was direct diagnostic inspection, not an independent semantic population score. [Report](reference-meaning-inspection-2026-11-06.md). |
| Complete-coverage continuation | 12 passes over all 3,376 views: 5,064 new updates/40,512 presentations. Training panel 36/191→71/191; best intermediate count 73/191. Loss 0.683→0.398. Development exact recovery stayed 0/100 across all 24 observations. | Training improvement with regressions and no held-out exact recovery. User stopped after pass 12; no convergence claim. [Results](reference-coverage-results-2026-11-06.md). |
| Behavioral replacement for reconstruction gate | Actual X results: Noul 13/100, Choice 31/40, Score 0/40; total 44/180, complete pairs 16/90. Sixteen invalid Reason decodes remained failures. | Actual receiver semantics also failed; exact reconstruction became diagnostic rather than the product criterion. [Results](reference-behavior-results-2026-11-06.md). |

Focused encoder variants, all using the same six training controls:

| Variant | Exact training recovery |
|---|---:|
| Shared pooler, ordinary cross-entropy | 5/6 |
| Shared pooler plus dense latent-target MSE | 4/6 |
| Field-value-only memory | 5/6 |
| Shared pooler, five-times learning rate | 5/6 |
| Shared pooler, focal token loss | 5/6 |
| Target-only fitting | 2/2 Targets, 3/6 overall |
| Target-private pooler, mixed-role loss | 6/6 |

The MSE variant also used 96 disposable latent-variable optimization steps to
construct six targets; those were separate from model-parameter updates.

## 4. Receiver learning and causal diagnostics

| Experiment | What changed / exposure | Actual result and interpretation |
|---|---|---|
| First full receiver pass | 188 updates/1,504 presentations: all 1,500 records plus four declared repeats. | X Choice 31/40→40/40; X Noul 13/100→9/100; numeric Score 0/50 assessed E+X. Assessed Noul delivery 11/80→0/80. Some Choice transfer, no qualified Noul/Score. [Results](receiver-learning-results-2026-11-07.md). |
| 32-record repeated probe | Eight repeats/256 presentations/32 updates from the saved 188 parent. | E+X behavior 83/310→127/310; assessed Noul delivery 0/80→23/80; Choice 47/50 retained; assessed Score stayed 0/50. Training Noul truth stayed 4/8 and training Score had no interval hits. [Results](receiver-training-probe-results-2026-11-07.md). |
| Loss-balance comparison | Same probe initialization/order/exposure; status and feasibility coefficients divided by ten. | Overall 127/310→123/310 vs control. Noul training Brier improved, truth accuracy stayed 4/8; final Score distance worsened 0.166759→0.248728. Candidate rejected. [Results](receiver-loss-balance-results-2026-11-07.md). |
| Frozen-feature head-only fit | Only 3,072 Noul/Score parameters; 1,812 cached head updates/14,496 visits. | Score training MSE fell to 1.85e-6, but only 3/10 strict hits, all clamped zeros. Development Score distance worsened 0.271739→0.319199; assessed Noul truth 25/80→24/80. Delivered total stayed 83/310; Choice unchanged. [Results](receiver-head-only-results-2026-11-07.md). |
| Architecture/data audit | Metadata and ownership review; no model fitting. | Identified narrow Score direction/range coverage, Noul class shift and changing D-dependent source features. [Baseline](architecture-baseline-2026-11-07.md). |
| Information-loss inspection | Frozen intermediate captures on all 180 X cases; reused training features. | All 70 changed-input pairs had different final hidden states; no identical-slot collapse in 150 changed role comparisons. Saved Noul head training AUC 1.0 still gave 4/8 at fixed threshold; X AUC fell to 0.373. Distinguishability did not establish useful meaning/generalization. [Results](information-loss-results-2026-11-07.md). |
| Ordered full-width means | Replaced learned pooling with ordered means, receiver frozen; 180 X cases. | Behavior 49/180→31/180; Choice 40/40→11/40; complete pairs 23/90→0/90; Score stayed 0/40. Rejected; this still compressed inputs. [Results](simple-latent-results-2026-11-07.md). |
| Residual Noul/Score repair | 133,184 new parameters on all 1,003 Noul/Score rows; 2,000 windows/16,000 visits. | Overall 83/310→123/310, mainly status gains. Assessed truth stayed 25/80, Score delivery 0/50; Score distance worsened. All 90 Choice receipts unchanged. Value-learning acceptance failed. [Results](receiver-residual-repair-results-2026-11-07.md). |
| Matched value-only residual | Same architecture/features/16,000 visits; removed status loss. Ten empty-target windows yielded 1,990 actual updates. | Overall 78/310 vs joint control 123/310; Noul truth 25/80, Score delivery 0/50; Choice unchanged. Removing status competition did not fix values. [Results](receiver-value-only-results-2026-11-07.md). |
| Direct affine readouts | GPU ridge probes over saved 1,024-wide features: 24 grouped-training CV solves and two final fits. | X Noul 48/70, below the 50/70 majority reference; balanced accuracy 55.5%. Score 0/40, distance 0.336694. Diagnostic head predictions, not joint delivered behavior. [Results](direct-latent-readout-results-2026-11-07.md). |
| Training-to-X semantic audit | Metadata plus reused GPU captures; no new fitting/model execution. | Training Noul only positive AND, X adds negation/OR. Score training uses increasing completion and denominators 20/100; X uses 200 plus new operators. Familiar tasks also failed. Current reference had actually seen 3,376 views, not just the original 50 meanings. No single failed layer isolated. [Audit](training-x-semantic-audit-2026-11-07.md). |
| Direction hash and numeric-access test | Fixed omitted rubric direction in input identity. Compared source states, added literal counts, and the same counts as INT64 on 40 X Score cases; no training. | Score hits 0/40 in every path. Distances 0.233890 original, 0.234632 literal, 0.234862 INT64. Explicit numeric access alone did not help the frozen receiver. [Results](numeric-access-results-2026-11-07.md). |
| Quantity/operator learning comparison | Two matched arms, literal vs INT64 counts; 254 assessed Score records×4, 1,016 presentations/127 updates each. | Training distance fell to about 0.040 in both arms, but no exact interior hits. X Score stayed 0/40; distance worsened to 0.337477/0.338374, correct Assessed status fell to 7/40 and 12/40. Neither recovered both source counts on any X case. No clear INT64 advantage; neither arm promoted. [Results](quantity-receiver-learning-results-2026-11-08.md). |

## 5. Redesign queue and final matched comparison

| Slice / experiment | Result | Evidence |
|---|---|---|
| RD-01: coverage/supervision | Audited selected training/development records and fixed comparison panels. Data limitations were recorded, not repaired by inventing labels. | [Report](redesign-rd01-results-2026-11-08.md). |
| RD-02: prepared runs | Explicit source-bound preparation, identities and resume contracts implemented; model-free validation passed, no fitting. | [Report](redesign-rd02-results-2026-11-08.md). |
| RD-03: frozen-W sources | Stable offline source policy and caching rules implemented; tensor checks deferred to RD-07. | [Report](redesign-rd03-results-2026-11-08.md). |
| RD-04: direct role states v1 | Ordered full role states implemented, but original bounds admitted 0/1,500 training records. | [Report](redesign-rd04-results-2026-11-08.md). |
| RD-05: joint objectives | Binding, owners and masks implemented. Unsupported new-profile Reason supervision remained masked. | [Report](redesign-rd05-results-2026-11-08.md). |
| RD-06: native eligibility | Wiring/static checks passed; audit found zero complete compatible direct-profile formation, native continuation or supplied-reader supervision. Existing native-category record counts did not establish eligibility. | [Report](redesign-rd06-results-2026-11-08.md). |
| RD-07: GPU qualification | Ten deferred plus three new checks passed, with zero optimizer updates. Technical acceptance did not remove the data-layout blocker. | [Report](redesign-rd07-results-2026-11-08.md). |
| RD-08 initial attempt | Blocked before fitting: direct v1 common training support was zero. No smaller subset substituted. | [Historical report](redesign-rd08-results-2026-11-08.md). |
| Compact format v2 | Complete roles plus compact bound returns admitted all 1,500 training records in both arms, without cropping/new data. E direct support 127/130, X 180/180; three overlength E cases retained as failures. Final affected-path qualification: 19 checks passed, zero updates. | [Coverage](evidence/compact-role-layout-coverage-v2.json), [qualification](evidence/compact-role-qualification-v2.json). |
| RD-08 completed matched fit | Frozen-W pooled control vs direct-state candidate, same shared initialization/moments/RNG/objectives/order. Each completed 750 updates/6,000 presentations, plus full initial/final observations and then E/X evaluations. | **Learning acceptance failed.** Detailed results below and [final report](receiver-matched-v2-results-2026-11-08.md). |
| RD-09 | No run. Initially blocked by RD-08 coverage; now blocked by failed RD-08 acceptance/no accepted receiver, plus native-supervision limits. Latest user instruction also prohibits starting it. | [Original blocker](redesign-rd09-results-2026-11-08.md), [current ledger](redesign-progress.md). |

Final RD-08 actual delivered behavior:

| Arm / panel | Noul | Choice | Score | Total |
|---|---:|---:|---:|---:|
| Pooled control / E | 16/40 | 22/50 | 13/40 | 51/130 |
| Direct candidate / E | 13/40 | 16/50 | 14/40 | 43/130 |
| Pooled control / X | 50/100 | 40/40 | 0/40 | 90/180 |
| Direct candidate / X | 20/100 | 0/40 | 0/40 | 20/180 |

- Across all 310 cases: **28 candidate wins, 106 losses, 35 both pass, 141 both
  fail**. Totals: control 141/310, candidate 63/310.
- Complete X intervention pairs: **30/90 control, 0/90 candidate**.
- Candidate lost **all 47 historically passing Choice cases** in the fixed
  50-case retention panel. Current control retained 46 of those 47.
- Familiar Noul: E 4/4→3/4, X 50/50→10/50; candidate only matched the
  training-defined false-constant baseline. Familiar assessed Score cohort was
  absent, so the required improvement could not be established.
- Across assessed Score cases, mean distance worsened E 0.312811→0.492258,
  X 0.238843→0.517047. Correct interior delivered Score stayed zero in both arms.
- Training did occur: control Noul truth head 168/256→256/256 and Score distance
  0.404325→0.046883; candidate 168/256→179/256 and 0.408468→0.266343. These
  training gains did not establish the required held-out improvement.
- All training exposures completed; this does not mean all predicted returns
  decoded correctly. Return failures remain in evaluation denominators. Both
  evaluation processes made zero optimizer updates, and frozen-owner checks passed.

The interrupted control had 599 logged updates but only 500 saved. Recovery
replayed 99 lost windows/792 presentations with exact original receipts, then
finished the fixed schedule. Recorded physical completed control updates were
849 versus 750 retained; any unflushed work before process loss is unknown.
The compatible checkpoint preserved weights, Adam state, RNG and sampler state.
No extra effective learning pass was introduced.

## 6. CPU/GPU performance experiments

These measurements use different workloads and baselines. Their speedups must
not be multiplied together or presented as a single end-to-end training speedup.

| Experiment | Measured result | Decision / scope |
|---|---|---|
| Causal parallel teacher forcing | Two GPU comparisons passed, preserving serial loss/gradient agreement and future-target causality. | Removed per-target-token transformer calls for reference fitting; [report](reference-fit-results-2026-11-06.md#slice-1-causal-teacher-forcing). |
| Fused Triton recurrence | Local tests passed; full 512-position feedback comparison exceeded state/gradient tolerances. | Rejected, no relaxed tolerances. [Report](recurrence-graph-results-2026-11-06.md#rejected-fused-kernel-attempt). |
| CUDA-graph recurrence replay | Warm single-episode forward/VJP 4.17× faster; real two-window accumulation benchmark 3.21× faster, with no optimizer step. 45 full-model traces passed. | Accepted within measured scope; not a full trainer speedup. [Report](recurrence-graph-results-2026-11-06.md). |
| Read-only training performance audit | About 1.25 CPU cores busy while host 88.6% idle; sampled GPU average 19.85%. Runtime strongly correlated with separate component VJPs (r≈0.979). No competing-trainer explanation observed. | Supported investigating serialized host/autograd dispatch; correlation was not a full causal profile. [Audit](receiver-performance-audit-2026-11-07.md). |
| Aggregated finite checks | Predicate microbenchmarks 7.45×/8.58× faster; real accumulation windows only 1.84%/1.76% faster. | Correctness checks passed; modest workload gain. [Results](fused-finite-results-2026-11-07.md). |
| Immutable manifest access | Eight-view CPU preparation 0.909318→0.015810 seconds, saving 0.893508 seconds. | 57.51× for that metadata path, not measured full training. [Results](manifest-access-results-2026-11-07.md). |
| Asset lookup index | Saved about 8.3 ms per eight-view CPU batch; preparation reduced 47–55%. Audit serialization left unchanged. | Small CPU preparation optimization, not seconds of model training. [Results](asset-audit-results-2026-11-07.md). |
| Two-GPU runner and CPU-01/02 instrumentation | Model-free preparation/transport/profiling tests passed while GPUs were held off. | Build evidence only at this point. [Runner](two-gpu-diagnostics-v1.md), [instrumentation](cpu-01-02-results-2026-11-07.md). |
| CPU-03 actual hardware baselines | Eight recurrence tests passed on each card. Similar small-pack card timings; two-card end-to-end throughput about 1.112× 3090-only. Trace exposed many host launches/synchronizations. | Read-only 3090/3060 diagnostics, not distributed training or general 3060 qualification. [Results](cpu-03-results-2026-11-07.md). |
| CPU-04 recurrence cache, 8→32 entries | Three paired comparisons per GPU: wall −5.295% on 3090, −4.041% on 3060; CPU roughly −5.06%/−3.92%. Warm recaptures 37→0. | Useful bounded retention option. [Results](cpu-04-results-2026-11-07.md). |
| Full RMS graph replay | Exact local checks; isolated speedup 1.160–1.179×. | Not selected; no certified end-to-end gain. [Kernel screens](throughput-kernel-screens-2026-11-07.md). |
| Pointwise RMS-tail fusion | Standard Triton launch slower; compiled launch only about 1.09–1.12× isolated gain. | Not selected. [Kernel screens](throughput-kernel-screens-2026-11-07.md). |
| Native fused RMS | Isolated 1.535–1.584× gain; first full-feedback numerical comparison failed strict state/gradient limits. | Rejected; remaining planned full-model pairs not run after rejection. [Kernel screens](throughput-kernel-screens-2026-11-07.md). |
| Single-CTA finite predicate | Four local GPU tests passed; isolated speedup only 1.026–1.044×. | Not selected. [Kernel screens](throughput-kernel-screens-2026-11-07.md). |
| Resident worker pool | Two processes/card vs one: main throughput +38.12%, end-to-end +5.32%; worker CPU/case +32.91%, setup-inclusive CPU/case +81.30%, roughly doubled model allocation. | Opt-in throughput trade-off, not CPU-work reduction. [Results](resident-pool-results-2026-11-07.md). |
| Shared context/frozen weights | Allocation −46.77%, setup-inclusive CPU −21.65%; main CPU/case +8.23%, throughput −15.45%. | Correctness passed, requested CPU/case improvement failed. [Results](shared-context-results-2026-11-07.md). |
| Larger pure recurrent-attention replay | Complete-case CPU −3.5924%, resident throughput +3.5649%; 36 model traces passed. | Modest opt-in gain; most host work remained. [Results](linear-attention-block-results-2026-11-07.md). |
| Script projection dispatch | CPU/case +4.4944%, throughput −4.3178%; checkpoint JIT guard also failed. | Rejected and runtime/test prototype changes removed. [Results](projection-dispatch-results-2026-11-07.md). |
| Retained-activation receiver execution | Six alternating paired windows: pooled 48.322→35.763 s (−25.99%); direct 52.837→39.992 s (−24.31%). CPU time −37.37%/−34.77%; exact gradients/updates/resume verified. | Qualified opt-in policy used for RD-08 recovery. Twelve discarded diagnostic updates were separate from RD-08. [Results](receiver-performance-v1.md). |

**Current performance conclusion:** improvements are measured, but removal of the
CPU bottleneck is not demonstrated. An instantaneous GPU reading of 100% does
not establish sustained GPU utilization. No CPU cap was requested or applied.

## 7. What has not been demonstrated

| Question | Current evidence |
|---|---|
| Does the complete redesigned autonomous native flow work? | Not established; RD-09 unstarted and compatible native-supervision gaps remain. |
| Is there an accepted improved RD-08 receiver? | No. Final direct-state candidate failed the declared acceptance checks. |
| Does a larger foundation fix it? | No controlled model-size comparison is recorded. All main model experiments used the 0.8B foundation; capacity remains a hypothesis. |
| Is the full 262,144-position system qualified? | No. Shorter prescribed numerical traces do not establish full-context capacity or semantic performance. |
| Is the new independent architecture validated? | No. A fresh, requirements-only agent produced a proposal and training addendum; nothing from it was implemented or trained. [Proposal/review](blind-architecture-review-2026-11-09.md). |
| Have the performance percentages solved the user's CPU requirement? | No. Host work must keep up with the GPU; the current measured gains do not prove that. |

## Overall lessons from the recorded results

1. Correct arithmetic, connected gradients, immutable bindings and exact resume
   can be established without establishing useful learned semantics.
2. Lower training loss repeatedly failed to transfer to held-out families,
   quantities, operators or wording. Some small-panel successes regressed in
   broader training.
3. Changing representation alone could destroy already-working Choice behavior.
   Richer or wider inputs were not automatically better under the tested fits.
4. Coverage gaps are real, but familiar-task failures also exist. Neither missing
   data nor model size has been isolated as the sole explanation.
5. Local kernel speedups and CPU metadata gains must be distinguished from
   complete training throughput. Several faster local operators failed whole-path
   numerics; several correct implementations failed performance acceptance.

There is no defensible complete GPU-hour total in this summary: measurements
overlap, controls were reused, some updates were replayed, and some interruptions
left unknown unflushed work. Individual linked reports retain their measured
durations and scope. Repetition axes and derived records are not independent
experiments or independent semantic evidence.
