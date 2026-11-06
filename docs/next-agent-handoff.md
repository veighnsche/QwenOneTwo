# Next-agent handoff — 2026-11-06

Read this first for current status, then the
[implementation specification](architecture-and-training-plan.md) and its linked
topic contracts. Historical reports describe their recorded sources and runs.
This handoff records measured mechanisms, not learned capability.

## Current checkpoint

**The exposure-controlled reference comparison is complete. Extra exposure
repairs all four tested Question/Target training failures, with no diagnostic
regressions; development recovery remains 0/100. No job remains active.**
Implementation/protocol `ade8246` preceded the run. Read the
[measured report](reference-exposure-results-2026-11-06.md),
[portable receipt](evidence/reference-exposure-comparison-v1.json) and
[fixed comparison protocol](reference-exposure-comparison-v1.md).

The completed broad control was reused. The treatment restored its exact initial
weights, optimizer and RNG, then completed **1,000 pooler-only updates / 8,000
eligible presentations** on the RTX 3090. Exactly **336 same-role repeat slots**
were reallocated, preserving all **3,376 distinct views**, every original first
occurrence and every slot's role. Questions 0000/0010 increased from 1 each to
86 each; Targets 0001/0011 from 2 each to 85 each; Keys 0008/0018 stayed at 81.
All **49 original computation sources** matched, and all **35 unchanged initial
updates** reproduced control loss and gradient norm exactly.

| Metric | Original control | Treatment 900 | Treatment 1,000 |
|---|---:|---:|---:|
| Six training controls | 2/6 | **6/6** | **6/6** |
| Fixed first diagnostic set | 11/20 | **15/20** | **15/20** |
| Fixed separate screening set | 10/20 | 10/20 | 10/20 |
| Combined training diagnostics | 21/40 | **25/40** | **25/40** |
| Original development recovery | 0/100 at both | **0/100** | **0/100** |
| Original development contrasts | 100/100 at both | 100/100 | 100/100 |

Exactly the four targeted cases improve. All 21 baseline diagnostic recoveries
remain exact, with zero losses across the fixed 40 cases. Diagnostic baseline
is the old final 1,000-update control; development compares the corresponding
900/1,000 checkpoints. Both >=95%/90% development gates still fail. This is
evidence that targeted exposure allocation contributes to these training
failures, not qualification or a unique explanation of generalization failure.
Most low-exposure views were not boosted; neither broader retention nor a fully
balanced schedule was tested. Do not claim catastrophic forgetting: the old
broad control restarted from the common initial state, not the six-case probe.

**Three metadata tests passed in 0.001 s**, syntax checks passed, and the full
metadata/serialized-byte audit passed. All **1,396 initial tensors**, optimizer
and RNG bytes match the control. All eleven checkpoints retain **1,338 unchanged
non-pooler tensors**, including **483 T/codec state tensors**. All **320 frozen
base tensors/tying** pass; decoder/live identities are unchanged; all gradients
finish None. Existing numeric matrices were reused with unchanged computation
sources. No new source data/reviews, agents, external model calls, capability
updates, locked-label fitting, promotion or service changes occurred.

Training: **3,376 W feature prefills / 131,255 positions**, CUDA cache;
**246,970 charged positions / 238,970 teacher-forced forward input positions**.
Run time after metadata preparation **3,236.2155652950023 s**, median update
**2.550225070999659 s**, peak CUDA **4,193,888,256 bytes**. No arbitrary time cap.
Runner exit **0/COMPLETED**, `reference_qualified=false`. Raw evidence:
`data/reference-exposure-20261106/run/`, plus `audit.py`, `audit.json`, `fit.log`
and `command.json` in its parent. Final unqualified reference fingerprint:
`d741b65f85ef4d381c74bf48b3bf2330de3009f8e561e20d9e3bdc389ffdeb42`.

Next unfinished slice: **reference recovery/generalization under broad training
coverage**, preserving retention and original paired qualification. A broader
exposure schedule is supported as a next experiment to discuss; no additional
fitting, data generation or capability training is queued. Preserve all W,
RTX 3090, latent/cache, precision and original-tolerance constraints.

## Previous completed block search

**The arbitrary time cap is removed, and all 32 candidates plus all four
screening evaluations are complete. No finalist met the original improvement/
retention rule. No job remains active.** Implementation `0d6b1e9` adds result
resume and removes the deadline. `AGENTS.md` records the user's instruction to
keep local machinery simple, finish agreed finite work, reuse valid measurements
and avoid unrequested agent-chosen time/token/cost cutoffs.

Read the [completion report](reference-block-search-completion-results-2026-11-06.md),
[portable receipt](evidence/reference-block-search-completion-v1.json) and
[resume protocol](reference-block-search-resume-v1.md). The old timeout report
below remains historical evidence and no longer describes current coverage.

All **21 complete reports were reused byte-for-byte**, and mask 20 reused its
**nine completed greedy recoveries**. Its nine sensitivity forwards had to be
reconstructed because the old partial log omitted foil log probabilities; these
are now recorded. The remaining 11 candidate reports completed. Ranking over
all 32 selects **20, 1, 4**, and all received the original separate screening.

| Mask | Components | First set | Screening |
|---|---|---:|---:|
| 0 | Baseline | 11/20 | 10/20 |
| 20 | First attention block + lift | 12/20 | 7/20 |
| 1 | Queries | 11/20 | 10/20 |
| 4 | First attention block | 11/20 | 9/20 |

Mask 20 gains two first-set examples while losing one; it gains none on screening
and loses three. Mask 1 preserves but does not improve screening; mask 4 loses
one. None meets strict screening gain with preservation of every baseline
recovery in both sets. **NO_FINALIST** is a completed experimental outcome.
No development qualification was warranted, and no new development score,
locked-label use or live promotion occurred. Both diagnostic sets are historical
training data; they are not unseen during prior model training.

New work: **40 W prefills / 1,514 annotation positions**, **300 sensitivity
prefills**, **291 recovery episodes**, **7,807 decoder positions**. Two-run physical
totals: **729 sensitivity prefills**, **720 recovery episodes**, **19,346 decoder
positions**; logical complete coverage is **720 cases per metric / 19,252 decoder
positions**. The 94-position difference is the nine necessary sensitivity
reconstructions. No greedy recovery was repeated. Runtime **511.3701073270022 s**;
peak CUDA **3,622,856,704 bytes**; exit **2/NO_FINALIST**, `search_complete=true`.

**Seven metadata tests passed in 0.016 s** for resume/corruption/case identity and
existing selection rules. Unchanged prior GPU block checks were reused. All
320 frozen base tensors/tying, 483 T/codec tensors, live identity, source artifacts
and release manifests remain unchanged. Baseline restored; all gradients None;
zero agents, gradient/optimizer/capability updates or new source records. The
metadata audit verifies reuse, complete coverage, actual work and selection.
Raw evidence: `data/reference-block-search-completion-20261106/run/`; older run
and all evidence retained unchanged.

This block-replacement family and its screening are finished. Next unresolved
work is reference recovery/generalization and retention, including a causal
exposure-controlled comparison for the documented imbalance. No further search,
data generation, gradient fitting or capability training is queued. Preserve
W freezing, the RTX 3090-only rule, original >=95%/90% paired qualification and
latent/cache contracts. Runtime is an observation, not a stopping condition.

## Previous interrupted block search (historical; completed above)

**The diagnostic runner is implemented; the bounded block search stopped
INCOMPLETE at its fixed 600-second search limit. No job remains active.**
Implementation/protocol `87b01c0` preceded measurements. Read the
[measured report](reference-block-search-results-2026-11-06.md),
[portable receipt](evidence/reference-block-search-v1.json) and
[fixed protocol](reference-block-search-v1.md).

The runner tests five-component replacements from the saved 6/6 pooler into
original checkpoint 1,000, always restoring from immutable source tensors.
There are 32 masks including baseline/full donor; **21 completed**, mask 20
completed **9/20 cases**, and masks 21–30 were never started. No completed
configuration exceeded baseline **11/20** on the fixed training search set.
Full donor reproduces 6/6 controls but only **7/20** overall. Masks 18/19 gain
one Target while losing three other recoveries, finishing **9/20**. No shortlist,
second-set evaluation, finalist, new development qualification or live promotion
occurred. Do not call this complete search coverage or a recovery fix.

The two prepared 20-case sets cover all ten roles with disjoint annotation hashes
and families; both are historical training data. Prior exposures are a concrete
finding: control Questions **1 each** in the broad run vs **86 each** in the donor;
Targets **2 each** vs **85 each**; Keys **81 each** vs **85 each**. The broad run
restarted from the original parent, so this is not evidence of forgetting the
6/6 checkpoint. Baseline correct packets improve the desired token probability
in **15/20** paired checks; the first differing token is predicted correctly in
**14/20**. This supports partial packet use, not semantic competence.

**26 checks passed** (5 metadata, 2 CUDA exhaustive apply/restore, 19 related
regressions). All **320 base tensors/tying**, live identity, **483 T/codec tensors**,
parent artifacts and release manifests remain unchanged; original baseline was
restored. All owner gradients are None. Independent byte audit reconstructs all
**22 applied fingerprints**, verifies **55 source/protocol/test files**, case
separation and exact counters: **40 W prefills / 1,514 annotation positions**,
**429 sensitivity prefills**, **429 recovery episodes**, **11,539 decoder positions**.
Zero gradient/optimizer/capability updates, agents, external calls or new records.
Development and locked record labels were not loaded. Selection on labels is
explicitly data-driven weight optimization despite having no gradients.

Total runtime **691.9866396940015 s** including setup/final integrity checks;
median completed configuration **27.0861807830006 s**; peak CUDA **3,622,856,704
bytes**; exit **2/INCOMPLETE**. Raw evidence is
`data/reference-block-search-20261106/run/`, including all complete and partial
measurements and exact reproducible edit recipes. No further run is queued.
Next unfinished machinery is a cheaper initial screen, reserving full recovery
for shortlisted candidates, with a measured budget before another bounded run.
Partial mask 20, ten unstarted masks and second-set checks remain unmeasured.
Reference recovery/generalization, semantic binding and the full learned flow
remain unresolved. Preserve original >=95%/90% paired qualification, W freezing,
RTX 3090-only workloads and all existing scope restrictions.

## Previous single weight repair (historical)

**The one authorized weight-space repair completed and failed. No job remains
active.** Implementation/protocol `54a171a` preceded application and all semantic
feedback. Read the [measured report](reference-weight-repair-results-2026-11-06.md),
[portable receipt](evidence/reference-weight-repair-v1.json) and
[fixed protocol](reference-weight-repair-v1.md). One common-mode projection rule
was applied to both existing final reference checkpoints, with no strength/search
retry. Both were exported before development feedback. Calibration reused 71
reviewed training views, 71 W prefills and 142 pooler forwards, plus four checks
on cached states. No new records/reviews, agents, gradients, optimizer steps,
fine-tuning, capability updates or external model calls occurred.

Both repaired 900/1,000 exports remain **0/100 exact recovery and 100/100
contrasts**, matching the hash-verified parents; the production loader rejects
both. The original >=95%/90% paired criterion is unchanged. Only four lift
weight/bias tensors changed per artifact; all other **537/541 tensors**, including
**483 T/codec tensors**, are byte-identical. All **320 base tensors/tying**, live
identity, parent exports and three release manifests are unchanged. Locked labels
were not loaded. Projection residuals pass; recovery does not. **2 metadata tests,
2 RTX 3090 synthetic tests and 19 related regressions pass**; all 54 archived
source/protocol/test hashes and export hashes verify independently. Runtime
284.00155400899894 s; peak CUDA 3,622,856,704 bytes; exit 2/FAILED.

Raw evidence/exports: `data/reference-weight-repair-20261106/run/`. Repaired
fingerprints: 900 `41bb9e035d9036cb182d16f965039d64a720c9cc85e0f6dbd7cf61235a1c0f29`;
1,000 `e40c025b621a4c587f93811f4e22aa66c25a1fb9b0b06874212a7d2ca6c66ada`.
These are unqualified and never promoted. This candidate is retired; no follow-up
edit/fit is queued. Next unfinished is full-mixture reference recovery and
retention; semantic binding, observed correction materialization and full
participating-trainer memory remain separate. Keep W frozen and all existing
scope restrictions. Numeric/cache/runtime code and live services were unchanged.

## Previous encoder checkpoint (historical)

**The Target-private encoder block finished and failed qualification. No fit is
running. The user's requested recovery fix remains unfinished.** Implementation
`fb920fd` (after `7b5aaed`) is committed/pushed. Read the
[full measured report](reference-encoder-results-2026-11-06.md),
[portable receipt](evidence/reference-encoder-v2.json) and
[preregistered Target-private v2](reference-encoder-target-v2.md).

Target-only learning recovered both Target controls while regressing other
roles. The minimal private Target branch then passed **6/6 focused training
recoveries**, exact initial migration for all ten roles, 30 total queries,
private gradients and export/load. **Six GPU regressions passed in 15.560 s;
68 model-free checks passed in 2.811 s.** Pooler parameters are 4,213,760 versus
2,110,720 originally. Other nine roles remain shared; W/T/codec are not duplicated.
Five earlier candidate probes failed and are retired; seven probes used exactly
448 reference updates / 3,584 exposures and 96 separate latent-variable steps.
All raw sources/results remain under `data/reference-encoder-20261106/`.

The accepted architecture's full run reset to the original parent, not probe
weights. It completed **1,000 reference updates / 8,000 eligible exposures** over
all **3,376** existing reviewed views, exact sampler/owners, no failed exposures.
Every gate 100–1,000 remained **0/100 exact recovery, 100/100 contrasts**; both
900/1,000 are rejected by the real qualification loader. Initial checkpoint
save/restore succeeded. All 483 T/codec tensors are byte-identical to the parent
at all 11 checkpoints, live weights unchanged, 320 frozen tensors/tying pass,
all checkpoint/export/source hashes verify and all three release hashes are unchanged.
Mean first/last-100 CE is 0.5453864361013985 / 0.5070646067904454; median window
2.4458106085003237 s; run 3913.555469761999 s; peak 4193888256 bytes; exit 2/BLOCKED.
This is semantic failure, not unavailable hardware. Capability updates are zero.

Final export: `run/reference-1000/`, fingerprint
`6b90e7b706239374a088462777eb072cd166a055e353135b0058c583b9da1dd8`.
GPU readback verifies it and connected frozen-decoder input gradients, but the
full-data artifact recovers only **2/6 training controls**. The focused 6/6 gain
does not transfer to the full recipe. Do not certify the encoder/learned flow.
The lineage has 3,000 reference updates; physical reference work is 3,464 including
448 probes and the original discarded 16-update setup. Latent-variable steps
are separate (96 in current probes, 384 in earlier isolation).

Before the bounded repair above, the user asked an opinion on direct weight-space
programming. Primary research was checked and a **read-only** RTX 3090 weight audit performed;
no edit or external model/agent call in that read-only audit. Projection spectra do not show obvious
rank collapse or dominant bias. This does not rule out a targeted repair, nor
demonstrate one. W remains frozen. No additional fitting, generation, agents,
capability training, numeric matrix rerun or service changes are queued/performed.
Next unfinished is full-mixture reference recovery/generalization and retention;
keep the unchanged >=95%/90% final-two gate and locked-data prohibition.

## Prior recovery revision checkpoint

**The earlier recovery-v1 revision finished and failed.**
The user's requested recovery fix is **not proved**. Implementation `908d8b5`
reuses 3,376 existing reviewed role views with source hashes/field addresses,
exact coverage/resume and the Reason exception. A separately documented
[revision](reference-recovery-revision-v1.md) warm-started the old reference,
reset reference AdamW, and ablated preceding-target vectors in half the exposures.
See the [final measured report](reference-recovery-results-2026-11-06.md) and
[portable receipt](evidence/reference-recovery-v1.json).

Exactly **1,000 reference-only updates / 8,000 eligible exposures** completed on
the prescribed RTX 3090. All 3,376 views were used, one to 82 times; 4,000
exposures used each history mode. Every gate 100 through 1,000 remained
**0/100 exact recovery, 100/100 contrasts**. Both final reports are rejected by
the actual qualification loader. No promotion, live initialization or capability
update occurred. Live hashes match, all 320 frozen tensors/tying pass, all
checkpoint/export hashes and 49 archived source files verify, and all three
release hashes remain unchanged. Mean first/last-100 losses were
4.7712684374183665 / 1.1359519423067104; median window 3.4190447954897536 s;
complete runtime 4916.259019798017 s; peak CUDA allocation 4276769280 bytes;
runner exit **2/BLOCKED**. This is failed recovery, not unavailable hardware.

**68 model-free checks passed in 2.658 s; four GPU objective checks passed in
13.270 s; three focused catalogue checks passed in 0.044 s.** The final reference
fingerprint is `d0d4aef44b4415beb19311350eb24533190e1a27406c5d6f59a15a5a57402dec`.
Its lineage has 2,000 reference updates; total physical reference work is 2,016
including the previous discarded 16-update setup attempt. Do not confuse those
counts with this revision's 1,000 updates. Raw evidence is under
`data/reference-recovery-20261106/`, final export `run/reference-1000/`.

Read-only diagnosis supports conditional/template overfitting: the old reference
recovered 31/50 unique training meanings versus 0/50 development. Revised actual
reconstructions recover unrelated reservation/criterion/key meanings, often the
same text for different inputs. These are substantive errors, not paraphrases.
A 20-case attention probe does not support chat-header-only pooling. A read-only
persistent-latent conditioning intervention worsened CE and was not installed;
no extra fitting block followed it. Sparse Key/Midpoint training views still
contain only five meanings each. These findings do not prove recovery impossible
or establish that new data is necessary. Read the final report before selecting
any further revision; retain the unchanged >=95%/90% final-two gate.

The [bounded training-only isolation probe](reference-recovery-isolation-v1.md)
completed: original pooled vectors recover **1/6**, disposable optimized payloads
recover **6/6** by step 16 and at step 64, with W/pooler/codec/T/live weights
unchanged. There were **384 disposable latent-variable steps and zero model
optimizer updates**; do not add these to model-fitting counts. These are fitted
positive controls, not a trained encoder or held-out recoveries. For these six
training cases, the decoder/wire can express the annotations but the produced
representations fail. Next unfinished is a causal encoder/pooling learning
revision; held-out decoder generalization also remains unproved.
No source generation, agents, capability training, numeric matrix rerun or
service change was performed. Actual semantic qualification, predicted-packet
binding, observed correction materialization and full-trainer memory remain
unfinished.

## Prior CPU performance checkpoint

The user authorized fixing CPU dispatch overhead with **no generation, agents
or extra training**. Implemented the separately versioned
[graph-v1 recurrence](codec-recurrence-graph-v1.md), now the expanded backbone
default. It replays the original token-ordered CUDA operations and PyTorch VJP;
it preserves v3 BF16 storage / FP32 transformer / compensated BF16 vocabulary
arithmetic. Eager remains explicit and supplies the genuine reference. See the
[measured report](recurrence-graph-results-2026-11-06.md) and
[portable receipt](evidence/recurrence-graph-v1.json).

**45/45 focused full-model traces passed in 944.7234295289963 s**:
128/512/1024 positions × Noul/Choice/Score × eager/graph functional/graph
checkpoint/graph serving/genuine FP32. Graph states, logits, scalars, budgets,
discrete decisions and functional/checkpoint gradients were exact against eager.
All thirteen status cells occur. Genuine FP32 comparison worst logit/gradient
RMS errors were 4.169371270284753e-6 / 8.316676446703025e-6, within existing
tolerances. Live weights were unchanged; all 320 frozen tensors and physical
tying passed. This focused protocol does not claim a repeat of the entire
original 432-case matrix, full-trainer memory or learned semantic qualification.

**Eight operator tests passed in 3.745 s**, including 32 output/all-input-gradient
comparisons, 512/1024 connected compositions, cold checkpoint capture, LRU
eviction, stream isolation, three repeats, nonfinite guards and four small GPU
FP64 directional checks. **65 model-free regressions passed in 2.801 s.**
**Eight affected integration checks passed in 42.783 s**, covering teacher-forcing
parity, frozen projection/vocabulary derivatives, scoped T checkpoint replay,
export/tying and frozen-reference input gradients without live-owner gradients.

Measured existing trained-reference forward/VJP median: **1.4445 → 0.3463 s
(4.17×)**. Host `cudaLaunchKernel` calls fell **52,243 → 9,007**; graph replays
are separate submissions. Device kernels increased **52,497 → 58,905**.
The graph path added 72 replay submissions (9,079 total launch API calls).
The fix reduces host dispatch, not device kernel count. Two actual scheduled
eight-exposure windows with real finite guards/accumulation improved
**18.7282 → 5.8341 s (3.21×)** warm median. Buffers were discarded; optimizer
updates stayed zero and reference weights were unchanged. Cold/capture times
and all samples are retained. No full optimizer-step throughput is claimed.

An initial fused Triton candidate passed local tests but failed 512-position
serial-feedback FP32 tolerances. It was rejected; its source/logs remain under
`data/recurrence-fused-20261106/`. Accepted graph evidence is under
`data/recurrence-graph-20261106/`. No tolerances were widened, extra fitting was
performed, existing data changed or live service touched.

**Next unfinished remains reference generalization:** the previous fitted
reference still failed recovery. Performance is improved; no semantic promotion
was granted. Investigate the retained fit/existing data and document a development
revision before another fitting run. Full participating-trainer memory, actual
correction bindings and later semantic/promotion checks remain outstanding.

## Prior reference fit checkpoint

The user subsequently authorized **reference fitting and qualification with the
existing data on the RTX 3090**, no generation, agents or capability training.
The fixed fit is complete, but **qualification failed**. See the
[fit/performance report](reference-fit-results-2026-11-06.md) and
[portable receipt](evidence/reference-fit-v1.json).

`scripts/fit_reference.py` completed **1,000 reference-only updates / 8,000
eligible exposures**, every one of 500 IDs exactly 16 times, with zero failed
exposures. Those rows contain 50 unique training meanings. Both final checkpoints
900/1,000 achieved **0/100 exact recovery (100 unresolved), 100/100 contrast**;
all ten scheduled gates had those counts. The actual qualification loader rejects
both final reports. No reference promotion or live initialization occurred.
The blocker is now **failed held-out recovery, not a missing trained artifact**.
Mean loss over the first/last 100 windows was 2.658715932392515 /
0.046998847847844444. Live parameters remained byte-identical; frozen W passed
320-tensor equality and physical tying. Peak CUDA allocation was 3,627,365,888
bytes; elapsed 10,568.742410472012 s; runner exit 2/BLOCKED.

The causal teacher-forcing optimization passed **2/2 RTX 3090 checks in 12.574 s**
for serial/parallel loss and connected input/T gradients plus suffix causality.
Existing tolerances/arithmetic are unchanged. **63/63 model-free checks passed
in 2.799 s.** Checkpoint 0000 saved/restored before the corrected fit; all periodic
full checkpoints and frozen reference-only exports are retained. Final export
readback verified fingerprint
`fb38a31359a4ff367780bb8729695894e663810efd124c016bc2d5ba3fdbf7de`.
The first attempt's 16 updates were discarded after a checkpoint owner-name bug;
its log/evidence remains. They are additional physical work, not part of the
final artifact's 1,000-update lineage. No automatic extra epoch followed failure.

The user's suspected CPU bottleneck is supported by measurement: **99.39% of one
CPU core / 16% mean GPU use** during the fit. A subsequent derivative-only trace
of one existing episode counted **52,497 CUDA kernels**, 172.78 ms kernel union
within a 3.07 s instrumented episode (profiler overhead applies). Weights,
features and gradients were CUDA; no CPU model fallback occurred. Two warm VJPs
and one traced VJP added **zero optimizer updates**. Eager recurrent/backward
dispatch is the first performance target; accumulation/optimizer costs were not
included in that trace. No performance kernel/arithmetic change was made.

Raw evidence: `data/reference-fit-20261106/`; final export
`run/reference-1000/`; checkpoints/reports/source snapshot in `run/`.
Data hashes and live services remain unchanged. No capability training, new
data, agents or completed numeric matrix reruns occurred.

**Next unfinished:** investigate failed reference generalization from the retained
run/existing data, document a development revision before another fitting run,
and obtain a real passing final-two-checkpoint gate. Independently, reduce CPU
kernel-launch overhead only with appropriate numeric/gradient qualification.
Do not extend epochs, relax gates, tune on locked tests or start capability
training automatically. Full-trainer memory and later semantic/promotion checks
remain outstanding. Earlier missing-artifact/no-fitting statements below record
their historical checkpoint, not the current state.

## Previous checkpoints (historical)

The user requested the remaining reference qualification/binding work with
existing data, **no generation, agents or training**. The read-only runner,
qualification-evidence gate and conservative actual-content binding are now
implemented. See the [reference/binding report](reference-binding-results-2026-11-06.md)
and [portable receipt](evidence/reference-binding-v1.json).

**Actual learned reference qualification is BLOCKED.** No compatible trained
reference artifact was found among 107 local safetensors headers. The explicitly
untrained frozen reference recovered **0/100 exact canonical annotations** (all
100 unresolved, ten per role). The same tensor fingerprint separated **100/100
role contrast groups**, reused from 40 existing reviewed intervention pairs
across 39 families. No new contrasts/scenarios were authored or reviewed. The
missing standalone contrast manifest was resolved by materializing the existing
reviewed transformations and identical same-family positives. Contrast separation
alone cannot satisfy the >=95% recovery / >=90% contrast gate. No qualification
was granted and no fitting was performed.

**63/63 model-free checks passed in 2.647 s; nine distinct focused GPU checks
passed** (5/5 in 37.414 s, 3/3 in 53.287 s, and 3/3 in 18.618 s with two relevant
comparison reruns). Alignment/reconstruction retain input gradients through frozen
T without live-adapter/reference gradients in the isolated engineering fixture.
Binding checks actual role contents, source identity/states/coordinates, rubric
metadata and stable candidate identities, including reorderings. Outcome labels
require a measured qualified reference and current admitted snapshot; actual
return compatibility additionally gates continuations and later guided calls.
Unresolved learned deviations remain masked; a decoded gloss alone cannot approve
them. Explicit diagnostic reference episodes reject optimizer accumulation.

`scripts/qualify_reference.py` is read-only. Supply the pinned development release
and a frozen reference bundle to qualify; `--diagnostic-untrained` is explicitly
nonqualifying. `--contrast-only` avoids repeating completed RF measurements and
cannot grant qualification. Actual diagnostic/preflight exits were 2/BLOCKED.
The original setup rejections and exact recovery/contrast source snapshots are
retained. Existing annotation release hashes, services and prior numeric evidence
remain unchanged. Model-service calls, agents and optimizer updates were zero.

**Next unfinished:** a genuinely trained frozen reference artifact and its passing
qualification. That requires later fitting authorization; do not create one under
the current no-training instruction. Existing data already supplies the recovery
and applicable contrast cases. After qualification, enable supported producer
bindings and perform the outstanding full-trainer memory and semantic/promotion
checks. Observed correction materialization remains distinct. No further data
generation is queued. Earlier missing-runner/binding statements below are history.

The user authorized episode production and proof of latent Qev invocation using
existing data, **no generation, agents or training**. Implemented:
`training/producer.py` supplies native complete-episode callbacks;
`training/component_producer.py` materializes clean receiver/reader/decision replay
and selects reference-stage annotations. `Session.evaluate` rejects textual and
dictionary requests before any forward. Clean packets can execute their actual
preparation positions in the same Session. See the
[producer report](episode-producer-results-2026-11-05.md) and
[source/log-bound receipt](evidence/episode-producer-v1.json).

**57/57 model-free checks passed in 2.217 s; 11 distinct focused RTX 3090 GPU tests
passed.** Eight actual checks passed in a 77.136 s run that also had one incorrect
test-class selector (overall exit 1); the corrected selector plus a new free-window
guard then passed 2/2 in 15.833 s. A final tied-choice reader binding regression
passed 1/1 in 16.793 s. This runner error and the initial genuine
untrained Reason failure are retained. No numeric matrix was repeated.

Transport proofs use existing train-writer-0004/0000/0001: Noul/Choice/Score request
tensors have 103/143/110 positions, width 1024, FP32 CUDA. Embedding lookup and
tokenization are disabled during each call. Actual Qev outputs are delivered;
native losses have connected packet gradients, with one prefill/cache lineage,
exact accounting and no remaining holds. Optional calls have no textual decision
prompt. Initial human-text W+D prefill/continuous effort selection is preserved.
Positive fixtures prescribe writer controls and Reason child-count zero; clean
component tests use an untrained frozen reference. **This is transport/construction
evidence, not learned invocation, qualified reference semantics or training readiness.**
The unmodified Reason decoder still fails honestly with zero losses and no gold
repair. No optimizer step, reference fitting, data-service call, agent or service
change occurred; existing annotation release hashes remain unchanged.

Next unfinished: qualified reference recovery/contrast and verified binding of
actual predicted request meanings to supervision. Joint semantic/outcome and
post-return continuation losses stay masked; later annotated guidance stops at
the first unverified return. Writer semantic-loss integration, compatible serial
continuations and actual correction bindings remain unfinished. Generic unknown
candidate feasibility is masked rather than inventing a four-class label. Full
trainer-memory qualification still precedes training. Do not launch training or
regenerate data. Older missing-producer statements below are historical.

The user authorized the remaining data collections with cheap batched generation,
reuse, model content review and **no training**. Complete annotation releases
now exist under `data/training-collections-20261105/`:

- Training: **8,000** records; manifest
  `c70d4f44135a34c00e7fa0348833b1ec56613def15f5753231589492ed35478f`.
- Development: **1,000** base cases plus 40 secondary mechanical executions;
  manifest `5b8f4cffdc4f33e255d3a34306cac03a68c50dcd9dab905ae8d8adcf89c81e77`.
- Locked: **2,000** cases; manifest
  `d13594f18738056ee179b3498d54d8261f7f7654aad633fabb776810681aa1cf`.

See the [collection report](training-collections-results-2026-11-05.md) and
[portable receipt](evidence/training-collections-v1-summary.json).
`catalogue.json` identifies usable releases; `*-candidate-*` folders are history.
The pilot remains development-only. Content review is explicitly compositional:
actual source-state/canonical-definition reviews plus exact reviewed view recipes,
with per-record hash and transformation checks. It does not claim 11,000 separate
literal-row model judgments or independent scenarios.

Actual usage: **31** capped Muse calls, 965,889 input / 254,566 output tokens;
no agents, automatic retries, local model runs or training. **64/64 model-free
tests passed in 2.198s**. Independent audit accepts all releases, exact allocations
and family/status/domain minima, disjoint split families/input payloads, five
reference meanings per role, 30,302 exact source literals and 35 hidden-fact pairs.
Production sampler ID windows work in all five phases. One unsupported development
effort preference is masked and omitted from effort-scored free-job selection.

The 400 correction records contain prescribed intervention specifications;
correction losses stay masked until a producer binds an actual runtime packet/
snapshot. Replay is a generated preservation proxy, not recovered original data.
The producer slice above now implements token/source and episode materialization;
qualified reference artifacts, semantic binding and
the documented full-trainer memory qualification remain prerequisites. No
training, fitting, deployment, service changes or further generation is queued.
Earlier no-training-release statements below are historical.

The user explicitly authorized reviewing the existing 24 drafts and generating
the remaining 96 for the pilot, with no humans, agents or training. **Complete:**
`data/pilot-120-20261105/release/` loads as an approved 120-record development pilot
under `automated-content-v1`, with 20 per domain and all 13 operation/status cells.
See the [pilot report](pilot-120-results-2026-11-05.md) and
[portable receipt](evidence/pilot-120-v1-summary.json). Manifest SHA256:
`177147b46629f7ee2ff39b567fd90543621fba07fed6960bca123b6aa32b4c9d`.
All 120 final records bind actual separate outcome-blind model review; no human
review/certification is claimed. Existing 24 inputs and 125 original run files
remain unchanged. 14 capped Muse calls used 257,768 input/76,344 output tokens;
six generation, six initial reviews, one 21-case correction review and one
replacement review after a duplicate was caught. No automated retry loop.
Independent checks bind final inputs/outcomes to actual review bytes; 55/55
model-free tests passed in 1.334s. Draft validation now correctly allows unknown
nonselected candidates in assessed any-workable Choice while requiring selected
candidates to be known feasible. No numeric/model/GPU tests were rerun. No
training, service changes or deployment occurred. Earlier no-pilot statements
below are historical; do not regenerate this pilot.

The user explicitly removed mandatory human review for this experiment.
Current policy is mechanical checks plus one separate, outcome-blind
model-assisted content review per record (batched calls permitted), followed by
automated release approval. This applies to pilot, training, development, locked,
subjective Score/effort and new rubrics. No reviewer hiring, mandatory second pass
or human sign-off remains. `kind:model` review evidence binds actual model/session
and prompt/response assets; `kind:automated` approval binds policy and release.
Old human-review evidence is still readable but is not required. Unreviewed,
rejected, stale or quarantined records still reject. Do not automatically retry
generation/review or mislabel model review as human certification. This policy
change performed no model calls and approved no real dataset. Historical review
requirements below describe earlier checks, not the current policy. See the
[validation receipt](evidence/automated-content-review-v1.json) and
[current review procedure](data-operations.md#5-independent-semantic-and-outcome-review).

The user authorized small fixes for confirmed overengineering, excluding generation
orchestration/cost controls; no model calls are permitted for this cleanup.
Draft inspection now lives in `scripts/validate_drafts.py`, outside the training
package. It preserves source/hash, role/Reason and unreviewed-draft checks without
fixed counts, IDs, domain quotas, producer-name heuristics or run/release identity
equality. Run `.venv/bin/python -m scripts.validate_drafts <scenarios.jsonl>`.
Acceptance: arbitrary-size/ID/run batches validate, duplicate IDs reject and draft
records cannot pass release. `PYTHONPATH=tests .venv/bin/python -m unittest
test_drafts test_training_inputs -q`: **25/25 passed in 0.012 s**. Both saved
12-record batches also pass draft checking; **24/24 reject release**. No data
bytes, annotations or historical evidence receipts changed.

Pilot policy now accepts authored or model-proposed fictional roots without
changing the 120-family/20-per-domain allocation, two independent human reviews,
source/review/approval byte binding or split protections. Model origin/author and
assistance metadata remain intact; no real release was approved. Acceptance:
schema-only reviewed model roots load, missing provenance/second review rejects.
`PYTHONPATH=tests .venv/bin/python -m unittest test_training_release
test_training_inputs test_drafts -q`: **39/39 passed in 0.934 s**.

Release/sampler binding is now checked once against immutable catalogue/pool
contents. Records are snapshotted individually; windows decode only the eight
selected exposures, with fresh copies even for repeated IDs. Acceptance: mutation
of sampler contents rejects, forged initial catalogues still reject, resume is
unchanged, and two simulated windows decode exactly 16 records without rebuilding
the catalogue. `PYTHONPATH=tests .venv/bin/python -m unittest test_training_driver
test_training_release test_training_schedule -q`: **28/28 passed in 1.179 s**.
These are CPU metadata/counter simulations, not model or throughput measurements.

Cleanup code commits: `1e86c08` (draft tool), `3cfa11c` (pilot provenance),
`fa94776` (release windows). README/topic headers now defer current status to
this handoff; obsolete status/authorization banners were removed and older
checklists/audits labeled historical. Acceptance: all **610 local links across
58 Markdown files resolve**, and `git diff --check` passes. Historical measured
reports/receipts and normative numeric/cache/gradient contracts are preserved.
No model calls or GPU reruns were made. Optimizer rollback and finite guards
remain unchanged; their full-trainer memory/performance costs remain unmeasured.
Next unfinished work is still genuinely reviewed records and concrete episode
producers, followed by the already documented trainer-memory qualification.

The user then explicitly requested deletion of the old Muse framework. The
active local `muse-implementation` skill has been replaced with the cheap
single-call default; its four coordinator/runtime/tmux reference files and
reference symlink are deleted. The existing skill name/link still resolves.
Repository `AGENTS.md` now makes the same default persistent. See the
[removal receipt](evidence/muse-default-migration.json). Both generated batches,
raw responses, saved run scripts and pre-existing evidence were hash-checked
unchanged. No model call was made for this change. Historical launch scripts
must not be reused as the workflow. Further generation is not queued.

The user rejected the first batch's disproportionate generation overhead and
requested a fresh batch with cost saving. The replacement is complete in
`data/muse-economy-12/drafts/`: **12 drafts in one Muse completion**, no retries,
tools, children or reminders; 29.969 seconds, 16,293 input / 4,637 output tokens.
See [bounded-generation evidence](muse-economy-12.md). Fifteen model-free tests
pass; all twelve records pass draft validation and reject approved release.
Original response and explicit Codex syntax/content repairs are retained.
This batch is also unreviewed model-proposed development data, not a training
release. Prefer this bounded single-call workflow for small draft requests;
do not restart the rejected coordinator/worker/observer workflow. The old batch
and its evidence remain historical, unchanged. No further generation is queued.

Earlier, the user explicitly authorized Muse setup repair and **12 draft
scenarios**. This bounded batch is complete: two per domain, 17 source files,
all development/model_proposal, zero human reviews and no approvals. See the
[Muse draft report](muse-draft-pilot-2026-11-05.md) and
[portable evidence](evidence/muse-draft-pilot-v1-summary.json). Local data:
`data/muse-pilot-20261105T191615Z/drafts/`; open `review-inputs.md` before the separate
`outcome-proposals.md` for outcome-blind review. This user authorization does not
start bulk generation, capability fitting or deployment. Muse has exited cleanly
and all run-owned workers/commands/tmux resources are released.

Backend implementation slices are committed and pushed directly to `main`.
The [slice report](backend-slices-2026-11-05.md) records acceptance conditions,
results and limitations. No capability/reference fitting, bulk semantic data,
frontend integration or deployment was performed. The Operating Table and other
live services remain unchanged.

Historical release-boundary implementation commit: `4d96af1`.
The prior numeric/finite-guard evidence remains at `abd8dcf`, followed by the
original final handoff at `2da8e43`. The new
[release-boundary slices](backend-release-slices-2026-11-05.md) implement actual
source/annotation/review/approval byte checks, immutable release snapshots,
pilot/driver/sampler binding, exact resumed exposure cursors and a read-only
loader for the existing numeric evidence. No real reviewed pilot was created.

The original **324/324** v3 cases at 128/256/512 remain passed and were not rerun.
The separate **108/108** `v3-1024-v1` matrix passed in **5598.540981531143 s**:
one prefill/cache lineage, connected gradients, exact accounting, unchanged live
parameters and frozen/tied storage. Worst state RMS was 0, logit RMS
`3.12525587881255e-06`, and gradient RMS `3.848441887921371e-06`;
peak allocation was 13,093,139,968 bytes. See the
[1024 measured report](codec-1024-results-2026-11-05.md).

The matrix retains its exact source snapshot from before the subsequent finite
validation guards. Those guards reject overflow before clamp/argmax or dispatch;
finite arithmetic, weights, fixtures and tolerances are unchanged. The separate
post-guard 44-test GPU run passed. Neither result silently substitutes for the
other's source provenance.

The complete learned decision capability remains untrained/unpromoted. Prescribed
operations/statuses, diagnostic weights, fixed continuations and engineering
reference fixtures do not establish invocation, semantic understanding or answer
quality. The old BF16 transformer arithmetic still fails; do not relabel it.

## Preserve these requirements

- One human prompt, one prefill, one continuous hybrid cache and a normal answer
  from that same job. D for initial effort, R for native work, D for internally
  requested Noul/Choice/Score; no keyword/domain routing or second decision job.
- Textless typed requests/results, immutable bindings and selected contents.
  No derivative through argmax; task scalars and selected continuous contents
  retain connected gradients. Startup budget feedback alone is detached.
- Initial effort is continuous scalar regression in its own real LM-head row.
  Preserve the ordered CUDA FP64 S-curve `floor((3*s*s-2*s*s*s)*131072+0.5)`;
  context 262144, separate answer/thinking allowances, cancellation and closure.
- Qualified v3: BF16 frozen storage, FP32 transformer/LoRA computation and causal
  state, compensated BF16 vocabulary projection, genuine FP32 reference.
  Original BF16 transformer arithmetic remains a preserved failed profile.
- Model workloads only on RTX 3090 UUID
  `GPU-225509c3-d702-6c27-de03-76b5e99b33b1`. Never CPU/RTX3060 fallback. Driver
  access required execution outside the sandbox here; diagnose access/device
  selection before declaring GPU unavailable. Do not kill unrelated services.
- Keep The Operating Table's snapshot read-only and its services/artifacts intact.
  Diagnostic checkpoints must never replace live capability weights.
- Commit authorized work directly to `main` and push, per AGENTS.md. No PR/branch
  or repeated permission request is needed. Deployment needs separate authority.

## Completed slices and evidence

1. **Contract/code review and versioned qualification.** Direct implementing-agent
   review of codec/v3 and measured reports; no independent Astra approval claimed.
   The runner now distinguishes the 324-case `v3-512` protocol from the 108-case
   `v3-1024-v1` extension and rejects incomplete/duplicate/foreign/failed coverage.
   Direct review additionally found nonfinite Score clamping, gate acceptance
   and norm overflow. Pre-fix regressions reproduced the defects; finite guards
   now reject them, including terminating invalid gates without dispatch.
   No finite codec arithmetic, fixture statuses, reference or tolerances changed.
2. **F7 inputs/provenance.** Strict per-role/per-branch whitelists, transitive
   dependency checks, byte/hash/version/source coordinates, exposure flags,
   family/split and review-manifest gates. Current labels stay outside request
   encoders; non-assessed Reason has its own role/status exception. Mechanical
   checks cannot establish honest annotation or authenticate human reviews.
3. **F3 clean construction.** Fresh W+D derived source contexts; frozen W/reference
   semantic contexts; independent registries/local identities; complete clean
   packets through production rendering/admission; grounded exact kinds and
   absence/empty distinction. Parent hashes/coordinates remain external audit.
   Hidden-fact twins have identical complete request/return tensors. Unverified
   overlapping token offsets reject before encoding rather than being invented.
4. **Losses and optimization.** A4 subtype losses/masks, actual-snapshot binding,
   per-role semantic normalization, per-component CUDA VJPs/eligible-example
   denominators, eight scheduled exposures, one combined clip/step, unused
   `grad=None`, transactional rollback and named unique owners. Scalar rows are
   isolated. Tiny synthetic optimizer tests ran; real-model optimizer steps=0.
5. **Reference/live and scheduling.** Independent pooler/codec/T sharing frozen W;
   reference objective and frozen decoder preserve input gradients. Reference
   checkpoint replay reinstalls the scoped T mapping; reconstruction shares one
   frozen vocabulary slice per episode. Deterministic A4 sampler, replay recipes,
   prefix guidance, complete-session collection and guarded window driver exist.
6. **Export/resume.** Versioned small-weight bundles, actual reference tensor
   fingerprints, hashes, private effective rows, optimizer moments, phase/sampler/
   RNG state, strict v3/tied-W restore and exact next-update restoration tests.

Final focused GPU suite: **44/44 passed in 22.329 s**, comprising 18 existing
codec/precision, 6 finite-guard and 20 backend tests. Model-free: 10 input/provenance, 4 schedule, 2 matrix coverage and 6
existing exact codec tests pass. Exact commands/timings/log hashes are in the
[slice report](backend-slices-2026-11-05.md) and
[portable validation](evidence/backend-validation-2026-11-05.json).
Two concurrent reference-objective attempts hit GPU OOM; sharing the identical
vocabulary slice resolved the auxiliary retention defect and the affected GPU
suite passed. No numeric matrix restart, tolerance relaxation or CPU fallback.

Continuation validation: **42/42 model-free tests passed in 0.950 s**; after the
final numeric Boolean-marker guard, the **14/14 affected numeric/driver tests
passed in 0.312 s**. The new evidence loader separately validated the existing
108-case aggregate, three child reports and 28 archived source/test files.
Release/driver/qualification preflight imports no torch. These are metadata/schema
checks, not new GPU/model runs. Tests use explicit temporary schema fixtures;
their human/pass markers are never actual review or measured-model evidence.
Exact log/test source hashes are in the
[portable continuation validation](evidence/backend-release-validation-2026-11-05.json).

## Next unfinished slice / blockers

**Reference recovery still fails after the Target-private v2 block.** Use
`data/reference-encoder-20261106/run/` and the current result above. A focused
6/6 training gain fell to 2/6 controls under the separately reset full-data run,
while all 100 held-out cases still fail. Full-mixture generalization/retention,
encoder/decoder behavior and any mechanistic weight repair remain unfinished.
No further fit is running. Do not rerun/extend failed blocks, relax gates, repair
from locked labels, invent semantic approvals or initialize live weights from
these unqualified artifacts. The weight-space question did not authorize a
change to the frozen base; no programmed weight repair has been attempted.

The dispatch performance slice is now implemented and qualified through the
separate graph-v1 protocol above. Keep its source-bound evidence distinct from
the historical eager 324/108 matrices. Capture overhead, dynamic shapes, full
optimizer-step throughput and full participating-trainer memory are not waived
by the reported warm speedups. Preserve all original tolerances and exact
cache/accounting/gradient invariants in subsequent changes.

The read-only qualification runner, reference formation objectives, actual-content
and compatible-return gates, and native/clean complete-episode producers exist.
Qualified reference evidence is still required to initialize live weights and
supervise actual learned deviations. Supported exact fixtures are construction
proofs, not learned semantic flow. Arbitrary predicted slots, observed correction
materialization, full-trainer memory and semantic/promotion evidence remain
unfinished. No capability training is authorized by the reference-fitting request.

Before capability training, qualify the full A4 mix of participating component
buffers, optimizer moments and auxiliary graphs at applicable lengths. The 1024
numeric matrix's conservative memory envelope is not that qualification.
Later phases must retain live optimizer moments, preserve final-two-checkpoint
promotion, and use actual returns without gold repairs. Deployment and live
service changes require separate authorization. No data generation is queued.

## Entry points and artifacts

| Location | Purpose |
|---|---|
| `codec/` | Existing typed runtime, session, v3 arithmetic and qualification |
| `training/inputs.py`, `clean.py`, `clean_packets.py` | F7 builders/provenance and fresh F3 artifacts/packets |
| `training/producer.py`, `component_producer.py` | Native/clean complete-episode callbacks and existing annotation materialization; no auto-launch |
| `training/reference_qualification.py`, `binding.py`, `scripts/qualify_reference.py` | Read-only recovery/contrast gate, qualified-reference evidence and actual-content/return binding |
| `training/release.py`, `qualification.py` | Content-checked release/pilot and existing numeric evidence ingestion; no model execution |
| `training/losses.py`, `episodes.py` | Losses/masks and actual snapshot/session guards |
| `training/engine.py`, `profile.py`, `sampler.py`, `driver.py` | CUDA component buffers/owners, scheduling and producer orchestration |
| `training/reference.py`, `checkpoint.py`, `manifest.py` | Independent reference/T, small-weight export/resume and identity pinning |
| Legacy `train.py`, `train_feedback.py`, `curriculum.py` | Prototype material; not the complete trainer or reviewed data |
| Legacy `server.py`, `thinking.py`, `static/app.js` | Unchanged migration targets; no deployment occurred |

Local inputs, never committed:

- Base revision `2fc06364715b967f1860aea9cf38778875588b17`, read-only snapshot at
  `/home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17`.
  Weights SHA256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
- Qev revision `1d2a6d7f891daab7483a49c3abe507c0c9d7ea62`, local `data/qev/`.
- Frozen diagnostic child `data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors`,
  SHA256 `1dd2e8527e14caa9002d672b4187253cdc174fb3c204ab56e80f257f23943640`.
- Prior 324-case raw bundle: `data/codec-qualification/feedback-v3-20261105/`;
  [portable evidence](evidence/codec-feedback-v3-summary.json). Do not rerun for
  documentation-only or unrelated training-package changes.
- New numeric raw bundle: `data/codec-qualification/1024-v1-20261105/`.
- Backend logs: `data/backend-slices-20261105/`; continuation logs:
  `data/backend-release-20261105/`. Current source/contract identities:
  [continuation backend manifest](evidence/backend-profile-release-v1.json).
  The [prior backend manifest](evidence/backend-profile-v1.json) remains unchanged
  for the earlier slice evidence.

Missing pinned artifacts block model execution; do not download another model,
refit the diagnostic checkpoint or use CPU. The 1024 command is in its
[protocol](codec-1024-protocol-v1.md); always use a new output directory for an
affected rerun. No completed numeric matrix needs repeating for these docs.
