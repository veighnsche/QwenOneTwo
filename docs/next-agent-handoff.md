# Next-agent handoff — 2026-11-06

Read this first for current status, then the
[implementation specification](architecture-and-training-plan.md) and its linked
topic contracts. Historical reports describe their recorded sources and runs.
This handoff records measured mechanisms, not learned capability.

## Current checkpoint

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

**Reference recovery failed after the authorized fixed fit.** Use the retained
1,000-update reference, exact 900/1,000 reports and pinned existing releases.
Investigate held-out recovery and document a development revision before any
new fitting. Do not generate more data, extend epochs automatically, relax the
95%/90% gate or use locked cases to repair development results. A frozen export,
low training loss and 100% contrast separation cannot promote this reference.

An independent performance slice can reduce the measured eager recurrent/
backward dispatch overhead. Preserve FP32 transformer computation, compensated
BF16 vocabulary projection, cache continuity, connected gradients, accounting
and existing tolerances. Qualify affected arithmetic before relying on it; do
not repeat the completed 324/108 matrices for unrelated changes.

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
