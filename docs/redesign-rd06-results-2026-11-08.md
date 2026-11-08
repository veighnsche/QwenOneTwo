# RD-06 — native direct flow and supervision eligibility, 2026-11-08

Staged model-free implementation acceptance passes. **GPU qualification remains
pending RD-07; no model execution, extraction, actual Session, fitting or optimizer
update occurred.** This audit finds zero complete direct-profile writer formation
targets, zero native decision/post-call continuation bindings and zero direct
supplied-reader support in the existing releases. These are concrete later-fit
blockers, not missing audit work or permission to invent labels or enlarge bounds.

## Native implementation

The [native contract](direct-native-flow-v1.md) versions opt-in
`EpisodeProducer(packet_profile='direct-role-states-v1')`. It reuses the runtime,
registry, writer/Decoder, renderer, Session and existing episode machinery.
Question/target/candidates/rubric use ordered live writer vectors and predicted
1..32 lengths. Rubric offsets are explicit; absent midpoint emits no padding.
Discrete length logits can receive supported structure targets without claiming
an argmax derivative. Attribute/provenance and envelope ABIs remain unchanged.

Sources are the original prompt prefill's selected hidden states, with actual
token/byte coordinates. One prompt/prefill/cache continues through native R
emission, D evaluation, the actual bound return and native continuation. The
direct path prepares optional frozen-W annotation targets before creating the
Session, checking all fields/request/worst-case-return costs before any target
forward. Later source binding attaches only the actual prefill Ref/hash; no
offline source encoder or annotation is inserted into the live invocation.
Legacy pooled/reference controls remain named and unchanged outside explicit
profile additions.

Shape-matched direct alignment is training-only. Variable-layout reference
reconstruction and new-profile Reason supervision are unsupported/masked.
Alignment, vector proximity, a matching record ID or a copied expected answer do
not admit native semantic labels. `bind_labels` rejects native direct semantics;
approved post-call continuation and correction losses remain masked. Guidance
stops at the first incompatible return. Actual results are decoded from Qev heads,
bound to the snapshot and consumed through the existing return renderer/Session.

`NativeEpisode` provides explicit actual-prefill provenance for accumulation;
it permits applicable control/formation objectives, rejects unsupported decision/
Reason/post-call native losses and rejects development accumulation. Prepared
receiver components and supplied readers stay separate. The diagnostic-reference
guard is retained. Approved no-call direct/language replay remains a separate
branch, never evidence that calls or returns are understood.

Reports disclose control prefixes, construct/no-attribute choices, operation,
candidate count, rubric direction, source spans, budget overrides and teacher-
forced native tokens. Actual semantic vectors/lengths remain writer outputs.
Free native tokens carry phase/control tags; returned status/value/selected Ref
and snapshot/result binding are recorded separately from final-answer evidence.
Default numeric kernels, startup mapping, budgets, cancellation and one-cache
Session computations remain unchanged.

## Complete existing-data inventory

The metadata-only Rust BPE/Jinja audit retains **5,590 records**: 5,000 training
and 590 development writer/interleaved/reader/direct/language-replay cases. It
reuses RD-01's verified release, record, source, review and call identities. Raw
rows include every field length/mapping hash, complete observed packet cost,
source mapping, original binding gap and exclusion. No rows are cropped or hidden.
The [portable evidence](evidence/direct-native-flow-v1.json) pins raw artifacts
under `data/redesign-rd06-20261108/accepted/audit/` and current source migration.

| Native formation inventory | Records | Calls Noul /Choice /Score | Complete direct formation | Decision /post-call continuation |
|---|---:|---:|---:|---:|
| Training writer | 750 | 257 /243 /250 | 0 | 0 /0 |
| Training interleaved | 2,000 | 791 /805 /804 | 0 | 0 /0 |
| Development writer W | 70 | 24 /23 /23 | 0 | 0 /0 |

All **3,220 calls exceed the unchanged 64-position worst-case return bound**.
Overlength questions affect 569 writer, 2,023 interleaved and 33 W calls;
overlength alternatives affect 43/134/2 calls. Complete requests exceed 256
positions in 157/989/13 calls respectively. Additionally 67 interleaved records
exceed eight full-source spans. Overlapping exclusions are retained, including
unbounded observed costs even when an earlier field guard rejects.

Existing gaps remain: 224 interleaved calls lack their own Reason; 746 calls in
373 interleaved records have ambiguous source occurrences. Assessed Choice
continuations lack a unique selected identity in 44 writer, 139 interleaved and
four W calls. These conditions are not repaired using expected answers or newly
generated corrections. Arbitrary native writer meanings remain unverified even
if the packet envelope is valid.

| Supplied-reader inventory | Noul | Choice | Score | Existing legacy metadata passing continuation guard | Direct reader support |
|---|---:|---:|---:|---:|---:|
| Training | 257 | 243 | 250 | 706 /750 | 0 |
| Development R/X9 | 51 | 38 | 31 | 118 /120 | 0 |

Legacy metadata potentials are Noul257/Choice199/Score250 in training and
Noul51/Choice36/Score31 in development. They are not executed/qualified examples.
Direct supplied-reader construction remains unsupported; 408 training and 70
development readers also require unsupported new-profile Reason semantics.
No supplied-return case becomes actual-native-return supervision. The 500
training direct and 1,000 language-replay records retain their separate approved
no-call target potential; 400 held-out F cases remain evaluation only.

Actual eligible gradient exposures and actual native traces in RD-06 are **zero**
because no model work was run. A control-only or zero-semantic-loss path is not a
trainable semantic demonstration. The earlier 0/1,810 receiver-layout finding
also remains: an explicit layout revision and full re-audit are required before
the proposed direct-profile receiver fit.

## Evaluation preparation and its limits

The held-out manifest contains **770 cases**: 470 native W/F, 180 prepared-
receiver X and 120 supplied-reader R/X9. All 100 X change pairs retain their
actual exposures: 90 prepared-receiver pairs and ten supplied-reader pairs;
none is relabeled an actual-native-final pair. Existing misleading/withheld
supplied-return reader cases remain separately labelled.

Explicit native development evaluation uses original W/F prompts, free control,
no gold/effort targets or budget override, and cannot enter training accumulation.
Expected final values/identities, source assets, reviewed continuation hashes and
pair identities are pinned. `native_evaluation` grades separately supplied typed
native-final observations with existing Noul/Choice/Score rules and FP32 interval
endpoints, rejecting component/head/return substitutions and inconsistent typed
fields. It cannot verify that a caller extracted a typed observation correctly
from prose. Existing prose-only continuations supply no structured final-answer
extractor; no parser, semantic review or new judge was invented. Missing final
observations stay unsupported, not passed. Return consumption alone is insufficient.

## Checks and recovery

Clean main `0cd19cd1936b17788facdfa6fe53f8d2bbc680fb` matched authenticated remote
main. RD-01..05 source/test/owner-contract/raw evidence, recorded migrations and
pushed ancestors were verified before edits. Valid unchanged results were reused;
historical evidence/raw receipts remain intact. No matching RD-06 run existed;
sandbox process inspection has limited host visibility. Changed predecessor
sources and intentional frozen-source static-test migration are recorded in
portable evidence; old prepared plans are stale.

- `PYTHONPATH=tests:. python -m unittest tests.test_native_redesign tests.test_episode_producer tests.test_component_admission tests.test_joint_receiver tests.test_direct_role_encoding tests.test_direct_role_packets tests.test_frozen_source tests.test_training_inputs tests.test_receiver_plan tests.test_learning_report`: **86 pass**, 116.312 s.
- After native-output telemetry and typed-field rejection updates,
  `PYTHONPATH=tests:. python -m unittest tests.test_native_redesign`: **10 pass**,
  14.224 s; other unchanged checks reused. Final GPU collection/registration
  guard also passes; no GPU suite was executed or counted as a passing skip.
- Actual full native audit, with model libraries blocked: exit 0. BPE/token/byte/
  position metadata only; no features or neural model loaded. Reader-operation
  counts were reduced from unchanged audited rows without rerunning tokenization.
- Four final `accepted/verified/` CLI preparations, with model imports blocked:
  direct/repeat/frozen-pooled/legacy pass, repeated direct plans byte-identical,
  current source hashes matching and all execution counters zero.
- Compilation including deferred GPU code, staged whitespace, source/artifact
  hashes and local Markdown targets pass. Native tests use literal metadata
  symbols, not CPU tensor/model execution or qualified semantic examples.

Audit-time source bytes were preserved/hash-verified where later telemetry or an
early unsupported-return exit changed a file. The audited `native_plan` and
annotation builders remain unchanged. Session/core registry/runtime/renderer,
loss coefficients, Eligibility, Window/checkpoint and qualified numeric kernels
retain their existing computation. No older receipt is relabeled a new run.
The pre-existing historical profiling input drift remains unchanged.

## Pending RD-07 checks

`RD06-GPU-NATIVE-RETURN` →
`NativeRedesignGPU.test_actual_native_return_and_prefill_source` prepares actual
Noul/Choice/Score native-return traces, one prefill/cache, original source mapping,
real returned scalar/selection and accounting. Guided controls, Assessed status,
one existing continuation token and budget512 are disclosed engineering forcing.

`RD06-GPU-CONNECTED-GRADIENT` →
`NativeRedesignGPU.test_native_loss_through_actual_return_receiver_and_writer`
prepares a return-specific Noul VJP: native loss -> actual returned scalar ->
receiver input -> live writer and R, with receiver weights frozen. This separates
the return derivative from the direct causal-cache route. No optimizer is used.

Both methods in `tests/test_native_redesign_gpu.py` are implemented but **not run**.
They require submitted RD-07, explicit opt-in and exact RTX3090 UUID
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1` before model imports. All eight earlier
IDs also remain pending: ten total. Fixtures certify no semantic continuation,
learned invocation, final-answer correctness or useful improvement.

No agents, new data/reviews, locked access, fitting, deployment, service or
Operating Table changes. Stop at RD-06; next unfinished slice is RD-07 technical
qualification. Future fits remain blocked by layout/support limits and unrun checks.
