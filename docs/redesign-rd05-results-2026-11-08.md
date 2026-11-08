# RD-05 — joint receiver objectives and explicit binding, 2026-11-08

Staged model-free implementation acceptance passes. GPU qualification is
**pending RD-07**; learning outcome is **not_applicable**. No model execution,
feature extraction, Session, fitting or optimizer update occurred. The initial
direct layout still admits **0/1,500 training, 0/130 E and 0/180 X** receiver rows.
An explicit layout revision and complete re-audit remain necessary before a
direct-profile real-data fit; RD-05 grants no waiver or smaller panel.

## Implementation

The [prepared receiver contract](prepared-receiver-component-v1.md) versions the
experimental path and updates module/optimizer, semantic-supervision, input-rule
and training-profile owners. Existing `ReceiverProducer`, `decision_losses`,
`Eligibility`, `Window`, runtime, registry and checkpoint code are reused.

`component_admission.py` compares the exact reviewed permitted view and independent
clean encoding with the actual sealed request: ordered role/candidate payloads,
rubric metadata, source states/token/byte coordinates, complete evidence spans and
stable candidate Refs. The binding includes the actual Runtime instance and
snapshot/content signature. It rejects foreign/divergent snapshots, unreviewed
attributes/views, source changes, leaked request outcomes, ambiguous identities,
development training and unsupported Reason targets. A matching record ID or
expected answer cannot establish that binding. This is trusted-constructor
provenance, not a semantic oracle or cryptographic attestation of arbitrary data.

Labels bind only after the complete actual forward and revalidation. The receipt
explicitly records experimental component provenance, null reference qualification
and unaccepted receiver behavior. `collect_session` retains its diagnostic-
reference rejection and requires this admission for new-profile gradients.
Failed/cancelled empty components still consume scheduled failed exposures;
they receive no loss credit. Unsupported writer/native semantics remain masked.
Existing native producer and reference-binding code are unchanged.

Status remains eligible regardless of predicted status; reviewed Assessed Noul,
Choice and Score targets remain eligible when the model abstains. Boolean-only
candidate feasibility uses the actual bank; generic unknown annotations do not
invent a fourth-class label. Existing coefficients, eligible-example means,
clipping and optimizer algorithms remain unchanged. No auxiliary count regression
or development-based tuning/sampling was introduced.

**New-profile Reason targets are unsupported and masked.** They are never encoded
or inserted into request features/caches/lengths/routing, and are not zero-filled
or counted as qualified explanations. Existing post-forward Reason/isolated-reader
exceptions and actual-status matching masks remain unchanged for supported old
branches. `Eligibility` was extracted unchanged into a model-free module and
re-exported from `losses`; `decision_losses` itself is unchanged.

W, R, startup row and reference stay frozen. Preparation and actual construction
share exact projection owner specifications; the writer-only length head stays
frozen. `extend_profile_window` restores the exact existing parent first, then
retains old masters, Adam moments, RNG and counters while adding the zero-delta
projection. New moments remain absent until eligible gradients. Transactional
save/resume still uses the existing implementation; none of it was executed here.

`summarize_joint_learning` separates full-panel head predictions from actual
delivered grades, reports operation/status cells, confusion, raw readouts,
eligible counts and Reason masks. Startup retention compares paired raw scalars
and mapped budgets under identical input/record identities and the unchanged
ordered FP64 mapping; overrides and missing data are explicit. A frozen private
row does not protect shared D features. The existing runner's VJP diagnostics
now report actual norms and gradient-bearing parameter counts for every declared
owner group, without another forward, replay, gradient call or update.

## Metadata eligibility and actual execution

`data/redesign-rd05-20261108/objective-inventory.json` retains all 1,500 original
receiver records, families, reviewed identities, per-cell potential objectives,
Reason presence and unchanged direct-layout exclusions.

| Objective | Reviewed potential after exact admission | Actual new-profile eligible examples in RD-05 |
|---|---:|---:|
| Status | 1,500 | 0 — not executed |
| Noul | 256 | 0 — not executed |
| Choice | 158 | 0 — not executed |
| Score | 254 | 0 — not executed |
| Feasibility | 497 | 0 — not executed |
| Reason alignment / structure | 0 — unsupported new-profile targets | 0 |

These are metadata potentials, not model measurements or a waiver of packet
admission. All direct-profile admitted counts remain zero. **832 reviewed Reasons
are retained but not encoded/supervised** by the new profiles. All 13 operation/
reference-status cells remain visible. No actual joint delivered behavior,
startup retention or owner gradient was measured; unit fixtures and pending GPU
fixtures establish no learned accuracy or invocation skill.

## Validation and recovery

Clean baseline `75765409f7a6b2ab94c7bdb38791713477bde5a8` matched authenticated
remote main. RD-01..04 source/test/owner-contract/raw hashes, recorded migrations
and pushed ancestors were verified before edits. Valid unchanged results were
reused. No matching RD-05 job/manifest was found; sandbox process inspection has
limited host visibility. Earlier evidence/raw artifacts remain unchanged. The
[portable evidence](evidence/joint-receiver-objectives-v1.json) records current
source migration, exact commands, receipts, artifacts and test hashes.

- `PYTHONPATH=tests:. python -m unittest tests.test_component_admission tests.test_joint_receiver tests.test_learning_report tests.test_training_inputs tests.test_receiver_learning tests.test_receiver_plan tests.test_training_schedule tests.test_episode_producer tests.test_frozen_source`: **74 pass**, 120.531 s.
- After the failed-empty-exposure guard fix,
  `PYTHONPATH=tests:. python -m unittest tests.test_component_admission tests.test_joint_receiver`:
  **14 pass**, 22.151 s, including the new failure test; unchanged other results
  reused. Together these cover 75 distinct model-free tests.
- Four final actual CLI preparations with torch/numpy/safetensors/transformers/
  peft imports blocked: direct, direct-repeat, frozen-pooled and legacy pass.
  Repeated direct plans are byte-identical; all execution counters zero. They
  retain 1,500 rows /four passes /6,000 presentations /750 proposed windows and
  bind the zero-support blocker. Normal new-profile execution stays gated.
- Compilation including pending GPU methods, `git diff --check`, current source/
  artifact hashes and local Markdown targets pass. GPU files were inspected,
  compiled and collection-imported only; skipped suites are not counted passing.

Model-free packet/loss tests use explicit literal symbols and named loss-dispatch
stand-ins, never CPU tensor algebra or model execution. They assert the real
binding, masks, dispatch, accounting and reporting machinery. They do not claim
their diagnostic symbols are admissible corpus role features. Earlier preparation
receipts are retained as superseded after source changes; only final `verified-final/`
plans bind the delivered implementation. Older prepared plans remain stale.

The 34 unchanged-boundary checks include Eligibility/decision-loss ASTs, original
graders/reports, native producer/binding, CleanView/encoder, Session/runtime,
Window/checkpoint, coefficients, scalar mapping and qualified numeric kernels.
The pre-existing historical profiling receipt input drift remains unchanged;
it was neither rerun nor relabeled passing.

## Deferred checks and scope

All eight checks below are **pending RD-07, never executed**, exact 3090 UUID
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1`; no CPU/3060 model fallback.

| ID | Owner / implemented method |
|---|---|
| RD05-GPU-GRADIENT-ACCUMULATION | `JointReceiverGPU.test_joint_VJPs_and_eligible_accumulation` |
| RD05-GPU-RESUME | `JointReceiverGPU.test_complete_window_resume_and_frozen_owners` |
| RD04-GPU-IDENTITY-PROJECTION | RD-07 / `DirectRolesGPU.test_identity_projection_and_legacy_wire` |
| RD04-GPU-DIRECT-ROUNDTRIP | RD-07 / `DirectRolesGPU.test_direct_packets_returns_and_immutable_bindings` |
| RD04-GPU-DIRECT-GRADIENTS | RD-07 / `DirectRolesGPU.test_live_projection_and_consumer_gradients` |
| RD03-GPU-D-INDEPENDENCE | RD-07 / `FrozenSourceGPU.test_frozen_features_ignore_D_updates` |
| RD03-GPU-CACHE-PARITY | RD-07 / `FrozenSourceGPU.test_cache_matches_uncached_and_preserves_bindings` |
| RD03-GPU-CONSUMER-GRADIENTS | RD-07 / `FrozenSourceGPU.test_consumer_gradients_and_frozen_W` |

RD-05 checks require submitted RD-07 plus explicit opt-in before model imports.
The gradient check uses an existing reviewed frozen-W/pooled receiver component,
actual joint VJPs/eligible accumulation and frozen-owner assertions. Direct corpus
admission remains unsupported; RD-04 separately tests its projection engineering
path. The resume check uses the real parent moments and a declared engineering
perturbation/save/restore, including new projection owners/RNG/counters, without
an optimizer update. Neither claims useful behavior or Reason qualification.

No agents, new data/reviews, locked access, fitting, model workloads, deployment,
service or Operating Table changes. Existing RD-01 supervision/operator gaps
remain. Stop at RD-05; next unfinished implementation slice is RD-06.

Delivery: implementation/evidence `fe90af01540fab0962167f2b4375d07cb169f84b`
was pushed to main and authenticated remote verification returned that exact SHA.
Ledger/handoff delivery finalization follows as a separate commit. All 217 checked
local Markdown targets resolve. Eight GPU checks remain pending; no learning or
GPU qualification result is claimed.
