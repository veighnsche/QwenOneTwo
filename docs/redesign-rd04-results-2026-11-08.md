# RD-04 — direct ordered role states, 2026-11-08

Staged implementation acceptance passes; **GPU qualification remains pending
RD-07**. No model execution, extraction, training or optimizer update occurred.
The unchanged limits admit **zero existing receiver rows**, so this initial
layout cannot support the proposed real-data fit. That measured coverage limit
must be resolved explicitly before a fit; it is not a useful-learning result.

## Implementation and contracts

The [layout/admission contract](direct-role-states-v1.md), `codec/layout.py`, four
contract tests and seven owner specifications were committed and pushed first as
`22476193bb517b503e927cd141feddf56e02af7b`, before the implementation edits.
The legacy layout remains the default; opt-in records, immutable snapshots,
packets, metadata and runtime carry `direct-role-states-v1` explicitly.

Question, target, alternative, attribute-key and rubric-subfield payloads retain
ordered complete content-token states, width 1024, FP32, lengths 1..32. Offline
states use frozen W and the verified body mapping of the complete permitted
canonical annotation, excluding template positions. There is no pooling, binning,
truncation or annotation text in a runtime request. Absent rubric midpoint emits
zero positions. The version-2 metadata carries all field lengths; candidate ends
follow the entire candidate including attributes. Exact atom bytes, values, units
and provenance use the existing ABI. The old profile's metadata and renderer
fixtures remain exact.

`Codec.enable_direct_roles()` explicitly constructs one shared FP32 residual
projection, `x + linear(x, delta_weight, delta_bias)`, with zero delta. Raw registry
states stay immutable; the projection runs live before existing wire framing.
The role-conditioned 33-logit length head masks zero for present fields. Native
writer emissions support 32 vectors per field within the existing 32 events,
32 records and 16 attempts. Each new parameter has one owner; matrix/bias groups
retain existing learning-rate/decay conventions. Receiver plans include projection
owners and freeze the writer-only length head. Construction uses zero fills and
does not change the legacy constructor's RNG draws.

The request and every possible return are admitted together under unchanged
**256 request /64 return** bounds. Reservations include full question/target/
rubric echoes, the largest candidate with attributes, and every non-assessed
status's four-slot Reason plus 32 attachment positions. Reason is the existing
post-forward return control, not a compressed direct request annotation.
The old fixed 50/61 reservations apply only to the legacy profile. Structural
admission precedes offline forwards and component Session prefill; runtime sealing
also rejects atomically before adopting a snapshot or dispatching a request.
One prefill/cache, cancellation, continuous budget and numeric kernels remain.

## Complete metadata coverage

The pinned Rust BPE tokenizer and actual Jinja template were used without neural
libraries or feature extraction. All 1,810 receiver rows from the RD-01 full
training/E/X panels are retained in
`data/redesign-rd04-20261108/coverage-2/records.json`, with family/operation,
record/input identities, actual field lengths, mapping hashes, complete observed
position costs and every exclusion. The
[portable coverage manifest](evidence/direct-role-layout-coverage-v1.json) binds
the exact releases, tokenizer files, panel IDs, audit sources and raw artifacts.

| Panel | Rows | Admissible | Q >32 | Alternative >32 | Request >256 | Return >64 | Excluded families |
|---|---:|---:|---:|---:|---:|---:|---:|
| Receiver training | 1,500 | 0 | 1,136 | 86 | 0 | 1,500 | 288 |
| Development E | 130 | 0 | 63 | 7 | 3 | 130 | 91 |
| Development X | 180 | 0 | 155 | 0 | 0 | 180 | 60 |

Field columns count affected rows, not individual fields. Exclusions overlap;
unbounded observed costs are reported even when an earlier field guard rejects.
Training operations remain Noul501 /Choice497 /Score502; E40/50/40 and X100/40/40.
No operation has support. No rows, families or failed attempts were hidden, no
allowance increased, and no new annotations or development training were used.

Metadata token lengths also match all 105 saved source lengths across the
32-record historical probe, with matching record/input hashes. This verifies
tokenization metadata only. It does not compare hidden states or measured quality.
The first coverage artifact is retained; the second adds full observed costs.

New direct-profile preparation binds this zero-support finding and preserves the
complete proposed 1,500-record /four-pass /6,000-presentation /750-window schedule.
It does not authorize execution or select a smaller panel. A real-data fit is
blocked pending an explicit layout revision and coverage re-audit. Later
implementation slices remain unstarted here; their code work does not waive this
fit blocker or any pending GPU checks.

## Validation and predecessor recovery

At clean baseline `483c256c49c3787edbdaa7a129b2b34afa2e540c`, RD-01/02 were verified
through their recorded RD-03 source migration, and RD-03 source/test/raw evidence
was verified. Delivered commits are ancestors. Original evidence and raw receipts
are retained. The new [implementation evidence](evidence/direct-role-states-v1.json)
records every changed predecessor source against that baseline. CleanView,
token mapping, native Session computation methods, producer/binding, checkpoint
machinery, frozen-source implementation and numeric kernels remain unchanged.
Session.open adds only explicit profile validation/runtime selection before
prefill. Its old whole-file static check now compares the unchanged native methods.
All 1,810 existing permitted view identities revalidate during current preparation.
Old prepared plans are stale and are never relabeled current.

- `PYTHONPATH=tests:. python -m unittest tests.test_role_layout tests.test_codec_exact tests.test_direct_role_packets tests.test_direct_role_encoding tests.test_frozen_source tests.test_training_inputs tests.test_receiver_plan tests.test_receiver_learning`: **61 pass**, 83.413 s.
- After adding explicit selected-candidate exact-attribute return assertions,
  `PYTHONPATH=tests:. python -m unittest tests.test_direct_role_packets`: **6 pass**, 0.069 s; other unchanged results reused.
- Four actual CLI preparations, with torch/numpy/safetensors/transformers/peft
  imports blocked: direct, direct-repeat, frozen-pooled and legacy all exit 0.
  Repeated direct plans are byte-identical. All execution counters are zero.
  Exact argv, receipts, hashes and host intervals are in the portable evidence.
- Compilation of changed code and GPU tests, `git diff --check`, dependency/
  artifact verification and local Markdown targets pass. GPU methods were only
  inspected/compiled/collection-imported; no skipped suite is counted as passing.

Static packet tests use explicit symbolic payloads for metadata, accounting,
binding and rendering assertions. They do not bypass GPU-only execution to run
tensor algebra on CPU. They cover all 13 operation/status cells, complete echoes,
variable lengths/keys, exact INT64 bytes through selected return, metadata bits,
largest-candidate ends/reservations, mixed/foreign-profile rejection, immutable
bindings and atomic overlength failure. Legacy renderer signatures match the
pre-RD-04 renderer. An initial symbolic attribute-validation fixture error was
fixed; its raw failed log is retained separately from passing evidence.

The pre-existing historical profiling-test `profiling receipt input drift`
remains as previously documented. No stale receipt was relabeled passing.

## Pending GPU qualification

All six IDs below remain **pending, not executed**, owner RD-07, exact 3090 UUID
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1`; no CPU/3060 fallback.

| ID | Implemented method |
|---|---|
| RD04-GPU-IDENTITY-PROJECTION | `DirectRolesGPU.test_identity_projection_and_legacy_wire` |
| RD04-GPU-DIRECT-ROUNDTRIP | `DirectRolesGPU.test_direct_packets_returns_and_immutable_bindings` |
| RD04-GPU-DIRECT-GRADIENTS | `DirectRolesGPU.test_live_projection_and_consumer_gradients` |
| RD03-GPU-D-INDEPENDENCE | `FrozenSourceGPU.test_frozen_features_ignore_D_updates` |
| RD03-GPU-CACHE-PARITY | `FrozenSourceGPU.test_cache_matches_uncached_and_preserves_bindings` |
| RD03-GPU-CONSUMER-GRADIENTS | `FrozenSourceGPU.test_consumer_gradients_and_frozen_W` |

RD-04 methods in `tests/test_direct_roles_gpu.py` require submitted RD-07, explicit
`QOT_RD04_GPU_QUALIFY=1` and exact device selection before model imports. They
load the existing parent/reference and use declared deterministic engineering
latent tensors to test identity, all-status rendering/immutability, and live
projection/D gradients with a real one-prefill Session. They do not claim those
tensors are direct features from admissible corpus records. Frozen W/reference
are checked; no optimizer is constructed. Existing-data semantic qualification,
Reason meaning and useful improvement remain separate, unmeasured outcomes.

No agents, new data/reviews, locked access, fitting, model workloads, deployment,
service or Operating Table changes. RD-01 supervision/operator gaps remain.
This task stops at RD-04; next unfinished implementation slice is RD-05.
