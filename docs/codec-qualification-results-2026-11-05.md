# Full-codec qualification — 2026-11-05

Historical report for the original expanded BF16 arithmetic. The subsequent
[feedback v3 diagnosis/rerun](codec-feedback-results-2026-11-05.md) uses an explicit
precision-profile revision; it does not change or overwrite these failed results.

**Full codec/runtime implemented. Expanded qualification: BLOCKED.**
The 512-position BF16/FP32 state and gradient comparison fails unchanged A3
limits. Component tests, shorter development probes and FP32 reference execution
pass separately. No training, optimizer update, frontend change, deployment or
production checkpoint promotion occurred. All model/encoder/gradient
work ran on RTX 3090 UUID GPU-225509c3-d702-6c27-de03-76b5e99b33b1; no CPU fallback.

See the [implementation](full-codec-implementation.md). The original G1 sources,
parent diagnostic checkpoint, raw reports and [G1-core pass](g1-gate-results-2026-11-05.md)
are preserved. This failure belongs to the larger path, not the original G1 gate.

## Measurements

| Check | Result |
|---|---|
| Exact ABI/control tests | 6 passed; model-free |
| Full-codec GPU component tests | 15 passed |
| Status transport | All 13 cells; assessed zero present versus non-assessed absent |
| Typed values | Six bodies, int64 extrema, decimal scale, UTF-8 including crossed atom boundaries, empty/absent, legal clipping |
| Request boundary | 256 accepted; 257 rejected without publishing sealing objects |
| Choice return boundary | Full selected contents at exactly 64 positions; original bank index retained |
| Simultaneous resource fixture | Request 255 positions/222 handles, position hold 305, completion handle hold 34; final Reason return reaches exactly 256 handles and 12 new records |
| Immutable/history/lifecycle | Version pinning, owned tensors, scalar gradients through history, packet-view mutation isolation, duplicate/late rejection, cancellation, fatal-state termination and cumulative releases passed |
| Startup/native controls | Ordered CUDA budget mapper endpoints, nonfinite rejection, zero-budget same-cache answer, explicit override, cancellation without extra positions passed |
| Modules/export | Block/pooler counts, bounded heads/pointers, unique owners, private-row exclusion and identical encoder save/load output passed |
| Recurrence replay | Original versus temporal-rematerialized layer output/state/input gradients matched exactly |
| Wire-vector artifacts | 18 packet presentations plus initialized encoder state, with exact bit/atom/handle sidecars |
| Initial 128/256 development probes | 24/24 passed across Noul/Choice/Score and four profiles, seed17/one repetition |
| Reviewed 512 mixed-precision probe | Functional/checkpoint/serving match; FP32 comparison FAILED |
| Reviewed FP32 reference path | 9/9 completed: 3 operations at 128/256/512 positions |
| Parameter stability | All live parameters unchanged; 320 frozen base tensors match original values |

All integrations are prescribed engineering traces with an explicit thinking
budget override. They do not certify learned unforced invocation, semantic
interpretation or complete natural answers. Source states are contextual
production-tensor transfers, not clean withheld-fact semantic fixtures.
The initial 128/256 probes retain their source-v1 snapshot; later startup/packet
ownership guards have dedicated regressions and the final reviewed-source runs.
No data review, reference/T training or broader A4 trainer was introduced.

## 512-position numerical failure

| Measurement | Actual | Required |
|---|---:|---:|
| Worst hidden/cache normalized RMS | 0.234006513 | <=.01 |
| Native R gradient normalized RMS | 0.711416997 | <=.05 |
| D gradient normalized RMS | 0.713635652 | <=.05 |
| Wire encoder gradient normalized RMS | 0.712991273 | <=.05 |
| Semantic emitter gradient normalized RMS | 0.712679746 | <=.05 |

The same-profile checkpoint and serving comparisons have zero sampled state
error; checkpoint gradient vectors match exactly. Cross-precision divergence
grows through serial non-assessed Reason feedback. A separate three-part BF16
ablation retained additional input bits but still failed (state error .262063,
gradient errors approximately .95). It is preserved as failed diagnostic evidence,
not adopted as a passing replacement or a reason to widen limits. The remaining
numerical mechanism needs further investigation; cache continuity alone does not
explain or excuse it. The genuine FP32 reference and weights were not altered.

## Reference-path cache, gradients and memory

These complete FP32 traces use one D prefill and one logical hybrid-state lineage,
including serial requests before/after phase closure and explicit historical
result presentation:

| Operation | P | B spent | A spent | H spent | Total | Dispatch/consume | Prefills/lineages |
|---|---:|---:|---:|---:|---:|---:|---:|
| Noul | 27 | 188 | 291 | 6 | 512 | 4/4 | 1/1 |
| Choice | 27 | 345 | 134 | 6 | 512 | 5/5 | 1/1 |
| Score | 27 | 291 | 188 | 6 | 512 | 4/4 | 1/1 |

All leave zero position holds. Native-loss derivatives through returned scalars,
request packets, question slots, writer R and active evaluating D are finite and
nonzero for Noul/Score. Choice delivers differentiable selected contents and
has a connected separate pointer objective; no gradient through argmax is claimed.
Visible filler/termination is prescribed for exact lengths, not a semantic success.

Maximum observed FP32 process allocation is 9.183 GiB, including 446489460
bytes of resident envelopes: two FP32 moment copies and three full-parameter
component-buffer copies. No optimizer is created or stepped. The envelope is a
conservative allocation test, not actual A4 component eligibility or full joint-
training capacity. Existing service allocation is excluded from these process
figures. 1024 positions and maximum context are unqualified.

## Evidence and exit decision

Final self-reviewed-source bundle: `data/codec-qualification/reviewed-v3/`:
`stages.json`, component logs, `vectors/`, `mixed512/report.json`,
`reference/report.json`, source snapshot and manifest. Earlier attempts remain
under `data/codec-qualification/`, including the failed startup-import regression
and precision ablations. Initial shorter results are under
`data/codec-qualification/final/mixed128256/` with matching source-v1 snapshot.
Generated weights/vector artifacts/raw JSON remain ignored local data.

Source hashes in the reviewed mixed and reference reports match the final codec
files. Original G1 source hashes also remain unchanged. The manifest pins hashes
of reports, sources, tests, vector payloads, encoder state and original inputs.

The full three-seed/three-repeat A3 matrix has NOT passed: these are initial
seed-17 development checks and the 512 cross-precision gate already fails.
Collecting more repetitions or passing the FP32 path cannot replace that exit
condition. Further numeric diagnosis is required before marking the expanded
path qualified. Reviewed-data, capability training, frontend and release work
remain unstarted, as requested.

Final mixed-report SHA256: `d2999c9eb675143436d2f867ddbf0b2f4d5f03d47189a0190de564ad1677cdd8`.
Final reference-report SHA256: `aa6e74c1bb68f396afac0009f6ec895aed61ac2391e6ee3385a10c372749a16b`.
