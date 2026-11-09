# Receiver execution performance — 2026-11-09

The user reopened GPU work with "GPU is open again." The explicit
`retained-activations-v1` policy now passes actual3090 validation, with unchanged
numeric limits, exact component gradients/updated weights/Adam moments/resume,
and all19 affected-path checks. No new data, RD09, service or fan change.

| Input interface | Original median window | Optimized median window | Wall reduction | CPU reduction | Optimized peak allocation |
|---|---:|---:|---:|---:|---:|
| Pooled control |48.322s|35.763s|25.990%|37.367%|3,323,937,280 bytes|
| Direct candidate |52.837s|39.992s|24.311%|34.766%|4,282,724,864 bytes|

Every one of six alternating timing pairs improves. These are cold eight-
presentation diagnostic windows, restoring parent0188 each time; setup and extra
qualification inspection are excluded. They establish a measured improvement,
not a guarantee of100% GPU utilization or an identical whole-catalogue speedup.
The performance suite passes2 tests in680.715s with12 discarded diagnostic
updates/96 existing presentations; no RD08 training state uses those updates.
Raw report/rows/log/source archive: `data/receiver-performance-20261109/gpu-1/`.

All19 affected-path GPU checks pass with0 optimizer updates, including24 traces
through1024 positions and18 comparisons under the original limits. State RMS is0,
worst logit RMS2.9022e-6, worst gradient RMS1.2337e-6, minimum gradient cosine
0.9999999999993345. Every class cleanup verifies all320 W tensors, physical tying
and the immutable reference. Both interfaces also pass an eight-record window
selected for the largest existing requests, including request256; direct peak
allocation4,486,932,992 bytes (4.18GiB), pooled3,369,452,032 bytes. Raw core report,
steps, metrics, logs and sources are under `data/receiver-performance-20261109/gpu-core/`.

The old trainer disappeared externally with500 saved/599 logged updates. Recovery
preserved exact checkpoint0688 weights, Adam/RNG bytes and sampler state, then
completed250 remaining windows. All99 lost/replayed windows match the original
input/head/loss/update receipts exactly. Control and candidate each completed750 retained updates, all1500×4/6000 effective
presentations, full initial/final panels and then read-only E/X evaluation. Useful
acceptance failed: candidate28 wins/106 losses and all47 historical Choice passes
lost. See [completed comparison](receiver-matched-v2-results-2026-11-08.md). Physical
recorded control calls849 versus750 retained are explicit; any unflushed work
before process loss is unknown. No599-update artifact is fabricated, extra
effective pass or RD09 started. Evidence commit a3edaf7 is pushed/verified.

## Remaining CPU limitation

The measured24–26% window speedup is incremental; it does not establish that the
CPU bottleneck is removed. The user's requirement is that the CPU keep up with
the GPU, rather than applying a CPU-use cap. During candidate459, a2.000s sample
found1.025 CPU-core equivalents used by the trainer and9.30% whole-host busy on12
logical CPUs. One separate instantaneous3090 reading was100%; it cannot establish
sustained utilization or contradict earlier sparse activity. Raw observation:
`data/receiver-performance-20261109/host-observation-candidate-459.json`.
No per-function trace was collected and no new speedup claim follows. Batching,
reducing synchronization and graph capture remain possible engineering work,
not changes made during this source-bound matched fit. No CPU cap was requested
or applied. The finite comparison remains fixed; no extra learning is authorized.

## Changes

The existing runner accepts `--execution-policy retained-activations-v1` in both
preparation and execution. Its prepared contract binds the exact policy, sources,
owners, order, objectives, checkpoint and qualification artifact. Default
`checkpointed-v1` retains the historical execution policy.

- Retain activations for **one episode** instead of recomputing backbone layers
  during each separate component backward. The existing Window still releases
  the episode before constructing the next. Component gradients, individual
  finite checks, eligibility denominators, accumulation order, clipping and
  AdamW settings are unchanged. Eight episode graphs are never retained.
- Use32 recurrence workspace entries instead of8 in the dedicated process.
  Keys, copies, saved inputs, equations, output independence and position
  accounting are unchanged. This retains more scratch memory; no new kernel,
  padding, lower precision or silent OOM fallback is introduced.
- Pack fingerprint byte views on the GPU in groups up to32MiB, then perform one
  host readback per group. An individually larger tensor is copied alone. The
  host hashes the original name/shape/dtype header and exact raw bytes in the
  original order. Every fingerprint still reads current contents; there is no
  version-counter hash cache or skipped mutation check. Default transfers remain
  unchanged outside the explicit policy scope. Scope and cache settings restore
  on errors, with no automatic retry.

No model, optimizer, loss, precision or finite-predicate implementation changed.
The larger existing attention-block backend remains unselected; this is not an
architecture search or a replacement training loop.

## Evidence and GPU validation

[Portable evidence](evidence/receiver-performance-v1.json) records actual passing GPU
IDs and exact sources. Literal-byte tests cover canonical ordering, empty and
oversized buffers, signed-zero/NaN payload bytes, mutations, copy failure and
truncated readbacks. Real checkpoint opaque-byte hashing reproduces all1024
tensors' original hash and the frozen reference artifact hash exactly. At32MiB,
the full bundle uses4 groups rather than1024 individual buffers; the541-tensor
reference uses2. These are byte-group counts, **not measured CUDA transfer
counts, GPU timings or end-to-end acceleration**. Model libraries were blocked;
weights were inspected only as artifact bytes, never loaded/executed on CPU.

`PERF-01-packed-hash-parity` checks actual CUDA mixed dtypes, noncontiguous and
empty tensors, readback counts and a deliberate mutation bypassing versions.
`PERF-02-retained-joint-update-resume` and `PERF-03-matched-throughput-memory`
use the existing reviewed assessed Noul/Choice/Score records and existing
Window/checkpoint machinery, both input interfaces, three alternating pairs of
cold eight-example windows. Each window restores parent0188. All component
gradient bytes, None/eligibility structure, updated masters/moments, frozen
R/startup/nonreceiver owners/reference and transactional resume must match.
All six timing pairs must improve wall time; median CPU time must improve in
each interface. Peak allocated memory is reported. Timing excludes setup and
extra gradient-hash/checkpoint inspection. Cold-window results do not establish
long-run speed or memory for arbitrary future workloads. Affected v2 long traces
and current-corpus maximum-request memory checks also pass as recorded above.

The executed finite validation command (fresh output path on each authorized run) is:

```sh
CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 \
QOT_RECEIVER_PERFORMANCE_GPU=1 QOT_RD05_GPU_QUALIFY=1 \
QOT_QUALIFY_SNAPSHOT=/home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
QOT_PERF_OUTPUT=data/receiver-performance-20261109/paired-windows.jsonl \
PYTHONPATH=. .venv/bin/python -m unittest tests.test_receiver_performance_gpu
```

That command includes exactly12 discarded diagnostic updates/96 existing-data
presentations, not RD-08 continuation. Do not execute it under any later GPU hold.  Never switch cards or use CPU fallback. No automatic extra trials.
Actual failures, OOM and timing regressions must be retained without promotion.

Admission rejects the candidate before model loading until the three performance
IDs pass with exact source-bound evidence committed/pushed. Existing v2 evidence
also rejects changed sources; it was not relabeled current or silently migrated.
Fresh prepared plans still require committed/pushed matching qualification;
preparation alone is not execution or a fitted receiver.

An attempted historical profiling-plan test still rejects old input receipts.
The same `profiling receipt input drift` failure reproduces at pre-change commit
8e63aaa with model imports blocked. That unrelated historical mismatch was not
waived or repaired by rewriting receipts. Focused byte/policy/plan/codec checks
are recorded separately.

## Preserving RD-08

The former paused worker is no longer present. A read-only host process check
found no matching replacement; logs and checkpoint files remain at599 logged /
500 persisted updates. The99 unsaved updates' exact optimizer/RNG state cannot
be reconstructed from logs. All four checkpoint0688 file hashes still verify.
No automatic restart or claimed599-update recovery is allowed. The GPU hold is lifted. Resolve the disclosed rollback/execution accounting and
qualified source/policy transition explicitly before continuing RD08. Bind unchanged persisted tensor/
moment/RNG/sampler bytes and retained observations; never silently repeat valid
saved updates or relabel missing state. Both arms retain their matching rules.
Native learning remains unstarted.

## Model-free preparation results

35 distinct tests pass:34 in110.141 seconds, with21 refreshed checks adding one
peer-policy test. Two GPU classes are skipped. Both final prepared contracts
retain1500 records,6000 presentations and750 windows and reject execution before
model imports with `receiver performance GPU qualification pending PERF-01..03`.
These are prospective contracts, not fit receipts or authorization to restart.

- control: `data/receiver-performance-20261109/plans/control/plan.json`, SHA `907e57f74e5ef41ab8af59a096993de65035e665baf384bf0b209e654a034e69`; contract `09a69345575a332c88b4feeaab61e92cb30483c6fc7e903380735e5c89ebeb7f`.
- candidate: `data/receiver-performance-20261109/plans/candidate/plan.json`, SHA `7398efdcf9cce5fadb885465150104e38a42c4c8a34f21f5edd40cf0b3d81479`; contract `db64f32fccfcf1670fb20ba2947e8d2b53be0287bd2938f4ab960a6c92e431f7`.

## Qualified checkpoint transition

[Recovery evidence](evidence/receiver-performance-recovery-v1.json) pins final
qualified plans and the metadata-only copy of checkpoint0688. Weights, Adam
moments, RNG, complete-window counters and sampler receipts are byte-identical.
Only qualified source/policy/protocol identities change. Manifest SHA
`00928d0150dc6e16055206651a0fb865b2b888ce2b94deca298075e7bd0ed968`.
Retain500 effective windows/cursor4000, execute the remaining250 windows and
preserve the original1500 baseline observations. Original599 logged windows
include99 whose state was lost; those are replayed from the compatible checkpoint.
Final retained learning still uses exactly1500×4/6000/750; report at least849
completed physical optimizer calls including the99 lost/replayed calls, plus
any unflushed pre-loss work whose amount is unknown. Twelve performance diagnostic
updates are separate discarded qualification work. Candidate starts from original
parent0188, never from the control. Both fits finish before development evaluation.
No extra effective pass, data, RD09 or deployment.
