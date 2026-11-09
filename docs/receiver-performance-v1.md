# Receiver execution performance — 2026-11-09

The user requested better performance after the RD-08 GPU hold. The new explicit
`retained-activations-v1` execution policy is implemented, with model-free checks.
**GPU validation and measured speedup remain pending.** The trainer is still
formerly paused at599 completed updates; it is now absent and500 remain on disk.
No newer checkpoint or matching replacement trainer was found. The disappearance
cause is unknown. This task did not resume, kill, hotpatch or roll back anything.
No new data, RD09 or service change.

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

## Evidence and remaining GPU work

[Portable evidence](evidence/receiver-performance-v1.json) records pending GPU
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
long-run speed or memory at every corpus extreme; affected full v2 qualification
and worst-case memory still need validation before fitting under the policy.

The deferred test command, **only after the user reopens GPU work**, is:

```sh
CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 \
QOT_RECEIVER_PERFORMANCE_GPU=1 QOT_RD05_GPU_QUALIFY=1 \
QOT_QUALIFY_SNAPSHOT=/home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
QOT_PERF_OUTPUT=data/receiver-performance-20261109/paired-windows.jsonl \
PYTHONPATH=. .venv/bin/python -m unittest tests.test_receiver_performance_gpu
```

That command includes exactly12 discarded diagnostic updates/96 existing-data
presentations, not RD-08 continuation. Do not execute it while the GPU hold
remains. Never switch cards or use CPU fallback. No automatic extra trials.
Actual failures, OOM and timing regressions must be retained without promotion.

Admission rejects the candidate before model loading until the three performance
IDs pass with exact source-bound evidence committed/pushed. Existing v2 evidence
also rejects changed sources; it was not relabeled current or silently migrated.
Fresh prepared plans are metadata, not permission or qualification to execute.

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
No automatic restart or claimed599-update recovery is allowed. After GPU
permission returns, resolve the disclosed rollback/execution accounting and a
qualified source/policy transition explicitly. Bind unchanged persisted tensor/
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
