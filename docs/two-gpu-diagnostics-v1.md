# Two-GPU diagnostic runner — build evidence, 2026-11-07

**Built and CPU-validated; both GPUs remain off.** No model runtime was imported
by the preparation/test processes, no GPU/status tools ran, and no model weights
were loaded. No agents, new semantic data/reviews, training or service changes.
3060 compatibility, numeric agreement, memory fit and throughput are **NOT RUN**.

## Scope and device policy

The runner executes independent, read-only latent receiver cases on two separate
processes. Each process owns its complete model/cache and selects one UUID as
local `cuda:0`; no Session, prefill, trajectory or cache is split across cards.
Frozen W is replicated per process, with the existing within-process tying and
reference/live isolation. This does not combine VRAM or implement distributed
training. GPU worker processes are ordinary application processes, not agents.

| Target | Explicit UUID | Current hardware validation |
|---|---|---|
| RTX 3090 | `GPU-225509c3-d702-6c27-de03-76b5e99b33b1` | Not queried in this slice |
| RTX 3060 | `GPU-744c1cf5-9578-6aad-95c1-7a6f1c9d11db` | Not queried; model name checked only on future execution |

The secondary UUID comes from the saved two-card inventory in
`data/g1-proof/run-20261105/manifest.json`; that inventory lists UUIDs, not both
model names. The expected secondary name follows the user's specified 3060 and
must match at actual worker startup. A mismatch/unavailable device aborts; there
is no discovery-based substitute or CPU fallback.

`g1_device.py` adds a process-local, explicitly entered read-only diagnostic
scope. Default model execution still requires the 3090 UUID. Both UUIDs together,
ordinal-only selection and unknown UUIDs reject before CUDA queries. Backbone,
codec, configuration and recurrence guards consult this scope. Optimizer
`Window` construction rejects any diagnostic scope, including the 3090 worker.
The worker disables gradients, freezes all model/reference parameters, calls no
optimizer and checks grad=None. Other training/qualification entrypoints remain
3090-only; this is no permission to use the 3060 for training.

Only device guards and the optimizer-entry guard change in existing modules.
Static comparison against `fdfa1d2` verifies **82 unchanged function definitions**
and unchanged transformer construction after the device check, arithmetic flags,
recurrence equations/shapes, and optimizer initialization after the new guard.
BF16 frozen storage, FP32 transformer computation, compensated BF16 vocabulary
projection, exact budget mapping and existing tolerances remain the requested
profile. Original BF16 transformer arithmetic remains failed. No new numeric
matrix or gradient check is claimed from a static comparison.

## Prepared existing-data plan

The CLI separates **prepare** from an explicitly opted-in **run**:

```sh
# CPU-only: reads reviewed data and checkpoint manifest metadata, never weights.
CUDA_VISIBLE_DEVICES='' python3 -m scripts.two_gpu_diagnostics prepare \
  --snapshot /home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
  --output data/two-gpu-runner-20261107/my-prepared-plan
```

The completed local preparation is
`data/two-gpu-runner-20261107/prepared-ready/plan.json`, status
**`PREPARED_GPU_NOT_RUN`**. It binds all sources, reviewed development release,
record and permitted-view hashes, final receiver checkpoint 188, its frozen
reference, expected base/initial diagnostic hashes, and device inventory.
Actual weight-file hash checks are deferred to explicitly executed workers.
The plan contains:

- **130 E cases** as individual jobs.
- **180 X cases / 90 intervention pairs**, each pair staying in one job/device.
- **220 main jobs / 310 distinct existing cases**, ordered for canonical reporting.
- **Three shared startup cases per card**: `development-E-0000` (Noul),
  `development-E-0040` (Choice), `development-E-0090` (Score). These six planned
  presentations are declared repeats of cases already in the 310-case main set.
- The existing 20 supplied-return reader X cases remain explicitly omitted;
  this receiver-only runner cannot qualify native answer consumption.

The source/data pins are recomputed before execution and in workers. The original
latent `evaluate_case` implementation is reused: neutral Session prompt, actual
`Packet` invocation, actual head/result decode and consumption, permitted source
bytes, one prefill/cache lineage, and P+B+A+H receipts. Labels are graded after
evaluation. No human prompt, gold result, annotation gloss or head override is
substituted into live Qev invocation. Only frozen annotation payloads are cached;
D-dependent sources stay fresh.

## Future execution — do not run during the GPU hold

Only after the user explicitly allows GPU execution:

```sh
.venv/bin/python -m scripts.two_gpu_diagnostics run \
  --plan data/two-gpu-runner-20261107/prepared-ready/plan.json \
  --execute-gpu
```

Use the repository virtualenv for actual model execution; the model-free system
Python preparation does not establish that model dependencies are installed.
Omitting `--execute-gpu` rejects before reading the plan or starting a worker.
The child independently requires the execution opt-in before changing visibility
or importing Torch. No GPU execution command was run in this slice.

After both workers load the same pinned weights, they evaluate the three shared
cases. The parent verifies matching model identities/read-only receipts and
compares actual status/selection/Reason presence, grades, transport and complete
control/accounting summaries. Continuous task readouts use the existing **0.002**
whole-path tolerance; budget integers/control decisions must match exactly.
Any mismatch stops before main-case dispatch. This small readout comparison is
**not** full layer/cache/gradient numeric qualification of the 3060.

Once ready, each process receives a new whole job when it finishes the previous
one. Faster completion naturally gets more work, with no assumed 3090:3060 ratio,
new batching rule or time cap. The parent restores canonical case order,
recomputes grades, checks record/view identities, duplicate/missing/foreign jobs,
prefill/lineage, holds, ledger totals and final cache positions. Each worker checks
all 320 W tensors/tying, receiver/reference fingerprints and grad=None before the
startup cases and after main processing. The report records actual device names,
Torch/CUDA versions, task timing and device case counts.

Receipts are flushed to an append-only event journal. Worker errors, early exit,
bad messages, interrupt or failed final integrity stop the run and terminate only
the runner's owned children. There is no automatic retry, device fallback or
service control. Partial receipts are retained as unqualified forensic evidence;
in-flight work may have no final receipt. Existing journals, reports or locks
refuse replay/overwrite. Automatic resume is not implemented.

Completed runs use **`COMPLETED_UNQUALIFIED`**, with the original E/X behavioral
scores reported separately. Infrastructure failures use **`FAILED_UNQUALIFIED`**.
The distinct experimental protocol cannot be loaded as existing
`ReferenceEvidence`; neither outcome grants accelerator qualification, learned
invocation, Reason meaning or native answer-use acceptance.

## CPU acceptance and evidence

**78 unique CPU tests pass across the broad 77-test run and final 11 targeted
tests.** Model-runtime imports are blocked, including in spawned CPU echo
workers. Tests cover explicit selection/scope restoration, unavailable/wrong
device and CPU tensor rejection, diagnostic optimizer prohibition, spawn/dispatch
coverage and canonical reordering, startup rejection, duplicate/foreign results,
worker errors/exits, premature finals, receipt identity/grade/accounting drift,
readout tolerances and exact control mismatches, and execution opt-in refusal.
Existing release/provenance/Reason/sampling/behavior/accounting tests pass.
The echo/readout fixtures are marked TEST_ONLY; they are not GPU measurements.

An added early-exit fixture exposed an unwrapped `BrokenPipeError` from a closed
dispatch channel. Channel send/receive errors now report consistent worker
failures and clean up owned processes. The failed log is retained; the final
targeted suite passes **11/11 in 0.745 s**. The broad run passed **77/77 in
16.830 s** before that error-path refinement.

The new receipt validator also checks **all 310 saved final 3090 E/X results**,
recomputing their identities, grades and accounting on CPU. This reuses old
observations; **zero new GPU or 3060 results** were produced. Two older prepared
plans demonstrate source drift and are rejected by the final source binding.
The final plan, source/AST audit, case/pair coverage, receipt rechecks,
compilation and whitespace checks pass.

Raw evidence: `data/two-gpu-runner-20261107/`. [Portable build evidence](evidence/two-gpu-runner-v1.json)
binds the final plan, source hashes, test/failure logs, source audit and saved
receipt validation. Source snapshots for the prior guard bodies are preserved.

## CPU-01/02 measurement extension

The subsequent [CPU-01/02 slice](cpu-01-02-results-2026-11-07.md) adds process
CPU/wall and dispatch measurements to this runner. `prepare --selection
cpu-profile` selects eight pinned existing cases/seven jobs; the default remains
full coverage. `--profile-host` adds selected Python-call, recurrence-cache and
explicit audit-copy attribution. `--trace-cuda` implies host profiling and, only
on a future explicit run, traces the first main job on each worker. Neither flag
executes models during preparation. They are optional profiling tools, not a
new qualified execution profile. Case result bodies remain unchanged.

Current plans live under `data/cpu-01-02-20261107/{full,cpu-baseline,host-profile,cuda-trace}/`.
Earlier plans/receipts above retain their original source identity and evidence;
fresh preparation is required for this extension. Both GPUs remain off.
See the [synchronization map](cpu-sync-boundaries-v1.md) for remaining boundaries.

**Next unfinished slice:** when the GPU hold is explicitly lifted, validate the
actual 3060 placement, model load/fit and targeted numeric/cache/gradient behavior,
then measure two-worker throughput on existing cases. No such execution or
training is queued now. The Noul transfer and Score failures remain unresolved.
