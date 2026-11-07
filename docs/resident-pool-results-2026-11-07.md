# Resident pool throughput results — 2026-11-07

The existing runner now supports **two application processes per GPU**, with
explicit repeated queues and separate logical-worker/device accounting. On the
fixed 24-presentation queue, four processes increase median main throughput
**38.1197%** over two. This uses **32.9089% more worker CPU per case**,
**81.2976% more setup-inclusive CPU per case**, and roughly twice the model
allocation. End-to-end throughput improves **5.3164%** on this short queue.

Defaults stay one process per card and one presentation per case. This is an
opt-in throughput mode, not a reduction in CPU work. No agents, new semantic
data/reviews, training, model/precision changes or live-service changes.

## Selection and acceptance

The [kernel screens](throughput-kernel-screens-2026-11-07.md) found no adopted
arithmetic optimization. Native RMS violated the original strict feedback
state/gradient limits; the exact prototypes had modest isolated gains. The
alternative uses available CPU concurrency with the **unchanged** model path.

An initial same-card probe reuses the existing dispatcher and a 24-case queue:
one versus two 3090 processes takes **29.391528 / 17.943791 s** in the main
phase, about **1.638× throughput**. Worker CPU/case rises
**1.214564 → 1.441011 seconds**. It is one exploratory pair, not the final
two-card acceptance measurement. Every repeated case receipt matches exactly.

The final [pool contract](resident-worker-pool-v1.md) binds each logical worker
to a physical UUID before Torch import. The normal runner is then used for
three alternating pairs, total workers **2,4 / 4,2 / 2,4**. Each run processes
three presentations of the same eight existing cases: six E extrema plus one
intact X pair, checkpoint 188 and the frozen reference. Models stay resident
within each queue. Detailed profiling is off. No chosen wall-clock cutoff.

## Final measured comparisons

| Workers / selected GPUs | Main median | Main cases/s | End-to-end median | Worker CPU/case | Total CPU/case incl. setup |
|---|---:|---:|---:|---:|---:|
| 2 / both | 16.858844 s | 1.423585 | 44.831816 s | 1.333451 s | 3.370115 s |
| 4 / both | 12.205966 s | 1.966252 | 42.568699 s | 1.772276 s | 6.109938 s |


Every pair improves both main and end-to-end throughput. Raw main-phase times:

| Pair/order | Two processes | Four processes | Throughput ratio |
|---|---:|---:|---:|
| 1: 2 then 4 | 16.858844 s | 11.267343 s | 1.496257× |
| 2: 4 then 2 | 15.783745 s | 12.540495 s | 1.258622× |
| 3: 2 then 4 | 17.171592 s | 12.205966 s | 1.406820× |


Initialization and verification dominate this small workload, so queue speed
does not translate to the same end-to-end gain. Parent timing includes result
validation and can overlap earlier workers’ final checks. It is not pure GPU
compute or a stable long-run scaling measurement. CPU process time includes
native threads; worker case CPU excludes parent work, while total CPU/case sums
the parent plus every worker including setup/startup/integrity.

## Memory and correctness

Maximum across runs of the sum of per-process main allocated peaks on each
card (an owned-allocation upper bound, **not a sampled physical-GPU peak**):

| Processes/card | 3090 owned allocation | 3060 owned allocation |
|---|---:|---:|
| One | 2,018,647,552 bytes | 2,011,403,776 bytes |
| Two | 4,030,240,768 bytes | 4,006,466,560 bytes |

Models are explicitly replicated in BF16 per process; no permanent FP32 dense
backbone duplicate is introduced. Setup and reserved-memory observations remain
in the raw receipts. Fit is established for this queue only; worst-case context
and training memory are unqualified. Errors stop owned workers and retain partial
receipts, with no CPU/card fallback.

All **198 final-comparison presentations** pass the independent admission audit:
144 main and 54 declared startup cases. The preliminary probe adds 57; these
are repeated observations, not new training records. Every main case matches
its **same-card CPU-04 baseline receipt exactly**, including audit hashes,
outputs, provenance/bindings, grades and complete Session control/accounting.
Coverage is exact for every `(repetition, case ID)` and for every worker/device.

The final comparisons account for **26,172 objective-ledger positions**.
Offline encodings and capture work remain separate computation. Every episode
has one prefill/lineage and exact P+B+A+H with released holds. GPU W/tying and
reference/receiver fingerprints pass before/after; all parameters stay frozen,
grad=None and optimizer updates zero.

**30 CPU tests pass in 10.791 s**, with model imports blocked, including spawned
echo workers. New coverage checks four-process scheduling, shared-card logical
identities, repeated accounting, duplicate/missing/drifting admission and refusal
before runtime initialization. Static compilation/source/whitespace checks pass.
A final metadata review strengthened exact startup comparison within **every**
physical-GPU group, including the non-reference card. The final admission suite
passes **7/7 in 0.007 s**, and the stricter validator rechecks all six saved
startup sets successfully. Timings bind the pre-strengthening source snapshot;
only `compare_ready` changes afterward. Worker/model/dispatch/measurement bodies
are unchanged. Both source hashes and the AST audit are retained; no GPU rerun
is needed to validate this additional CPU-only receipt check.

All model arithmetic, codec, g1 device/precision, source construction, reference
and case-evaluation code is byte-identical to `c8e9daf`. Existing numeric and
gradient qualification is reused; no unrelated matrix is rerun.

E/X summaries score **the first repetition only**. All repeated receipts and
measurements remain in the report/journal; repeated cases cannot inflate
semantic qualification. Existing Noul/Choice/Score failures are unchanged.
This does not establish learned invocation, native answer use, new accelerator
qualification, fitting or training capability.

## Decision and next unfinished work

Use `--workers-per-device 2` when queue throughput is worth additional CPU and
memory. Keep one per card as the default; nothing grows the pool automatically.
`--repetitions` declares observations explicitly and is not a training pass count.

The remaining challenge is reducing **per-case CPU work** while preserving every
numeric/error boundary. These experiments did not qualify a safe substantial
kernel improvement. CPU-06 remains open; a future larger region/shared-context
design needs exact arithmetic or successful original-limit qualification and
reference/live isolation before implementation. Do not enable native RMS, defer
finite guards, change precision or start training to obtain a speed claim.

All diagnostic workers exited. **QwenOneTwo’s server, PID 2094 under
`qwen-one-two.service`, was left running untouched**, as were other services.
No source/configuration or artifacts in The Operating Table were changed.

Raw evidence: `data/cpu-06-20261107/`.
[Portable pool evidence](evidence/resident-worker-pool-v1.json) binds sources,
plans, every timing/receipt, tests, the independent audit and final GPU inventory.
