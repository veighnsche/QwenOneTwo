# Resident diagnostic worker pool v1

This is an opt-in host scheduling extension of the existing
[two-GPU diagnostic runner](two-gpu-diagnostics-v1.md). It changes no model
arithmetic, numerical limits, training method or live service. Each process owns
one complete frozen model/reference and independent whole episodes. Multiple
processes intentionally replicate BF16 model storage; they do not combine VRAM
or introduce another backbone within an episode.

## Selection and ownership

`prepare --workers-per-device 1|2` selects one or two application processes per
selected physical GPU. The default remains **one per card**. With both GPUs,
two per card means four processes. These are application workers, not coding
agents. Only the configured 3090 and explicitly scoped read-only 3060 are valid.
Every child exposes its one physical UUID as local `cuda:0` before importing
Torch. A logical ID such as `<uuid>/1` names a process, not CUDA ordinal 1.

The source-bound plan contains both physical `devices` and logical `workers`.
Admission rejects an unknown worker before changing visibility or importing the
model runtime. Initial/final receipts bind worker ID, PID, physical UUID, actual
card name, frozen W/tying, reference/receiver fingerprints and grad=None. Optimizer
construction remains prohibited by the existing process-wide diagnostic scope.
No CPU/other-device fallback, automatic retry, service control or driver change.

No model/cache/registry or packet travels between workers. Each case retains
one prompt, one prefill and one causal-cache lineage, continuous startup score,
ordered FP64 budget mapping, exact typed values/bindings, cancellation and
P+B+A+H. Whole intervention pairs stay in one job. R/D source encodings remain
fresh; only existing immutable reference-payload caching is reused per process.
The latent request/result path stays inside the selected GPU worker.

## Explicit repeated observations

`prepare --repetitions N` schedules N complete presentations of the selected
case set. N is a positive integer, default **1**; it is not an elapsed-time cap.
Each repetition creates fresh Sessions and is explicitly disclosed. The plan's
`cases` remain unique; `jobs` carry the repetition index. Startup adds three
shared cases per process. Repeated data does not become new independent
qualification evidence or training data.

Admission/report coverage uses `(repetition, case ID)`, with every presentation
exactly once. Same-GPU repeated receipts must be exactly identical, including
source tensor hashes, values, provenance, bindings, grades and accounting.
Shared startup anchors are exact on the same card and retain the existing
cross-card scalar tolerance and exact control checks. No tolerance is relaxed.

Final reports retain original `cases` and E/X summaries for **repetition zero**.
`repeated_cases`, ordered measurements and the journal retain all presentations;
worker and physical-device counts are separate. Parent plus **all** worker CPU
is aggregated, with setup-inclusive and worker-case costs distinguished. Memory
statistics belong to each process; sums of their peaks are upper bounds on
owned allocation, not a sampled physical-GPU peak. Short-queue results do not
qualify full-context memory or long-run scaling.

Main-phase time runs from readiness admission through the last validated main
result. Earlier retiring workers can perform final integrity checks during that
interval. End-to-end includes initialization, all verification and teardown up
to the existing documented reporting cutoffs. Do not present the main interval
as pure device compute or exclusive steady-state work.

## Use

Prepare a new output directory with the existing snapshot and fixed subset:

```sh
.venv/bin/python -m scripts.two_gpu_diagnostics prepare \
  --snapshot /home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
  --selection cpu-profile --cache-limit 32 \
  --workers-per-device 2 --repetitions 3 --output data/my-pool-comparison
.venv/bin/python -m scripts.two_gpu_diagnostics run \
  --plan data/my-pool-comparison/plan.json --execute-gpu
```

Preparation is model-free. Execution requires current GPU authorization and
explicit opt-in. Existing locks/journals refuse overwrite/replay. Errors retain
partial unqualified receipts and terminate only owned children. Trace filenames
include logical worker IDs so same-GPU workers cannot overwrite each other;
simultaneous four-worker profiler support/overhead is not established here.

[Measured results](resident-pool-results-2026-11-07.md) give the CPU/memory cost
and the finite tested scope. One worker per card remains the CPU-efficient
default. No capability training or deployment is authorized by this profile.
