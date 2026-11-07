# Diagnostic recurrence retention v1

This specifies a **read-only diagnostic cache policy**, not different model
arithmetic or a new training method. It extends the [graph-v1 recurrence](codec-recurrence-graph-v1.md)
with an explicitly selected capacity of **8 or 32 entries** in each dedicated
diagnostic process. The ordinary model/training default remains eight.

`scripts.two_gpu_diagnostics prepare --cache-limit 32` pins the policy in the
source/data/weight-bound plan. Execution still requires `run --execute-gpu`.
The worker sets the capacity and clears old workspace before first model work,
records the actual capacity and keeps optimizer execution prohibited. It never
chooses another card, changes services or silently substitutes a policy after
an error. Prepared plans must be regenerated after source changes.

## Invariants

- The original device/stream/shape/dtype key, LRU order and locking remain.
  Eight is the retained comparison policy; 32 covers the 29 observed keys in
  CPU-03's fixed pack. It does not guarantee every future workload fits.
- No padding, cropping, extra logical positions, reordered recurrence operations,
  fusion, changed reduction, precision/tolerance change or dense-weight capture.
  Original BF16 transformer arithmetic remains failed. BF16 frozen storage,
  FP32 transformer work and compensated vocabulary products remain in force.
- Actual invocation inputs and adjoints are copied before every replay;
  independent outputs/saved inputs remain mandatory. Eviction can recapture
  workspace but cannot invalidate live causal states or another invocation.
- Cache scratch contains recurrence tensors, not promoted frozen base matrices.
  Retained memory is measured per card; this does not qualify training/backward
  memory or arbitrary context lengths. Cancellation, finite rejection and
  Session/registry/ledger mutations stay outside the existing pure capture.
- Configuration is local to the dedicated diagnostic process. All model weights
  and references stay frozen/unchanged; no optimizer is constructed or stepped.

## Finite resident comparison

`scripts.benchmark_recurrence_cache` uses the existing worker and dispatcher;
it does not introduce another coordinator or service. The source-bound plan
lists every job and presentation. `--execute-gpu` is required to execute; without
it, the script only prepares metadata.

For one explicitly selected card, load once and evaluate the same existing
eight-case CPU-03 pack throughout. Prewarm its immutable reference payloads once
under policy eight. D-dependent source states and every Session remain fresh.
Then run six trials, capacities **8, 32 / 32, 8 / 8, 32**:

1. Clear recurrence workspace at the trial boundary and set the requested policy.
2. Run one complete warmup pack with cache counters enabled. Record cold cost,
   bytes and memory; do not include it in the timed comparison.
3. Run one complete timed pack with detailed profiling **and counters disabled**.
   Record synchronized case endpoint CPU/wall intervals, parent interval from
   admitted final warmup receipt to received final timed receipt, and peak memory.
4. For the first trial of each policy only, run another untimed pack with counters
   to observe warm hits/misses/evictions independently of timed measurements.

That is **120 main presentations + 3 startup cases = 123 per card**, 105 jobs;
two cards are tested separately, **246 total presentations**. No agent-selected
elapsed-time cutoff or automatic retry is used. Warmups, reference encodings and
graph setup are real compute and remain visible; they are not extra positions
charged to a case. Every presentation has its own prompt/prefill/cache lineage.

The parent checks every repeated case against the first **exact case receipt**,
including outputs, tensor audit hashes, provenance/bindings, grades and complete
Session accounting. W/reference/receiver identities and grad=None are checked
before/after. The existing GPU recurrence suite is run under capacity 32 against
the retained eager operator oracle and original derivative/stream/eviction guards.
CPU-only fixtures cover explicit admission and disclosed job/phase counts.

Acceptance needs reduced recapture, reproducibly lower same-card CPU and wall
cost, measured workspace fit and passing output/gradient/cache/accounting checks.
Report all three pairs and both memory/latency tradeoffs; do not add speedups to
other historical changes or claim learned capability. If measurements do not
support adoption, retain eight. Hardware results and scope decisions are recorded
in [the measured report](cpu-04-results-2026-11-07.md), separately from this contract.
