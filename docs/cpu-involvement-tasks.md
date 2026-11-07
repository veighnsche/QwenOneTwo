# Task list: substantially reduce CPU involvement

Status: **CPU-01 instrumentation and CPU-02 source/specification work are
implemented and CPU-validated**. [Results](cpu-01-02-results-2026-11-07.md).
The GPU hold was lifted for throughput work on 2026-11-07. CPU-03's initial
hardware runs are measured in the [baseline report](cpu-03-results-2026-11-07.md).
CPU-04's bounded retention policy and initial CPU-08 subset integration are
measured in the [cache report](cpu-04-results-2026-11-07.md).
The subsequent [kernel screens](throughput-kernel-screens-2026-11-07.md) adopt
no kernel. [Resident-pool results](resident-pool-results-2026-11-07.md) add an
opt-in two-process/card mode: higher queue throughput with higher CPU/memory cost.
[Shared-context results](shared-context-results-2026-11-07.md) now test private
module views with shared frozen tensors: memory/setup CPU falls, but measured
CPU/case rises 8.2258%. This remains opt-in; CPU-06 is still open.
The subsequent [larger pure attention block](linear-attention-block-results-2026-11-07.md)
passes scoped numeric checks and reduces CPU/case 3.5924% with 67.94 MiB extra
allocation. This completes CPU-06's first measured pure-region slice; further
regions and broad integration remain open. Normal execution stays unchanged.
No CPU model fallback, agents, new semantic data, training or service changes.

The goal is fewer CPU instructions, launches, synchronizations and transfers per
completed case, with a smaller CPU controller around GPU tensor computation.
Track **CPU seconds per case**, elapsed time, throughput and memory together.
Low CPU utilization alone is not success; waiting longer can also lower it.
No speedup target is assumed before measurement, and no arbitrary time cap is
introduced. Finish each agreed finite experiment and preserve all outcomes.

Every candidate preserves W/reference contents, one prompt/prefill/causal-cache
lineage per episode, latent request/return transport, the continuous startup
score and ordered FP64 S-curve mapping, exact values/bindings, cancellation and
the existing tolerances. Execution changes do not grant semantic acceptance.

## Completed foundations — reuse their evidence

- [x] Remove repeated manifest decoding from per-record admission.
  [CPU measurements](manifest-access-results-2026-11-07.md).
- [x] Replace repeated asset-table reconstruction with an immutable index.
  [CPU measurements](asset-audit-results-2026-11-07.md).
- [x] Fuse same-boundary finite checks without removing checked tensors.
  Predicates improved substantially; complete measured zero-update windows
  improved about 1.8% under their recorded sources.
  [GPU measurements](fused-finite-results-2026-11-07.md).
- [x] Build resident, read-only 3090/3060 worker orchestration and CPU tests.
  Initial hardware execution and subset compatibility/throughput are measured
  in CPU-03/04; full secondary-card transformer qualification remains open.
  [Runner build](two-gpu-diagnostics-v1.md).

These are separate scoped measurements. Do not add their isolated savings or
infer a current full-training speedup. CPU-03 exercises the device-guard changes
on hardware; it does not qualify the entire training path.

## Work order

| ID | Priority | Task | What can be done while GPUs are off? |
|---|---|---|---|
| CPU-01 | First | Add minimal CPU/dispatch/transfer/cache measurements | Implementation, fixtures and static checks |
| CPU-02 | First | Map synchronization boundaries and select capture regions | Source inventory and execution specification |
| CPU-03 | Prerequisite | Establish actual GPU baselines and 3060 compatibility | Prepare existing-case selection and acceptance checks only |
| CPU-04 | High | Reduce recurrence graph recapture | Counter/identity design; GPU evidence required before changing defaults |
| CPU-05 | High | Consolidate audit transfers and frozen-payload hash work | Byte-layout and ownership design/tests; tensor execution deferred |
| CPU-06 | High | Replay larger pure tensor regions | Backend/fixture preparation; capture and parity deferred |
| CPU-07 | Structural | Reduce mandatory host synchronization with GPU failure handling | Specify semantics and failure cases; no behavior change yet |
| CPU-08 | Integration | Measure both GPUs together with accepted changes | Prepare comparison plan; execution deferred |
| CPU-09 | Later | Reduce backward/checkpoint/VJP overhead | Source analysis only; separate diagnostic authorization required |

CPU-01/02 come first. CPU-03 supplies the measurements needed to choose between
CPU-04/05/06; it is not a requirement to implement every candidate. CPU-07 is the
larger architectural option if remaining host checks impose a substantial floor.
CPU-01/02 form the first implementation commit, with evidence; subsequent
implementations remain separate reviewable slices.

## CPU-01 — measure CPU work rather than guessing

- [x] Extend the existing diagnostic runner; do not create another coordinator,
  monitoring service or general benchmark framework.
- [x] Record CPU process time for each worker, parent CPU time, wall time,
  completed case counts and cold/warm phases. Aggregate both workers' CPU time;
  faster completion with more total CPU work is a separate tradeoff.
- [x] Attribute preparation, source/reference encoding, Session execution,
  evaluation/decode, audit readbacks/hashes and reporting where practical.
- [x] Add lightweight recurrence cache hit/miss/capture/eviction counters and
  transfer call/byte counts. Keep counters out of model-facing packets.
- [x] Prepare a fixed profiling subset from existing receipts: Noul, Choice and
  Score, min/max observed Session position counts and an intact intervention pair. Pin IDs,
  record/view hashes, weights, sources and execution profile. Do not run all 310
  cases just to discover which region is expensive.
- [x] Make detailed profiling opt-in and provide an independent baseline plan.
  CPU fixture overhead is recorded separately; actual model profiling overhead
  remains deferred. Do not insert synchronization after every operation or
  label CUDA-event elapsed time as pure kernel compute time.

**Done when:** CPU-only tests validate aggregation, ordering and counter schema;
future GPU runs distinguish launch/dispatch work, host waits, transfers, capture
and device computation sufficiently to select a concrete next target. Profiling
must not alter outputs, ledger events, eligibility or Session counts.

Instrumentation/schema acceptance passes on CPU. Initial GPU attribution,
model-path profiling overhead and exact same-card instrumented receipt agreement
are measured in CPU-03. Stage labels separate setup/startup/main but do not
declare main cases warm; graph recapture is counted. Explicit audit-site byte
counts do not represent every scalar or bus transfer.

## CPU-02 — identify the real capture boundaries

- [x] Inventory hot `.item()`, Python scalar/bool conversions, `.cpu()` transfers,
  per-linear finite hooks, device-policy checks and shape-dependent control.
- [x] For each boundary, identify its purpose: numeric rejection, actual control
  decision, accounting/cancellation, provenance audit or redundant host work.
- [x] Draw the pure tensor regions between required boundaries. Start with
  repeated regions; record shape, dtype, stream, adapter and weight dependencies.
- [x] Specify a new versioned execution profile for any capture beyond the
  existing recurrence exception. Leave the qualified path as the comparison
  oracle. A global `torch.compile` switch is not an implementation plan.

**Done when:** the first candidate region has explicit inputs/outputs, cache-key
and invalidation rules, error/cancellation boundaries, oracle and acceptance
fixtures. No required finite check or semantic decision has been deferred.

See the [synchronization map](cpu-sync-boundaries-v1.md). The first normalization
tail candidate has a versioned specification only; it is disabled/unimplemented.

## CPU-03 — initial hardware baselines measured

- [x] Verify the configured cards and the new device guards. Check targeted
  3090 behavior and actual 3060 model loading, memory fit and numeric behavior.
  Shared startup readouts alone do not qualify the secondary card.
- [x] Run the fixed existing-case baseline on one card at a time, with models
  resident. Separate initialization/capture cost from warm repeated execution.
- [x] Record per-case CPU time, elapsed time, relevant host synchronization/
  launch/transfer counts and peak memory, with actual runtime versions.
- [x] Define the existing forward/cache/gradient and exact-control checks needed
  for each changed operator. Derivative-only checks take no optimizer steps.

**Done when:** subsequent A/B trials can compare the same card, cases, weights
and warm/cold conditions. Failures are recorded; no other card or CPU substitutes
for an unavailable or incompatible GPU. No capability training is launched.

Initial singleton baselines/profile and two-card runs pass; recurrence operator
tests pass on both cards. CPU-04 adds three alternating resident warm pairs per
card, with cold workspace warmups measured separately. Full transformer-state/gradient qualification of the secondary card
is not claimed. Actual subset recaptures, rather than simulated misses, select
CPU-04 as the next candidate.

## CPU-04 — stop unnecessary recurrence graph capture

- [x] Measure actual working-set shapes and cache churn. Astra's previous cache
  estimates were partial-trace simulations, not a current measured miss rate.
- [x] Compare capacity eight with a capacity selected from the observed working
  set. Record capture cost and retained workspace on each eligible card.
- [x] Preserve stream/device/dtype/shape isolation and invocation-specific saved
  inputs. Replayed outputs must own independent state buffers.

**Done when:** capture counts and CPU time fall reproducibly, outputs and
derivatives meet the existing recurrence oracle, accounting stays exact and
memory fit is measured. Do not pad, crop or invent logical positions to obtain
cache hits. Retain the existing policy if the candidate does not help.

Measured on the existing eight-case pack: explicit read-only capacity 32 reduces
wall/CPU on every pair, eliminates warm recapture and costs 51.41 MiB allocated.
Model/training defaults remain eight; full training/context memory is unqualified.

## CPU-05 — fewer audit transfers, identical evidence

- [ ] Consolidate compatible tensor readbacks into fewer transfers. Preserve
  original tensor order, headers, dtype/shape information and exact byte slices.
- [ ] Avoid extra host copies when hashing the transferred bytes. Keep canonical
  serialization and all required provenance/binding coverage unchanged.
- [ ] Consider retaining a hash alongside a frozen reference payload only after
  enforcing private immutable ownership and defensive copies. Never reuse a
  hash for a changed tensor or cache fresh D-dependent source states.
- [ ] Preserve current live-weight identity checks. Tensor versions alone do
  not detect every `.data` mutation and cannot justify hash reuse.

**Done when:** audit output and SHA256 bytes are identical, mutation/admission
regressions pass, and measured transfer/CPU cost falls. Ordinary JSON formatting
was already measured at roughly 0.4–0.5 ms per saved window; revisit it only if
new profiling establishes a larger cost.

## CPU-06 — larger GPU execution blocks

Screened, not adopted: exact RMS replay/pointwise variants have modest isolated
gains; native RMS fails the original strict feedback state/gradient limits.
Those prototype kernels are archived, not enabled; scalar agreement cannot
substitute for their failed state/gradient checks.
The [shared-context experiment](shared-context-results-2026-11-07.md) passes
isolation and exact receipts but also fails to reduce CPU/case. It shares 1,719
frozen tensors across private module views, not causal state across episodes.
The separate [larger attention-block profile](linear-attention-block-v1.md)
preserves the original operations and passes 36 complete numeric traces,
exact reference/input VJP and all receipt checks. Three pairs reduce full-case
CPU/case 3.5924% and direct launches 3.6977%, with unchanged synchronization
counts. It is explicit opt-in; profile/context/training scope stays bounded.

- [x] Prototype one pure tensor region using existing-order CUDA graph replay.
  Use compilation or custom fused kernels as separate candidates if justified.
- [x] Keep Session/registry/ledger mutation outside capture. Respect every
  existing host validation and control boundary in this first version.
- [x] Include all relevant shape, dtype, stream, device, adapter and captured
  weight dependencies in cache identity; define future weight invalidation.
- [x] Preserve BF16 frozen storage, FP32 transformer arithmetic and compensated
  vocabulary products. No implicit fast-math or permanent FP32 base duplicate.
- [ ] Extend to another region only after the first has a measured benefit.

**Done when:** CPU launch work and complete-case CPU time fall, with measured
wall time/memory, original numeric tolerances, connected derivative checks,
immutable prior caches, exact budgets/discrete decisions and P+B+A+H. Cold
capture costs and exclusions remain visible. Reuse unchanged qualification;
broaden checks only where the execution change makes them relevant.

## CPU-07 — the aggressive option: GPU-side failure handling

- [ ] Measure the synchronization floor remaining after accepted prior slices.
- [ ] Specify device-side failure flags and predication that prevent dependent
  computation or decisions after an invalid intermediate value.
- [ ] Define deterministic first-error selection, when the host observes it,
  cancellation behavior, unusable-state handling and ledger ownership.
- [ ] Review the specification against existing immediate rejection contracts
  before changing execution. Checking accumulated flags only at the end is not
  equivalent to the current guards.
- [ ] Implement one isolated region under a separate experimental profile.

**Done when:** injected NaN/Inf/overflow and cancellation failures preserve the
specified first error, prevent invalid downstream effects, retain exact
accounting/cache behavior and show a measured reduction in host round trips.
No blanket removal of per-linear hooks or relaxation of tolerance is permitted.

## CPU-08 — integrate and test two-card throughput

- [x] Establish each accepted candidate against its same-card baseline first;
  comparing 3090 baseline with 3060 candidate confounds hardware and software.
- [x] Run both resident diagnostic workers with complete cases/pairs, canonical
  ordering and read-only final receipts. Measure parent plus worker CPU time.
- [x] Report cases/second, CPU seconds/case, elapsed time and per-card memory,
  including setup/amortization. Check whether the second worker actually helps.

**Done when:** the report distinguishes per-case CPU reductions from parallel
throughput gains, preserves every case/failure and makes no assumed 2× claim.
Existing semantic failures remain failures; speed does not certify capability.

Initial eight-case integration only: 1.600930× main-phase throughput, 1.084579×
end-to-end; two cards use 6.6818% more worker CPU/case and 65.8653% more
setup-inclusive CPU/case. Full/long-run scaling is not established by one pair.

Later, three alternating 24-presentation comparisons of two/four processes on
both cards give **1.381197× main / 1.053164× end-to-end throughput**. Worker
CPU/case rises **32.9089%**; setup-inclusive total CPU/case rises **81.2976%**.
Use explicit `--workers-per-device 2` only when the queue-throughput tradeoff is
desired; default remains one. All same-card receipts/accounting and frozen-weight
checks pass. These gains must not be multiplied by the earlier eight-case results.

## CPU-09 — future training execution, not today's read-only diagnostics

- [ ] If separately authorized, compare activation checkpoint/rematerialization
  policies using derivative-only existing-data probes, with zero updates.
- [ ] Investigate reducing repeated component VJP dispatch only with preserved
  component sums/counts, eligibility, None gradients and failure detection.
- [ ] Treat whole-window retained graphs or altered per-component checking as
  contract changes, not equivalent runtime switches. Measure memory explicitly.

**Done when:** gradient/mask/ownership/accounting agreement and a measured CPU
benefit are recorded. Checkpoint-on/off comparisons do not optimize the current
no-gradient inference path. Training itself still needs separate authorization.

## Next unfinished slice

**CPU-06 per-case host-work reduction:** screened normalization kernels do not
qualify a safe substantial complete-case improvement. Select a larger region or
shared-context approach only after specifying original numeric/error boundaries
and reference/live isolation. Keep the narrower pool default for CPU efficiency.
CPU-05 transfers are not the next measured cost. GPU throughput work is authorized;
training remains unauthorized.

References: [current handoff](next-agent-handoff.md),
[performance audit](receiver-performance-audit-2026-11-07.md),
[recurrence execution contract](codec-recurrence-graph-v1.md),
[numeric profile](numeric-profile-v1.md),
[PyTorch CUDA graph constraints](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs).
