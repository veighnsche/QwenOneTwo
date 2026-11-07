# Projection dispatch screen — 2026-11-07

**Reject the tested Script-dispatch candidate and park incremental throughput
work.** Complete-case CPU increases **4.4944%**, throughput falls **4.3178%**, and
memory does not improve. Checkpoint recomputation also fails the required JIT
scope guard. All prototype runtime/test changes have been removed; backend,
scripts and tests are exactly the previously qualified `22e6646` versions.

[Portable evidence](evidence/projection-dispatch-screen-v1.json).
This measures one approach, not a ceiling on all possible projection work.
More invasive dispatch/kernel work remains possible, but this result does not
justify another optimization round ahead of the requested learning work.
No agents, training, data generation/reviews, service changes or time cutoff.
Model work used only RTX 3090 UUID `GPU-225509c3-d702-6c27-de03-76b5e99b33b1`.
W/reference stayed frozen. QwenOneTwo PID 2094 and The Operating Table stayed
unchanged.

## Target and bounded experiment

The saved CPU-08 larger-block Noul trace contains **1,860 FrozenFeedbackLinear
calls** and **5,582 matrix multiplications**. Profiled self CPU is 37.660 ms and
233.500 ms respectively. These include substantial profiler/counter overhead;
they are target-selection evidence, not unprofiled time savings or GPU compute.

The prototype replaces per-operation Python dispatch with two small Script
functions: the two adapter linears/scaling, and frozen-base projection plus
residual addition. JIT executor optimizations are disabled to keep the original
operations separate. Script IR has no fusion group, added `addmm`, or tensor
constant. Actual weights remain invocation inputs. The custom frozen addition
saves the original BF16 W and delegates its input VJP to the existing qualified
FrozenFeedbackLinear backward. No persistent promoted FP32 matrix is introduced.

One resident process compares ordinary and Script projections, with the accepted
attention-block profile and cache capacity 32 identical in both. The existing
eight-case subset, checkpoint-0188 and frozen reference are reused. Cold candidate
reference payloads are explicitly recomputed and checked against oracle receipts.
Three pair orders are ordinary/Script, Script/ordinary, ordinary/Script. Each
trial has warmup then timing. All six timed packs precede any profiling, preserving
the CPU-08 correction. No candidate search, automatic retry, new coordinator or
bulk data run was added.

## Results

| Median metric | Ordinary dispatch | Script dispatch |
|---|---:|---:|
| Worker-case + parent-phase CPU seconds/case | 1.046848 | 1.093898 |
| Summed eight-case worker wall, seconds | 8.418715 | 8.799032 |
| Parent cases/second | 0.940598 | 0.899985 |
| Peak allocated bytes | 2,063,090,688 | 2,063,090,688 |
| Peak reserved bytes | 2,334,130,176 | 2,334,130,176 |

CPU/case worsens in every pair: **4.6955%, 4.5689%, 4.2556%**. Worker wall
increases **4.5175%**. The same-case separate trace records **41,722 direct
kernel launches, 180 graph launches and 792 stream synchronizations in both**.
Script grouping removes no measured launch calls. Recorded device kernel events
are 72,700 / 72,502; profiler event counts and overlapping durations are not an
end-to-end compute estimate. No isolated-operation speedup is promoted.

All **147 presentations**, eleven distinct existing cases and 126 whole jobs,
pass independent receipt admission and match prior CPU-04 3090 receipts exactly.
The 48 timed cases have no profiler/counter fields. Both initial/final W checks
pass all 320 tensors and tied storage; receiver/reference hashes stay pinned,
parameters remain frozen with `.grad=None`, and optimizer updates are zero.
The process exits zero. These are forward-only receiver observations; existing
semantic failures remain failures.

## Numeric rejection and restoration

The GPU operator run has **four test methods, one error**. Eighteen ordinary
shape/seed/rank forward and input/adapter-gradient comparisons finish exactly
before the checkpoint case. The BF16 saved-base identity/no-promoted-base-tape,
IR separation and frozen-weight admission checks pass. During checkpoint
recomputation the guard observes JIT optimizations enabled despite the disabled
outer scope, and rejects the call. The precise internal propagation cause was
not further investigated for this rejected candidate.

The initial fixture scoped forward only. Extending that scope through backward
still yields the same checkpoint error; both runs/logs are retained. The direct
guard was not relaxed and JIT fusion was not enabled to obtain a pass. The long
fixture after that error and the full 36-trace model qualification were **not run**.
This candidate is not numerically qualified for checkpoint training. The
forward-only performance screen still finishes its agreed finite work, without
being relabeled as qualification.

Fixing the scope or trying another interpreter was not pursued after a consistent
full-case slowdown. Installed Torch also warns that `torch.jit.script` is
deprecated; migration/compiler work would add maintenance for a rejected gain.
The complete prototype and modified runner/test sources remain under
`data/cpu-09-20261107/source-v0/`, bound to the measured plan hashes. The active
module, selector branches and tests were removed/restored. Source equivalence
against starting commit `22e6646` passes for all backend/scripts/tests. Prior
numeric qualifications remain applicable; no unchanged expensive matrix rerun.

Raw source snapshots, tracked patch, both failed operator logs, full finite
screen, every timing sample, traces, commands and offline admission audit:
`data/cpu-09-20261107/`. Raw artifacts stay outside Git per repository policy;
portable timings, hashes, failures and restoration evidence are committed.
The directory name is chronological; it does not complete the separate CPU-09
backward/checkpoint overhead task.

## Return priority to Noul, Choice and Score

The existing checkpoint-0188 development E receipts already identify the useful
next work. These are **historical recorded evaluations**, not new results:

| Operation | Passed cases | Valid transport |
|---|---:|---:|
| Noul | 13 / 40 | 40 / 40 |
| Choice | 11 / 50 | 50 / 50 |
| Score | 10 / 40 | 40 / 40 |

Status-correct counts are 13/40, 13/50 and 15/40 respectively. The full source
receipt hash is recorded in the evidence. These forced-receiver component scores
do not establish learned invocation or native answer use.

Prioritize another receiver-learning iteration on the existing training release,
starting from the recorded artifacts, with W/reference frozen and development
E/X held out of updates. Assess Noul/Choice/Score status, values/selected IDs,
bindings and behavioral interventions under the existing semantic criteria.
Keep the accepted performance options available, without multiplying their gains
or blocking learning on further host tuning. No training is authorized or run
by this performance request; actual learning execution is the next task.
