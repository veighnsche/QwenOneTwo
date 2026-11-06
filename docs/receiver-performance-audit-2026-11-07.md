# Receiver training performance audit — 2026-11-07

The user requested one Astra subagent to investigate CPU bottlenecks and possible
competing instances while the receiver experiment continued. Astra performed a
read-only source/process/GPU audit. It launched no model workload, changed no
sources, settings or services, and did not interrupt training. The main agent
independently recomputed the reported runtime/VJP statistics from saved logs.
This is a performance audit, not new training-data review or qualification.

## Measured observations

Astra found one trainer, PID **37442**, with wrapper **37440**; the deliberately
interrupted original PID 23939 was absent. The trainer had 34 threads and affinity
to CPUs 0–11 on an i7-6850K. Inspected ancestor CPU quotas were unlimited, with
zero throttling. No competing-trainer explanation was observed.

During a 40.6-second sample, the trainer used **124.9% CPU** while the host was
**88.6% idle**. Main-thread CPU was 37.9%, CUDA-autograd worker 61.4%, and five
helpers approximately 5.1% each; other threads were effectively idle. A second
40-second sample gave main/autograd **35.8% / 63.5%**. They alternated: main was
waiting on a futex in 25/40 observations while autograd ran, and the opposite in
15/40. Combined scheduler waiting in that sample was only **3.6 ms**. These
observations favor serialized execution/dispatch over CPU scheduler contention.

Twenty GPU samples ranged **0–40%**, averaging **19.85%**. Subsequent per-process
samples showed trainer activity around 19–37%. Existing server PID 2094 held
**3,852 MiB**, but showed no GPU activity during those samples and negligible CPU
activity. The trainer used approximately 2.8–3.3 GiB. Memory residency is not
proof of compute contention; these samples do not establish that the server is
always idle. No service was stopped or reconfigured.

Windows **76–118** (43 windows, after the bookkeeping correction) averaged
**39.27806458060473 s**, ranging **30.519182003000424–56.35626310900079 s**.
They requested **11–25 separate component VJPs**, averaging **15.86046511627907**.
Runtime/VJP Pearson correlation is **0.9787398403525145**; Astra also reported
0.972 after controlling for source count. Window 105 had 11 VJPs/30.52 s;
window 111 had 25/56.36 s. The parent reproduced the simple statistics from
`continued/training.jsonl`, saved as `performance-timing-audit.json`.

The descriptive line is:

    window seconds ≈ 10.9301 + 1.78734 × component VJPs

This is correlation across different examples, not a component profiler or a
causal benchmark. The intercept is not a separately measured preparation cost.
OS/thread/GPU measurements above are attributed to Astra's audit; raw OS samples
were not separately persisted by the read-only agent. Do not present them as a
second independent measurement by the parent.

## Prioritized implementation candidates

1. **Fuse finite checks at the same existing boundaries.**
   `training/engine.py:119` traverses the shared graph separately per component;
   line 121 then synchronizes for each returned gradient. There are 414 owned
   tensors, including 372 D tensors. `codec/session.py:52` separately checks the
   hidden tensor and 48 cache tensors. Five sampled Session boundaries imply
   approximately 245 state-check synchronizations per episode, before codec and
   loss checks. Aggregate GPU flags and make one host decision at the same
   boundary. Preserve every checked tensor, None gradients, rejection/rollback
   and accounting. Test finite-case output/gradient equality and NaN/Inf failure
   behavior. No speedup has been measured. Defer per-linear-hook changes: moving
   those checks can change when invalid values reach later decisions.
2. **Compare selective/no activation checkpointing for one episode.**
   The launcher enables `b.checkpointing=True`; backbone layers are recomputed
   during backward, potentially once for every component VJP. Current memory
   headroom makes a comparison worthwhile, but does not prove worst-case fit.
   Measure peak memory, component/combined gradients and an optimizer update on
   the same examples. Preserve FP32 operations and fresh D sources. Combining
   this with eight retained graphs is a separate memory problem: frozen BF16
   matrices may have FP32 compute casts retained for input-gradient computation.
3. **Instrument recurrence graph-cache churn.**
   `codec/recurrence.py` retains eight shape entries, while recorded lengths
   exercise all 32 chunk lengths. Astra's simplified known-forward metadata
   simulation produced 952 misses/4,132 requests at capacity eight, versus 32
   misses at 32. These are **simulated partial-trace misses**, not measured cache
   counters; reference passes and backwards were omitted. Measure actual capture
   counts/time/memory before selecting capacity. Increasing retention can preserve
   equations and operation order; do not pad real positions or alter accounting.
4. **Consolidate audit transfers without dropping audit coverage.**
   Receiver fingerprints copy each tensor separately to CPU per example. Astra
   counted 43.29 MB in D, 24.36 MB in codec, with R additionally included. All
   inspected eight-example windows used one unchanged fingerprint internally.
   Batch transfers while retaining the canonical hash bytes/order. Reusing one
   fingerprint per update requires explicit invalidation/immutability guarantees;
   tensor versions alone do not catch every `.data` mutation. Preserve source,
   reference and CleanPackets byte checks rather than deleting them.

## Larger method changes

**Combine the whole-window objective after all actual masks are known.**

    L = sum_c coefficient(c) * sum_eligible_examples L(example,c) / eligible_count(c)

This preserves mathematical eligible-example normalization. It retains eight
complete graphs and conflicts with the current module contract's prohibition on
retained multi-episode graphs. One backward still traverses eight episode graphs;
it does not reduce the work to one episode traversal. Substituting eight into the
descriptive line suggests **25.228757165168645 s/window / 1.5568767150699303×**,
which is illustrative, not a measured speedup. Promising approximately 3× by
substituting one would be incorrect. Summation order changes; bitwise equivalence
is not promised. Checking only combined gradients can also hide nonfinite
individual derivatives through cancellation before a shared Jacobian. A contract
revision and memory, component/combined-gradient, update, failure-injection,
inactive-owner and resume/accounting tests are required before adopting it.

**Batched component VJPs** retain separate denominators and checks more naturally,
but the custom recurrence has fixed CUDA-graph gradient buffers, `copy_` and
batch-one assumptions. Vectorized autograd is not a drop-in speed flag. Validate
per-component gradients, None semantics and memory before benchmarking it.

**Combine alignment and structure where their masks are identical.** In inspected
receiver episodes both occurred together, with weights 1 and 0.5. A derivative of
`alignment + 0.5*structure` can share their common denominator. The fixed sample
would save about 0.86 VJPs/window, roughly 4% under the descriptive line, not a
benchmark. Preserve separate reporting/failure exclusions and resolve the
individual-gradient finite-check requirement. Do not generalize this to other
producers without proving identical eligibility.

Frozen-feature/head-only fitting removes the requested D-learning path. Larger
windows, altered weights or BF16 transformer arithmetic change the experiment;
BF16 transformer arithmetic still fails the preserved numeric qualification.
None is an equivalent acceleration of the running experiment.

## Recommended next slice

Finish the current pass with its current learning implementation. Then start
with fused boundary checks and actual timing/cache counters, followed by a
checkpoint/cache-policy comparison. Consider whole-window gradients only with
an explicit contract revision and measured memory/parity evidence. There is no
measured optimized implementation or achieved speedup in this audit.
