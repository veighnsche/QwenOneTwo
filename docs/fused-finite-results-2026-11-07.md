# Fused finite checks — 2026-11-07

This slice implements [Astra's first performance recommendation](receiver-performance-audit-2026-11-07.md): aggregate CUDA
finite predicates at existing validation boundaries. It changes no model
equation, data, objective, eligibility rule, summation order or precision profile.
The complete receiver experiment remains finished; this benchmark takes no model
optimizer steps. Semantic failures in its prior report remain failures.

## Implementation and acceptance

`codec/finite.py` uses a read-only multi-tensor infinity-norm reduction for dense
real CUDA tensors, then one host decision. Infinity norms avoid overflow on
large finite values, unlike an L2-norm shortcut. NaNs and either infinity reject
the entire boundary. Empty tensors and None gradients retain their semantics;
CPU tensors are rejected before tensor computation. Other tensor types use GPU
finite flags with one host decision. The checks do not create autograd edges or
mutate values/versions.

`Session.step` still checks the hidden state and every cache tensor before any
following action, at each original boundary. Version, position, lineage and
cancellation guards are preserved. `Window` aggregates loss validation before
any VJP, validates each component's derivatives before buffer/count changes,
validates combined gradients, and checks optimizer state before and after the
transaction. Per-component VJPs, eligible denominators, None gradients, clipping,
backup/rollback and counters retain their previous order. Per-linear hooks and
other early scalar/head checks remain unchanged.

Acceptance requires equal outputs, causal state, losses, accounting, eligible
counts, component accumulation buffers and combined gradients on the same
existing examples; rejection/abort/rollback/cancellation tests; an actual 3090
benchmark; and unchanged W, reference and receiver fingerprints. A passing
slice is numerical/mechanical evidence, not learned-flow qualification.

## Fixed measurement method

`scripts/benchmark_finite.py` loads final receiver checkpoint 188 and its frozen
pass-12 reference. It imports the literal pre-change `engine.py` and `session.py`
saved from main `d767129`, with pinned source hashes. Both paths share identical
weights and the existing training release; all D-dependent source states remain
fresh. Only immutable annotation encodings are prewarmed in both paths. The
original FP32 transformer, compensated BF16 vocabulary projection, activation
checkpointing and eight-entry recurrence graph cache remain in use.

Two fixed eight-record batches reuse historical windows 105 and 111. Those
numbers identify records, not restored optimizer windows or their historical
eligibility: current checkpoint predictions determine the actual masks. Each
batch first runs both paths for parity/warmup, then three alternating timed pairs
(legacy/fused, fused/legacy, legacy/fused), with CUDA synchronization at endpoints.
Timed work includes the actual producer, source/audit transfers, Session forward,
separate component VJPs, eligible accumulation and combined gradients. It omits
optimizer updates, checkpoint export, and untimed validation copies/hashes.

Parity compares head/loss/hidden/cache/saved-boundary **bytes via SHA-256** and
the complete source/accounting reports. It compares all present component-buffer
and combined-gradient values with `torch.equal`, including identical None/key
masks. These are exact value comparisons without a tolerance allowance; signed
zero encodings are not distinguished by `torch.equal`. No multiple episode
graphs are retained. Detached parity copies are discarded before timed runs.

Thirty read-only microbenchmark repetitions follow five warmups per predicate.
They use actual combined gradients and synthetic finite tensors with the actual
terminal hidden/cache shapes. A CPU-only profiler counts host scalar decisions
for these CUDA predicates; this does not profile or execute the model on CPU.
All raw observations are retained. Three pairs on two batches establish local
performance only, not a universal training speedup or an optimizer-step timing.

Raw evidence is under `data/fused-finite-20261107/` (excluded from git); portable
measurements and hashes accompany this report under `docs/evidence/`.

## Parity evidence

Both fixed batches pass. At checkpoint 188 they exercise **17 and 25 component
VJPs**, respectively, covering status, Noul, Choice, Score, feasibility, alignment
and structure across the two batches. There are **4,671 component-buffer tensor
comparisons** and **809 combined-gradient tensor comparisons**, with identical
inactive masks (405/404 active of 414 owned tensors). Every checked gradient
value is exactly equal. Head outputs, hidden states, all cache families, saved
boundary snapshots and loss bytes match. The intentional masked status-logit
negative infinities remain identical; no predicate was added over masked logits.

All 16 distinct existing records have matching source audits, eligibility and
complete Session receipts across the two paths. Each has one prefill, lineage
one, exact P+B+A+H, matching budget/switch/boundary/hold/dispatch/consumption
records, and no text invocation or supplied gold return. Per-batch objective
ledgers total **820 / 1,002 positions**. These request-to-heads episodes end with
no result consumption; they do not qualify result semantics or native answers.

The original 324-case numeric matrix is reused. No model arithmetic or tolerance
changed; this slice instead checks exact outputs and gradient values at the
selected trained checkpoint, plus explicit numeric failure paths. The original
BF16 transformer arithmetic remains failed. This is not a rerun or certification
of the full numeric matrix, 1024-position training, or learned invocation.

## Measured performance and integrity

All three pairs completed on RTX 3090 UUID
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1`, PyTorch `2.15.0+cu132`.
The benchmark exited **0** after **750.440664 s**, including setup, untimed
parity, predicate microbenchmarks and final frozen-weight checks.

| Existing record batch | Legacy median | Fused median | Time reduction | Speedup |
|---|---:|---:|---:|---:|
| Historical window 105, now 17 VJPs | 36.844481 s | 36.166659 s | **1.839684%** | 1.018742× |
| Historical window 111, now 25 VJPs | 51.328583 s | 50.424776 s | **1.760827%** | 1.017924× |

Individual legacy/fused times, in pair order:

- Batch 105: **36.824753 / 36.166659**, **36.894468 / 36.214174**,
  **36.844481 / 36.101506** seconds.
- Batch 111: **51.439502 / 50.424776**, **51.328583 / 50.389339**,
  **51.317641 / 50.431237** seconds.

Pair two ran fused first. Peak allocated tensor memory was identical between
paths: **2,570,116,608 / 2,670,931,968 bytes** for the two batches. Detached parity
copies are excluded from these timed-pass memory measurements.

| Predicate microbenchmark | Legacy median | Fused median | Host scalar decisions |
|---|---:|---:|---:|
| Actual-shaped hidden + 48 caches | 4.535733 ms | 0.608748 ms | **49 → 1** |
| Actual 405 combined gradients | 34.204892 ms | 3.986670 ms | **405 → 1** |

These predicates are **7.4509× / 8.5798×** faster in isolation; the complete
measured windows improve only about **1.8%**. This removes repeated boundary
synchronizations, but does not solve overall iteration speed. Remaining forward,
per-linear checks, separate VJPs, checkpoint rematerialization, recurrence capture
and audit transfers remain in the measured path. Their individual contributions
are not established by this benchmark.

All **320 W tensors** match the pinned base snapshot before/after, with physical
embedding/LM-head tying intact. Reference fingerprint remains
`5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`;
receiver/R/codec fingerprint remains
`94218aae6ec62a7745c349af8e9975fd4987520cf9f1345f38e44586e5e7742c`.
There are **zero model optimizer updates**, empty optimizer states, unchanged
schedule counters and no gradients remaining on any model/reference parameter.
No services, training records, annotations or reviews were changed.

There are **32 parity / 96 timed presentations**, with **14,576 objective-ledger
positions** in total. Offline encodings and rematerializations are separate work.
The independent CPU receipt audit recomputes every timed eligible count and
ledger, checks source hashes against the executed snapshot, compares parity
files, and recomputes the medians. It passes.

## Failure tests and reproduction

**25 unique targeted CUDA tests pass.** The initial run passes 23 tests across
`test_finite_gpu`, `test_training_engine_gpu`, `test_training_episodes_gpu` and
`test_codec_nonfinite_gpu`. Two additional boundary tests and expanded fault
variants then pass. The final six boundary-fixture tests also pass against the
literal legacy classes (**6 legacy + 6 fused, zero failures/errors**), after the
performance run ends. These small fixtures test optimizer mechanics, not model
training; no real-model optimizer call is permitted by the benchmark.

Coverage includes NaN/±Inf, maximum finite BF16/FP32/FP64, subnormals, empty/None,
strided views, complex finite extremes, no input/version/graph mutation, finite
loss with invalid derivative, invalid loss before any VJP, overflowing accumulated
gradients, invalid pre-state, separately corrupted post-step parameters/moments/
device metadata and rollback, inactive ownership, cancellation before/after
charged work, mutated prior cache, position/lineage divergence, and invalid
hidden/cache rejection at the original boundary. Existing early nonlinear/head
overflow guards continue to pass.

To reproduce, extract the two baseline sources from commit `d767129` into a
fresh local directory, then run:

```sh
CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 \
  .venv/bin/python -m scripts.benchmark_finite \
  --snapshot /home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
  --baseline data/fused-finite-20261107/baseline-source \
  --output data/fused-finite-20261107/reproduction
```

The output must be new. The script verifies baseline/checkpoint/base/data hashes
and preserves its source snapshot, exact parity receipts, every timing sample
and integrity report. `fault_equivalence.py`, its test logs, the original test
log, `audit.py` and `audit-result.json` remain alongside the original raw run.
[Portable evidence](evidence/fused-finite-v1.json) binds those files by SHA-256.

**Next unfinished slice:** separately measure activation-checkpoint/cache-policy
effects with gradient/accounting parity before choosing any change. No such
experiment, further training, new data or semantic repair is launched here.
