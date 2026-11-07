# Larger pure recurrent attention block — v1

Explicit diagnostic profile `v3/linear-attention-block-v1`. Disabled by default:
`Backbone.attention_block_backend='eager'` preserves the existing path.
`'graph-v1'` selects the new block only with FP32 transformer arithmetic and
the existing recurrence graph profile. No compile, fusion, reassociation,
precision/tolerance change, training or learned-flow certification is implied.

## Region and ordering

`codec.attention_block.eager_block` contains the original token-ordered recurrence
in original 32-position chunks, concatenation, RMS normalization and SiLU gating.
One graph replays the whole supplied sequence (1..1024 positions), replacing
multiple 32-position graph launches and the subsequent host-dispatched tensor
operations. Reductions, epsilon and elementwise operations remain separate
PyTorch operations. Device kernel counts need not fall when host dispatch falls.

Inputs are actual q/k/v, beta, decay, incoming memory, dense gate activation and
normalization weight. Output is normalized/gated y plus final memory. No model,
adapter, reference, registry, ledger or Session is looked up or mutated inside
capture. The norm vector is an explicit input in its original dtype. Dense
projections stay outside, so graph workspace contains no promoted frozen matrix.
The BF16-storage/FP32-transformer/compensated-vocabulary profile remains intact.

The independent `in_proj_z` gate projection runs before recurrence to supply an
explicit graph input. Its CUDA/frozen-weight admission stays outside capture
and happens earlier. The original temporal recurrence and gate arithmetic DAG
is unchanged; full parameter-gradient checks must validate the changed autograd
grouping. There is no transformer finite check between these operations.
All codec finite hooks, Session finite/version/cancellation checks, dense output
projection, control decisions, publication and accounting stay outside at their
existing boundaries. Nonfinite values remain visible to those same guards.

## Replay, derivatives and ownership

- Require explicit selected local cuda:0; model execution targets the 3090 UUID
  in AGENTS.md. FP32 state is the model path; FP64 is only for derivative tests.
  Norm vectors retain their original FP32/BF16 dtype. Unsupported shapes,
  dtypes, backends, epsilon or nested capture reject without fallback.
- Key workspace by device, current stream, every shape/dtype and exact epsilon.
  Inputs become contiguous before replay. Retain eight LRU entries by default;
  the explicit read-only throughput experiment selects 32 and measures memory.
- Copy every actual input each time, including current norm weight and memory.
  No implicit weight-pointer key can reuse another layer or reference mapping.
  Return independent output clones, never expose writable static graph buffers.
- Save invocation inputs through caller hooks. Backward recopies those inputs
  and incoming output/memory adjoints, and replays the original pure PyTorch
  forward/VJP. No stale activations or detached cache transition. Preserve
  unused-input `None` gradients for memory-only losses. First derivatives only;
  higher derivatives and nested capture have no alternate execution path.
- Capture warmups are execution setup and take zero optimizer steps/logical
  positions. A graph eviction can recapture a VJP, preserving saved-input and
  checkpoint semantics. Whole-call backward graphs can retain substantial
  activation workspace; qualification must report memory, not infer trainer fit.
- Runtime state, one prefill/cache lineage, latent request/return bindings,
  initial continuous score and ordered FP64 S-curve are unchanged. No case,
  Session or reference cache is shared with another case.

## Acceptance and finite experiment

1. Exact operator outputs and all first derivatives against the existing
   recurrence-graph plus eager normalization/gating oracle. Cover seeds
   17/29/43, lengths around 32, production heads/dimensions, 512/1024 connected
   positions, single-output gradients, FP64 directional checks, zero/near-zero,
   strided inputs, checkpoint/reuse/eviction/stream isolation, independent
   outputs and nonfinite visibility. No fitting or optimizer step.
2. `scripts/qualify_attention_block.py --execute-gpu` compares existing prescribed
   128/512/1024 Noul/Choice/Score traces: original oracle plus candidate
   functional/checkpoint/serving (36 traces). Preserve original discrete,
   budget/accounting, gradient RMS/cosine and strict FP32 pointwise/RMS limits.
   Initial/final pinned W/tying and diagnostic weights must remain identical,
   with parameter `.grad=None`. These are numeric fixtures, not new training
   data or semantic competence. Reuse unchanged 324/108-case matrix evidence.
3. After numeric acceptance, `scripts/benchmark_attention_block.py --execute-gpu`
   uses the existing eight-case development selection, checkpoint-0188 and
   frozen reference. One resident worker runs three alternating warm pairs
   (original/candidate, candidate/original, original/candidate). Both recurrence
   and candidate caches reset between trials. Frozen reference payloads are
   prewarmed, then explicitly cleared/recomputed under the candidate in an
   additional full eight-case parity pack. Exact receipt/audit agreement is
   required, so the frozen reference path cannot hide behind oracle memoized
   values. Fresh D-dependent source states are never cached.
4. Each trial has a full cold workspace warmup then an eight-case timed pack.
   Profilers/counters are off during timing. All six timed packs finish before
   any profiler starts: in-process profiler overhead can persist after exit.
   Separate cold/warm observation pairs then collect capture counters and a
   trace of the same first case for each backend. Keep all observations:
   8 prewarm + 8 cold-reference parity + 48 warmup + 48 timed + 16 observation
   warmup + 16 observed + 3 startup = 147 presentations, eleven distinct cases.
5. Every receipt passes existing independent admission and matches its original
   same-card baseline exactly. Frozen base/receiver/reference hashes and zero
   updates must pass. Report CPU seconds/case, parent wall/throughput, cold
   capture cost, memory, launch counts and trace overhead separately. A candidate
   that only improves an isolated operation is not a full-case improvement.

Worker timed CPU is the sum of nonoverlapping case intervals; parent timed CPU
includes dispatch, admission and journaling through the final result. Worker
IPC/Queue feeder work outside case intervals is not perfectly attributed to a
pack, so label this metric precisely. Whole-process CPU remains separately
available. Allocator peaks are not physical device peaks, and profiler API/device
durations overlap. Do not add gains to prior reports or apply them to training.

The numeric qualifier uses existing diagnostic fixture weights to preserve its
established connectivity checks. The data benchmark separately loads the pinned
receiver/frozen reference and validates their actual outputs. Neither promotes
weights, changes deployed defaults or grants reference/accelerator acceptance.
The separate frozen-reference test also compares loss/latent-input VJP without
unfreezing its parameters. Combining this profile with shared-thread capture,
secondary-card execution or training requires separate qualification; the old
recurrence cache's replay-only barrier does not seal this new cache.
