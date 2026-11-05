# Full codec implementation and qualification scope

The user authorized full codec/runtime implementation and the next RTX 3090
qualification checks after G1-core passed, with no training or frontend expansion.
The implementation is isolated in `qwen_one_two/codec/`; the legacy server,
frontend, running services, original G1 code and original checkpoints are unchanged.
Nothing in this document promotes a diagnostic checkpoint to learned capability.

## Implemented runtime surface

| Module | Contract |
|---|---|
| [schema](../qwen_one_two/codec/schema.py) | B1 enums, exact 128-bit layouts, role applicability, presence, atom bytes, strict selected-literal parsing |
| [records](../qwen_one_two/codec/records.py) | Typed values/spans, owned canonical tensors, immutable Ref versions, append-only registry transactions and status/value validation |
| [render](../qwen_one_two/codec/render.py) | Shared request/return/draft/object rendering, deterministic handles, historical translation, complete selected contents and return bounds |
| [neural](../qwen_one_two/codec/neural.py) | CUDA FP32 wire encoders, exact atom helpers, control trunk/reader, semantic emitter, bounded typed heads, nine pointers, status/feasibility and task readouts |
| [writer](../qwen_one_two/codec/writer.py) | Construct/reuse, copied/parsed/constructed attributes, learned span/clipping pointers, bounded seven-event draft updates and typed completion decoding |
| [runtime](../qwen_one_two/codec/runtime.py) | Independent B/C, handle and immutable-record holds; atomic admission/publication; serial lifecycle, terminal/cancellation and one-time consumption |
| [session](../qwen_one_two/codec/session.py) | Single D prefill, initial continuous budget, R/D switching on one causal state, native generation, writer execution, notices and same-phase return |
| [ownership](../qwen_one_two/codec/ownership.py) | Unique positive-LR owners for live parameters; no optimizer update |
| [backbone](../qwen_one_two/codec/backbone.py) | Original functional backbone equations with 32-token recurrent rematerialization; no detached history or cache replay |
| [qualify](../qwen_one_two/codec/qualify.py) | Prescribed 128/256/512 traces, numerical/gradient comparisons, memory envelopes and frozen-parameter checks |

All six typed bodies are implemented: Boolean, signed int64, scale-preserving
decimal, exact reference, source-backed string and constructed UTF-8. Absent
attributes retain their declared kind with no body; false, scalar zero and
present empty UTF-8 remain distinct. Numeric helper arithmetic runs on CUDA,
including the int64-minimum case. Source clipping requires verified original
byte mappings and valid UTF-8 boundaries; invalid representations are rejected.

All thirteen operation/status cells are represented. Non-assessed results have
no scalar/selection and carry a bounded Reason. Assessed Choice returns the
original bank index and complete selected entry. Feasibility is a separate
learned output, not inferred from pointer ranking. No argmax derivative is claimed.

Published tensors own their storage and preserve gradients; read access returns
copies. A new version never changes an old binding. Results and their stored
presentations are published atomically after validation. Historical rendering
translates only actually presented handles, flags and reference bytes, without
expanding omitted fields or detaching continuous values. Nested RETURN_END is
data inside a receiver packet, not a new dispatch/consumption event.

## Admission, accounting and lifecycle

The runtime implements the recorded 256-position request, 64-position return,
16-candidate, eight-evidence, eight-attribute, 32-writer-position, 16-attempt,
32-new-record and 256-local-handle limits. A final Choice bank and snapshot are
published once at sealing. Completion reserves Result plus possible Reason,
all-status/all-candidate return positions and the independent handle envelope.
Reason attachments cannot mint additional Ref-bearing objects or select future
Result/Reason identities as evidence. Resource rejection preserves spent work
and already published records without fabricating a semantic result.

Returns consume exactly once at their existing RETURN_END, with no extra ticks.
Multiple calls keep cumulative spent/released accounting; answering never resets
B. Cancellation and stale/duplicate completion cannot inject feedback. Native
control IDs perform phase closure/termination; roles 37/38/40 are not emitted.
One bounded mechanical notice is allowed per closed failed attempt.

`Session.open` is the complete startup entry point: validate P/A/H, prefill once,
read the actual budget row, map it with the ordered CUDA FP64 S-curve, determine
the grant/explicit override, then deliver detached predicted-score feedback on
the same cache. Zero thinking closes immediately into a normal answer. The
native step uses the learned four-way gate and permits Choice; runtime masks
only exhausted resource allowances. No domain keywords choose operations.

## Numerical implementation and current boundary

The expanded path now uses [feedback precision v3](codec-feedback-precision-v3.md):
BF16 frozen matrix storage, FP32 transformer base/LoRA arithmetic and compensated
BF16 native vocabulary products, with a genuine FP32 reference. This explicitly
supersedes G1's dense-product precision zones for the full codec only. Temporal
rematerialization checkpoints pure
32-token recurrence blocks, retaining connected incoming/outgoing state. A GPU
regression compares outputs, cache states and input gradients exactly with the
original layer equations. Outer-layer checkpointing is separately selectable.
Logical prefills/positions are not increased by backward recomputation.

Initial 512-position serial-Reason tests failed BF16/FP32 limits despite exact
same-profile checkpoint/serving parity. The optional three-part BF16 experiment
also failed and is not an approved precision replacement. No numeric tolerance,
fixture status semantics or reference arithmetic was changed to mark these passed.
FP32 reference-path success must be reported separately from full qualification.
The follow-up [diagnosis and v3 results](codec-feedback-results-2026-11-05.md)
record the explicit precision revision; the old BF16 arithmetic remains a
failed historical profile. The reference, fixtures and thresholds are unchanged.

## Explicit exclusions and evidence types

No optimizer is stepped, no adapter/head is fitted, no dataset is scaled up,
and no frontend/service is changed. Integration fixtures reuse the frozen G1
diagnostic weights where shapes match; newly introduced codec heads remain
initialized/untrained. The reference pooler is implemented and shape-checked,
but reference/T preparation and all capability-training phases remain excluded.

The test vectors are engineering transport fixtures. Their supplied statuses,
exact values and prescribed writing/dispatch are not learned semantic successes.
Real-backbone source states come from the uninterrupted prompt cache and are
production-tensor transfers, not clean withheld-fact semantic tests. Reviewed
semantic/reference data and clean semantic evaluations are later work.

The qualifier records requested axes and incomplete coverage explicitly. Initial
seed-17 development probes cannot certify the required three-seed/three-repeat
A3 matrix. Any failed applicable numeric check blocks broader qualification;
neither successful FP32 execution nor component tests override that failure.
1024-position qualification and full-context operation remain unmeasured.

GPU memory probes may reserve two FP32 moment copies and three full-parameter
component-buffer envelopes without constructing or stepping an optimizer. These
are conservative engineering allocations, not actual A4 eligibility guarantees.
All actual model/encoder/gradient/probe work stays on the selected RTX 3090.

## Reproduce the scoped checks

Use the pinned upstream snapshot and the existing local frozen G1 diagnostic
child; neither is bundled in git. Set their paths explicitly. Qualification
does not call the prototype trainer or update either artifact.

```bash
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export QOT_QUALIFY_SNAPSHOT=/path/to/pinned/Qwen3.5-0.8B/snapshot
.venv/bin/python -m unittest discover -s tests -p test_codec_exact.py -v
.venv/bin/python -m unittest discover -s tests -p test_codec_gpu.py -v
.venv/bin/python -m unittest discover -s tests -p test_codec_precision_gpu.py -v
.venv/bin/python -m qwen_one_two.codec.qualify_matrix \
  --snapshot "$QOT_QUALIFY_SNAPSHOT" \
  --checkpoint data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors \
  --output data/codec-qualification/new-run
```

The default requested matrix includes all three lengths/operations, seeds
17/29/43 and three repeats across functional/checkpoint/serving/FP32 paths.
The matrix runner schedules at most two workers on the selected RTX 3090,
checks every child's source/input/profile evidence and requires all 324 unique
cases before reporting a pass. Use `--workers 1` for serial execution.
The underlying `codec.qualify` entry point also accepts smaller axes.
It stops on a measured failure. Smaller development axes and reference-only
runs record their incomplete coverage explicitly; they cannot report a full
A3 qualification pass. Use a new output directory for every attempt.
