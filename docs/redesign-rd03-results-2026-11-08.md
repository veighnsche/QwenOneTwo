# RD-03 — stable frozen-W offline component sources

RD-03 implements the opt-in `frozen-w-source-v1` source policy. Its model-free
checks pass; all real tensor assertions are explicitly pending RD-07. No model,
feature extraction, optimizer update or fit ran. Delivery and qualification are
separate in the [ledger](redesign-progress.md) and
[portable evidence](evidence/frozen-w-source-v1.json). RD-04 remains unstarted.

RD-01/02 were verified before source edits: implementation/test, raw artifact and
reused-evidence hashes were unchanged, their delivered commits were ancestors of
clean `850a2da`, and remote `main` matched that commit. Valid recorded results were
reused. No unfinished predecessor or matching RD-03 run manifest was found;
sandbox process inspection has limited host visibility. Existing data/native
supervision gaps and the pre-existing historical profiling-test failure remain
as reported by the earlier slices.

## Implemented path and preserved controls

`CleanEncoder` now takes an explicit source policy. Default
`current-d-source-v1` retains the original fresh W+D source encoding and pooled
role payloads; source caching is forbidden for that mutable control.
`ReceiverProducer` can construct `frozen-w-pooled-role-slots-v1` using the new
policy, with optional source caching. Direct role payloads still reject here
pending RD-04. The runner's new-profile execution/qualification guard remains
closed pending later slices.

The expanded backbone has an explicit W forward that returns the existing v3
`feedback_linear` frozen projection without reading either adapter. W mode
requires an explicit offline-source flag and no-grad v3 arithmetic. The source
entry point requires fresh position-zero state/lineage. It does not zero, merge,
replace or retag parameters, change optimizer ownership, or use a substitute
weight dictionary. The reference artifact and its pooled payload algorithm are
unchanged. Normal R/D dense arithmetic after the new branch is structurally
identical to the delivered baseline.

The original `CleanView`, permitted-input/provenance checks, tokenizer span
selection and UTF-8 mapping remain unchanged. Complete excerpts are encoded with
the original 512-token structural ceiling; overlength or ambiguous boundaries
reject rather than crop. Original parent coordinates remain in the external
audit, and renderer-facing offsets address each derived local source. No label,
Reason target or withheld-parent text enters source inputs or cache keys.

This policy is offline component preparation. `Session`, native episode producer,
request binding, `_layer` context restoration and initial/cache state construction
are unchanged. Ordinary R/D forwards retain their original graph and state
transitions; native callers do not supply the offline W flag. No source encoder
was added inside a live job or substituted for actual one-prefill native states.

## Frozen identities and cache behavior

On first explicitly executed construction, the policy verifies all 320 frozen
CUDA W tensors against the pinned snapshot and checks physical tying. Its
fingerprint includes actual W and registered-buffer content, snapshot/configuration
files, live tokenizer backend/template/special-token settings, model configuration,
v3 arithmetic and execution backends, package/CUDA versions and implementation
file hashes, including the actual model module classes. No real fingerprint was
measured in this metadata-only task.

An unchanged resident backbone reuses its verified fingerprint. Parameter/buffer
identity, version, dtype, device, storage and ownership stamps are checked;
tokenizer/configuration/arithmetic drift and fingerprint mutation reject reuse.
The source identity contains no mutable D/receiver/checkpoint hash. Thus ordinary
D updates do not invalidate or redefine frozen-W inputs. Arbitrary untracked
external `.data` writes are outside the supported mutation contract; RD-07 also
checks actual W bytes against the snapshot after its work.

Source cache keys additionally bind the complete permitted bytes, complete
rendered token IDs, selected body-token indices and byte offsets. Only detached
body-state tensors are stored. Entries have tensor hashes and shape/dtype/device
checks; callers receive clones. Each encode publishes fresh immutable local
sources and reconstructs the current parent-coordinate audit. Parent IDs,
hidden bytes, registries and native causal state are never cached as features.
Frozen-profile reference-cache keys bind the same frozen dependencies plus actual
reference identity, role, annotation and slot range; legacy reference-cache
keys and payload computation retain their old behavior.

The consuming codec/source projection and D request computation remain live
outside the no-grad source preparation boundary. That connection is implemented,
not established by a CPU numerical fixture: actual gradient evidence is pending.

## Explicitly deferred GPU checks — owner RD-07

| Check ID | Implemented assertion | Status |
|---|---|---|
| `RD03-GPU-D-INDEPENDENCE` | Uncached source states and pooled role payloads remain exact after a real D perturbation; owner identities/flags and mode context remain intact | pending RD-07 |
| `RD03-GPU-CACHE-PARITY` | Cached/uncached states, pooled controls, bytes, token/parent bindings and hashes match; returned clones cannot mutate cached states; foreign registry refs reject | pending RD-07 |
| `RD03-GPU-CONSUMER-GRADIENTS` | Actual receiver status forward/backward has finite nonzero D and source-projection gradients; W/reference stay frozen/unchanged; one Session prefill/lineage | pending RD-07 |

The methods are registered in `training/frozen_source.py:GPU_CHECKS` and
implemented in `tests/test_frozen_source_gpu.py`. They use an existing reviewed
assessed Noul receiver record and the actual immutable parent/reference artifact;
this is targeted machinery coverage, not all-operation semantic qualification.
No optimizer is constructed or stepped. Collection imports no model library;
execution requires explicit RD-07 opt-in and the exact 3090 UUID before imports.

Only after submitting RD-07, use its approved model environment and recorded local
snapshot with:

```sh
CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 QOT_RD03_GPU_QUALIFY=1 QOT_QUALIFY_SNAPSHOT="$MODEL_SNAPSHOT" python -m unittest tests.test_frozen_source_gpu
```

This command was **not run**. No skipped GPU suite is counted as passing. Existing
v3 certificates identify reused kernels; they do not certify this new W branch,
cache parity or gradients. No tolerance was relaxed and no failed/unsupported
case was relabeled qualified.

## Model-free validation and predecessor migration

```sh
PYTHONPATH=tests:. python -m unittest tests.test_frozen_source tests.test_training_inputs tests.test_receiver_plan tests.test_receiver_learning tests.test_latent_semantic_audit tests.test_learning_report
```

The 50-test suite passes. It covers blocked model imports, explicit offline policy,
v3 admission, complete/fresh cache identities, hidden-input twins, token/UTF-8
coverage and rejection, unchanged native graphs/controls, and exact registration
of the three unrun GPU methods. Scalar/byte metadata fixtures perform no model
forward or fit. Compilation and whitespace pass; exact results, commands, source
hashes and raw logs are in the portable evidence.

Actual runner preparation for the frozen-pooled and legacy profiles also runs
with model libraries blocked. Repeated frozen-pooled plans are deterministic,
all execution counters are zero, and the new plan explicitly lists all three
GPU checks as pending RD-07. All 1,810 receiver training/E/X permitted view hashes
are checked against the unchanged RD-01 inventory during each preparation.

Source-file hashes intentionally change from RD-01/02: their reports and raw
plans remain immutable historical evidence. The migration evidence pins old/new
source hashes and the unchanged `CleanView`, tokenizer, native and ordinary dense
computation bodies. New RD-02 plans bind current source code; stale pre-RD-03
plans/checkpoints are not silently relabeled compatible. Original data/review,
panel and family identities remain unchanged. Raw evidence stays under
`data/redesign-rd03-20261108/accepted/`.

Learning outcome is not applicable. GPU qualification is pending. No agents,
new records/reviews, locked data, CPU model execution, training, deployment,
service or Operating Table changes occurred.
