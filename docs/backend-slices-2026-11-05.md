# Backend slices — 2026-11-05

This report separates implemented engineering mechanisms from semantic readiness.
The user authorized backend implementation and numeric tests, direct commits and
pushes, but no capability fitting, bulk data, frontend, service or deployment work.
Historical design-only authorization/status paragraphs do not override that task.

## Slice 1: review and 1024 protocol

Acceptance and direct implementation review are in
[the protocol](codec-1024-protocol-v1.md). Two coverage tests and six exact codec
tests pass. The 108-case 1024 run is in progress; no result is inferred from the
324 earlier passing cases. No arithmetic/tolerance change was made.

## Slice 2: role/branch inputs and clean construction

Implementation: `training/inputs.py`, `clean.py`, `reference.py`.

Acceptance: request serialization excludes current outcomes transitively;
Reason is admitted only in its own non-assessed role; source coordinates/hash/
versions and permitted-view dependencies validate before encoding; fresh GPU
source encoding produces identical artifacts for hidden-only twins, different
artifacts for visible changes, and no full-source registry resolution.

Measured checks:

- `test_training_inputs.py`: 9/9 model-free adversarial tests pass. Outcome swaps,
  transitive label/Reason contamination, branch exposure, grounded attributed
  source outcomes, source bounds/version/hash, family splits, and draft rejection.
- `test_training_clean_gpu.py`: 2/2 pass on the selected RTX 3090, 4.891 seconds.
  Hidden twins match canonical bytes, slot tensors, body states and audit hashes;
  visible evidence changes alter artifacts. Foreign clean-registry refs reject.
  Frozen T reconstruction has finite nonzero input derivatives with no live R/D
  gradients or parameter changes. Reference/live storage is independent.

Reference annotation encoding uses frozen W, while clean source preparation uses
fresh W+D as F3 requires. Separate T parameters share W through scoped functional
calls; no second backbone, data swap or live adapter trainability toggle occurs.
Reference outer layer checkpointing currently rejects explicitly, because replay
of a temporary parameter mapping after it is restored would be incorrect. Pure
32-token recurrence rematerialization remains available. Reference freeze and
live initialization are explicit; actual semantic promotion has not happened.

These are synthetic engineering fixtures and untrained reference weights, not a
reviewed pilot, semantic success or independent review. Mechanical provenance
validates supplied lineage/review artifacts; it cannot establish that prose was
honestly attributed, that humans actually reviewed it, or that labels follow
from the permitted view. Those require the real reviews prescribed in F7/data
operations. No reviewers, approvals or semantic annotations were invented.

Clean preparation constructs fresh source/reference artifacts; caller-built
receiver/reader packets must still use their local registry and immutable
bindings. Full reviewed clean semantic fixtures and F8 coverage remain absent.
