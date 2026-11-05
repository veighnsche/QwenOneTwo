# Backend slices — 2026-11-05

This report separates implemented engineering mechanisms from semantic readiness.
The user authorized backend implementation and numeric tests, direct commits and
pushes, but no capability fitting, bulk data, frontend, service or deployment work.
Historical design-only authorization/status paragraphs do not override that task.

## Slice 1: review and 1024 protocol

Acceptance and direct implementation review are in
[the protocol](codec-1024-protocol-v1.md). Two coverage tests and six exact codec
tests pass. The separate 108-case 1024 run now passes; see the
[measured report](codec-1024-results-2026-11-05.md). It did not repeat the
324 earlier cases. No arithmetic/tolerance change was made.

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
At this initial slice, reference outer checkpointing rejected explicitly to
avoid replaying a restored live mapping. Slice 5 below supersedes that guard
with scoped T/base replay and an exact-gradient test. Pure 32-token recurrence
rematerialization remains available. Reference freeze and
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

## Slice 3: loss/mask and optimizer-window mechanisms

Implementation: `training/losses.py`, `engine.py`.

Acceptance: raw-value supervision follows supported reference validity even when
predicted status abstains; incompatible Reason losses mask separately. Exact
int/decimal/UTF8 subtypes, acceptable sets, role-balanced semantic means and native
exclusions follow A4. Eight scheduled examples accumulate separate CUDA component
VJPs using eligible-example denominators, combine/clip once, and preserve inactive
`grad=None`. Failure must not leave a partial optimizer update.

`test_training_engine_gpu.py`: 8/8 pass on the selected RTX 3090 in 0.501 seconds.
The unequal-count gradient oracle is 6.0 (native (2+6)/2 plus structure 0.5*4),
with a separate 0.5 gradient on its other leaf. Inactive parameters/moments with
prior Adam state remain bit-identical. Overlength/incomplete exposures are counted
without backward; zero-eligibility windows advance without updating parameters.
Injected partial-step mutation restores parameters/moments. Loss subtype,
reference-status masks, native exclusions, detached semantic targets, schedule
endpoints and duplicate ownership rejection pass.

These optimizer steps use tiny synthetic CUDA algebra leaves only; no backbone,
adapter, codec capability checkpoint or dataset is fitted. The A4 full-model
eligible-component buffer/moment memory envelope remains a distinct future
measurement; the numeric runner's conservative allocation is not that result.
Phase advancement requires complete scheduled windows and supplied final-two-
checkpoint development evidence; no promotion evidence has been created here.

## Slice 4: versioned checkpoint/export/resume

Implementation: `training/checkpoint.py`; hashes pin base/Qev/protocol/precision/
reference/dataset/source identities. Artifacts contain small trainable masters,
reference weights, codec buffers, effective private rows, named optimizer owners,
CUDA moments, phase counters, sampler state and RNGs. Frozen W is never exported
as a second trainable table. Saving requires a completed window and never
intentionally overwrites an existing artifact. Load validates manifests, hashes,
shapes, dtypes and ownership before installation, with rollback on failure.

Acceptance/evidence:

- `test_training_checkpoint_gpu.py`: 2/2 pass in 0.377 seconds. Exact next-update
  and RNG agreement after resume; sampler/counter restoration; mismatched
  identities, corrupted files and mid-window save rejection.
- `test_training_clean_gpu.py`: now 3/3 pass in 6.601 seconds, including the added
  full-backbone export/load round trip. Forward tensors/private scalar readout
  are bit-identical after restoration; tied base storage and reference/live
  isolation remain intact. The full-model optimizer is constructed but never
  stepped (zero optimizer updates).

These checks do not promote a reference/capability artifact or validate learned
semantics. Callers must bind the expected immutable artifact identities and
supply actual reviewed sampler/data manifests; no fabricated release is created.

## Slice 5: actual packet/session integration and reference preparation

Implementation adds `episodes.py`, `profile.py`, `sampler.py`, `driver.py` and
reference-objective construction. Decision loss eligibility binds to the actual
sealed snapshot, resolves acceptable Choice refs through its bank, retains raw
value loss under predicted abstention, and masks unsupported/divergent outcomes.
Session collection checks one prefill/lineage, exact P+B+A+H, settled holds and
terminal completion before accumulating. Failed exposures retain their IDs;
there is no count-first replay or automatic substitution of an easier example.

The deterministic sampler records catalogue/release identity and cursor, uses the
A4 eight-example/replay recipes and operation/status/domain cycles, and restores
exact order. Guidance takes only a contiguous reviewed prefix and stops at the
first incompatibility. The native driver requires supplied reviewed release/pilot
and numeric prerequisite evidence; it has no CLI or automatic training launch.
Task-specific episode producers still need actual reviewed annotation releases.

Acceptance/evidence:

- 10 input/provenance tests pass after adding revalidation of canonical field
  bindings at clean materialization (forged bytes cannot bypass the role builder).
- 4 model-free sampler/profile tests pass: exact mixture/order/resume, missing
  pools reject without replacement, balanced component cycles and prefix stopping.
- 2 packet-loss GPU tests pass (0.395 seconds): assessed raw loss under abstention,
  changed-snapshot masking, stable candidate refs and partial feasibility labels.
- The combined infrastructure suite passed 16/16 (7.991 seconds) before the last
  two reference checks were added. The affected clean/reference suite then passed
  6/6 (14.407 seconds), including the completed-session buffer collection, frozen
  decoder input gradients, reference pooler/codec/T gradients, export, and exact
  checkpoint off/on outputs/input gradients. Full-model optimizer updates: zero.

Two overlapping reference-objective attempts hit GPU OOM alongside the numeric
worker and pre-existing service allocation. Inspection found an actual auxiliary
memory defect: each reconstruction target retained a new ~485 MiB frozen allowed-
vocabulary slice. Reconstruction now shares one slice per episode, preserving
identical projection arithmetic. The successful six-test run includes the full
objective under the same RTX 3090 allocation; no CPU/3060 fallback, tolerance
change, service interruption or matrix restart occurred.

Reference checkpoint replay now wraps the whole functional T/base mapping, so
recomputation reinstalls that mapping rather than accidentally using live R.
This supersedes the earlier explicit outer-checkpoint rejection in slice 2.
Annotation W remains frozen; T/reference codec/pooler have separate owners and
storage. The reference-objective test is backward-only engineering evidence,
not a reference fitting run, reviewed recovery score or semantic promotion.

## Final integration hardening

Model-aware restore now requires the qualified v3 flags, BF16 frozen tied W,
matching reference freeze state and actual reference tensor fingerprints before
installing a frozen reference bundle. Effective rows are checked against their
masters and rebound to the shared provider. Clean encoding rejects unverified
overlapping/gapped source mappings before GPU encoding; it never invents token
coordinates. The view hash also includes the declared operation.

The final combined GPU run passed **18/18 in 15.063 seconds**. After adding actual
reference-state hashing, the affected six-test suite passed **6/6 in 15.410
seconds**. Raw logs and hashes are listed in the
[portable validation summary](evidence/backend-validation-2026-11-05.json).
The model-free checks are 10 input/provenance + 4 sampler/profile + 2 protocol
coverage, plus the six existing exact codec tests. None executes a model on CPU.
No original 324-case matrix was rerun.

## Slice 6: complete clean packet construction

`training/clean_packets.py` builds clean offline component requests and supplied
reader returns with the existing immutable registry, production renderer and
admission/result checks. It accepts only role-matched, hash-verified clean
artifacts. Grounded exact fields cover Boolean, int64, scale-preserving decimal,
source-string, UTF8 (including quoted empty strings), local reference identities
and separately declared absence. Reference attributes require a neutral local
identity matching the selected source literal; no arbitrary original-registry
reference is imported. Attributes must have been issued by this construction.

The complete request and return wire tensors are identical across hidden-fact
twins. Return construction rejects a duplicate and verifies the Reason/carrier
status. `encode` rebuilds the authoritative packet and records its tensor hash,
codec state hash, construction transformations and external lineage. Source-parent
hashes/original byte coordinates are retained in that external audit and may
differ between hidden twins; they never enter the derived sources/model inputs.

Acceptance evidence: the expanded clean/reference suite passed **8/8 in 17.640
seconds**. After adding the external parent lineage and authoritative packet audit,
the three affected clean tests passed **3/3 in 5.901 seconds**. The tests exercise
all six exact kinds, absence versus empty, local-source rejection and full Noul
request/return equality. Choice/Score construction uses the same codec path;
these new clean tests do not claim reviewed Choice/Score semantic coverage.
This brings the backend GPU regression inventory to 20 unique tests; the earlier
18-test run plus the affected/new suites cover it, without rerunning the old
324-case numeric matrix.


## Slice 7: reject nonfinite decoding before lossy controls

Final code review found that an overflowing finite Score row could produce an
infinite raw readout, then clamp it to a seemingly valid 1.0. Gate logits and
normalization magnitudes also lacked rejection before argmax/normalization.
A CUDA regression reproduced three failures in four checks before the fix.

The codec now validates linear outputs, normalization magnitudes, raw scalar and
pointer products before clamping/selection. Intentional structural -infinity
masks remain downstream of finite-logit validation. A nonfinite native gate
terminates Failed without dispatch; writer/evaluator numeric failures retain the
existing unsafe-failure path. These guards preserve the same finite tensors and
floating operation order. They do not revise v3 arithmetic or numeric tolerances.
The matrix aggregator also now requires explicit tied-base storage evidence.

Acceptance: all **44 focused GPU regressions pass in 22.329 seconds** after the
patch: 18 existing codec/precision, 6 nonfinite-boundary, and 20 backend tests.
Finite linear values/input gradients match their direct PyTorch oracle exactly;
Score, Noul/Choice, gate/pointer and normalization overflow reject before an
apparently valid decision is emitted. Two coverage and six exact codec checks
also pass. Log hashes, including the failing pre-fix run, are in the portable
validation summary.

The 108-case matrix finished before these validation-only guards were applied,
so its source hashes/snapshot are preserved verbatim. The current source manifest
pins the subsequent guards separately. The passing matrix certifies the unchanged
v3 numeric core; the post-guard tests certify the new rejection behavior and
finite-path regressions. No expensive matrix was rerun to overwrite that history.
