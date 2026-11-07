# Direction identity and matched numeric access — 2026-11-07

**Direction hashing is fixed. The frozen numeric-access comparison completed
without a Score improvement; no candidate is promoted.** All existing data,
W/reference, receiver weights, labels and scoring tolerances remain unchanged.

See the [predeclared protocol](numeric-access-comparison-v1.md) and
[portable evidence](evidence/numeric-access-comparison-v1.json). Raw artifacts,
captured tensors and the independent receipt audit are under
`data/numeric-access-20261107/`. There were no failed GPU attempts.

## Slice 1: bind direction to the actual request

Commit `dcc3206` changes the clean input identity to **clean-view-v2**, including
an explicit rubric direction. Direction-only changes now change the hash.
Invalid directions fail, and request construction rejects a renderer direction
different from the hashed view. No packet ABI, tensor arithmetic or numerical
tolerance changes. The null direction is explicit for views without a rubric.

Acceptance: direction sensitivity and rejection tests, existing input/producer/
evaluator checks, then exact replay with both Score directions. **31 model-free
tests passed in 0.139 s.** The later comparison's four control anchors match the
original wire, final h, scalar/status, actual result, transport and entire ledger
exactly. The anchors are the first Noul, Choice, Score and lower-direction Score
cases. Noul/Choice are anchor checks, not fresh full retention qualifications.

Old receipts were not rewritten. The comparison explicitly maps their old
four-field input hashes to v2 hashes using the same full row, role inputs and
source bytes. Other historical launchers may intentionally reject source/hash
drift; their recorded source snapshots remain the authority for reproducing old
runs. There is no implicit legacy-hash admission fallback.

## Slice 2: exactly one matched comparison

Implementation/protocol commit `dbd94c0` preceded execution. The comparison uses
all **40 existing X Score cases /20 pairs**, with frozen parent checkpoint-0188
receiver `94218aae…` and pass-12 reference `5c6015d…`. No training or locked data
is opened. The previously fitted direct affine readout is not used here.

Both new arms preserve all original source-state evidence and add two target
attributes, one for required count and one for completed count:

- Literal control: exact source digits as one UTF-8 atom each.
- Typed candidate: those same source digits strictly parsed as INT64 atoms.

Both add **eight request positions** and use identical keys, order and source
bindings. Keys copy `requirement`, `complete` and `required checkpoints` verbatim
from the existing permitted facts, with byte provenance. No count enters Key
meaning. Two derived Key inputs are encoded by the frozen reference; these are
documented diagnostic transformations, not new authored/reviewed training rows.
No model sees a computed ratio, complement, scaled denominator or desired score.

For all 40 matched cases, all non-attribute wire rows and request event lengths
are **bit-exact** between arms. Source tensors/byte coordinates and original
question/target/rubric reference tensors match historical receipts exactly.
Literal/INT64 atom bytes match their bound source values. Only the declared
attribute format/type/mode/descriptor fields differ. The prepared writer tick
sees each actual draft, so its hidden prefix may differ; caches are never edited.

## Actual results

Every arm uses the original exact FP32 reviewed intervals. Errors below are mean
distance to those intervals. The historical baseline is reused, not a fresh run.

| Input path | Exact head / delivered Score | Correct status | Valid transport | Mean distance |
|---|---:|---:|---:|---:|
| Original source states | 0/40 /0/40 | 40/40 | 40/40 | **0.2338902032** |
| Source states + literal attributes | 0/40 /0/40 | 40/40 | 40/40 | **0.2346320546** |
| Source states + INT64 attributes | 0/40 /0/40 | 40/40 | 40/40 | **0.2348618014** |

INT64 reduces error relative to the matched literal arm on **20 cases**, increases
it on **20**, and ties on none. The mean worsens by **0.0002297468**. This is a
descriptive result on correlated development pairs, not a significance claim.

| Existing formula | Cases | Baseline | Literal | INT64 |
|---|---:|---:|---:|---:|
| Ordinary completion | 20 | 0.269677 | 0.265112 | 0.267719 |
| Remaining fraction | 10 | 0.194187 | 0.196665 | 0.196408 |
| Half-completion | 10 | 0.202020 | 0.211638 | 0.207602 |

All three paths have exactly the same paired direction counts: **5/10** for
reversed rubrics and **0/10** for half-scale changes; **0/20 pairs** have both exact
values correct. Adding typed numbers does not repair rubric-following behavior.

The inputs do affect the receiver: every matched final h differs, with median
relative L2 **0.091647** between literal and INT64. Baseline-to-literal and
baseline-to-INT64 medians are **0.210828 /0.220070**. These displacements are
sensitivity evidence, not proof of numerical or semantic comprehension.

## Checks and exact accounting

- **35 model-free tests pass in 0.236 s**: direction/input/producer/evaluator
  checks plus source-unit/substring, value-blind Key, explicit migration and
  forbidden model-library import tests. Compilation and whitespace checks pass.
- **84 new Sessions /10,950 causal positions**: four control anchors use 498;
  the 40 literal and 40 INT64 cases each use **5,226**. Every Session has one
  prefill, one cache lineage, zero held positions at completion and exact
  `P+B+A+H` conservation. Zero invocation text tokens; actual returns are consumed.
- Reuse the saved baseline and reference pools. Offline preparation executes
  **31 fresh frozen-D source encodings /656 input tokens**, with **222 source
  cache hits**, and **two frozen-reference Key encodings /53 tokens**. There
  are **657 reference cache hits**. Preparation positions remain separate from
  Session positions. Construction audits describe originating tensor lineage;
  these counters identify actual fresh versus reused work.
- Before/after comparisons pass for **all 320 frozen W tensors** and physical
  tying. Receiver/reference fingerprints and captured baseline tensors remain
  unchanged; all model parameters are frozen and all gradients remain None.
  Existing v3 numeric/gradient qualifications are reused; their computation is
  unchanged. This read-only input transform claims no new gradient qualification.
- Independent CPU receipt audit passes: exact source snapshots (**85 files**),
  unchanged original artifacts, old/new identity mapping, all 80 new cases/four
  anchors, atom bytes, matched event lengths, grades, intervals, paired direction
  counts and total positions. Tensor files are hashed as opaque bytes on CPU;
  all tensor/model execution occurred on the specified RTX 3090.
- Measured model-run region **78.289277 s**, peak CUDA allocation
  **3,622,856,704 bytes**. This includes loading/checks, excludes metadata
  preparation and final capture serialization, and is not a throughput benchmark.
- Only original service **PID 2094 /3,852 MiB** remains after exit. The 3060 was
  unused. No agents, new examples/reviews, optimization, fitting, deployment,
  production input-default change or Operating Table/service modification.

## What this resolves, and what remains

The audit-identity defect is fixed. The numeric experiment rejects an immediate
benefit from this exact typed representation with this frozen receiver. It
cannot establish that numbers are absent from h, that typed inputs cannot work,
or that a different receiver could not learn them. Its recorded receiver-training
path did not supply numeric attributes; exact atom representation alone does
not confer learned interpretation. The new keys and numeric path are therefore
also a compatibility test. The existing reference is still semantically unqualified.

**Next unfinished slice:** establish whether the receiver can learn quantity and
operator use with matched training-side exposure, using existing training cases
and an unchanged held-out development test. Separate raw-count access from the
requested scoring operation instead of another frozen format search. Such a
learning experiment needs its own authorization and fixed comparison; none is
started here. Noul transfer, status/value integration, Reason meaning, learned
native invocation and result use remain unresolved.
