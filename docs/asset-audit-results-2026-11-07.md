# CPU asset lookup and audit serialization — 2026-11-07

The user requested profiling and the next concrete CPU bottleneck fix, with GPU
off, no new data, agents or training. **The slice is complete.** No GPU tool,
model runtime, inference, fitting or training was used. Existing services and
model weights were not touched.

## Profile and selected fix

`ReviewedRelease.asset(path)` rebuilt `dict(self._assets)` for every lookup.
The training release contains **17,809 assets**, while each of the two fixed
eight-record receiver batches needs eight source lookups. Ordinary CPU profiling
and isolated timings on those existing records found:

| Operation, median of 25 initial repetitions | Batch 105 | Batch 111 |
|---|---:|---:|
| Prepare/materialize eight input views | 14.8451 ms | 17.3253 ms |
| Eight asset lookups alone | 7.7561 ms | 7.6895 ms |
| Canonical record audit hashes, eight records | 0.5382 ms | 0.6576 ms |
| Serialize one actual saved window audit entry | 0.3978 ms | 0.4469 ms |
| Serialize, write and flush that entry | 0.3988 ms | 0.4687 ms |

The entries are **55,106 / 62,991 UTF8 bytes**, from historical receiver windows
105/111. Serialization uses the launcher's unchanged `json.dumps(entry)+'\n'`
and its per-window flush on a temporary local file. Record hashing uses the
existing canonical encoder and SHA256. No tensor hashing/device transfers are
included: these cannot be measured with the GPU off.

`cProfile` over 25 eight-view preparations attributes **0.193734 / 0.195555 s**
of self time to 200 asset calls. Profiling overhead affects these totals; they
are separate from unprofiled timings. Asset reconstruction is the larger
measured target, so **audit serialization is unchanged**. Its output, schema,
flush behavior, hash order and audit coverage have not been reduced.

The release now creates one private read-only mapping from the fully validated
assets during snapshot construction. Lookups index that map directly. The
original sorted immutable asset tuple remains intact for existing byte audits;
the index shares those immutable byte objects and does not clone payloads.
This adds one CPU hash table at load time and removes the full-table allocation
on each lookup. No new dependency, mutable result cache or data-loader framework.

## Paired CPU benchmark

`scripts/benchmark_assets.py` loads the literal prior release implementation
from commit `c10c163` and the current one against pinned training/development
releases. Model-runtime imports are blocked and CUDA visibility is empty.
Each batch has five alternating legacy/current pairs, with 25 repetitions per
path/operation in each pair. The table gives medians of the five per-batch means:

| Operation | Batch | Legacy | Current | Saving |
|---|---|---:|---:|---:|
| Prepare/materialize eight views | 105 | 15.269198 ms | 6.943565 ms | **8.325633 ms / 54.5257%** |
| Prepare/materialize eight views | 111 | 17.796844 ms | 9.404333 ms | **8.392512 ms / 47.1573%** |
| Eight asset lookups alone | 105 | 8.108526 ms | 0.925440 µs | |
| Eight asset lookups alone | 111 | 8.057267 ms | 0.924640 µs | |

These are CPU preparation measurements on two fixed batches with preloaded row
objects; normal per-call record comparisons, fresh record decoding, source
validation, role construction and view materialization remain inside the timed
builder. Preparation still constructs the same inputs, not model states.
This saves about **8.3 ms per measured CPU batch**, not seconds of training.
**No full training speedup is measured or claimed.** All observations are kept.

After-change audit serialization medians are **0.380432 / 0.457160 ms**, or
**0.405575 / 0.504792 ms** including write/flush. The JSON byte hashes exactly
match the initial profile. A separate audit profile places most serialization
time in the standard JSON encoder. No serializer optimization was justified by
this comparison. After removing asset reconstruction, recursive provenance and
role/materialization checks account for much of the remaining CPU preparation;
those checks have not been weakened or cached.

## Acceptance evidence

- **All 20,641 asset payloads** (17,809 training + 2,832 development) match the
  baseline immutable snapshots through the new accessor. Full manifest, record,
  asset-tuple and summary bytes also match.
- **110 prior input/provenance cases** reproduce their committed byte hashes,
  covering reference roles, component views and native plans. Their unchanged
  baseline evidence is reused. **16 direct paired component cases** additionally
  compare legacy/current release access in this run, including source coordinates,
  permitted excerpts, role contents, dependencies and indices.
- **52 CPU-only tests pass in 16.179 s** with model-runtime imports blocked. The
  new regression covers all verified fixture assets, empty bytes, Unicode keys,
  immutable/shared payloads and index, unknown/traversing/type-invalid paths,
  no filesystem access during lookup, stale-disk isolation and rejection on
  invalid reload. Existing release/review/lineage, answer-blindness, Reason,
  selection, sampling/resume and 130 E-view checks continue to pass.
- The entire loader and all module code outside `ReviewedRelease` are
  AST-identical to the baseline. Within the class, only the index declaration,
  construction and accessor change. No validation is removed.
- The benchmark exits **0**. Source hashes, byte receipts, all timing medians,
  sample counts, prior-case evidence and serialization hashes pass an independent
  CPU receipt audit. Compilation and whitespace checks pass.

Raw evidence: `data/asset-audit-20261107/`, including the initial profile, raw
`.pstats`, literal baseline source, executed source snapshot, all timing samples,
case/hash results, tests and receipt-audit code. [Portable evidence](evidence/asset-audit-v1.json)
records the measurements and hashes of the full local receipts. Existing data,
reviews and annotations are reused; no semantic judgment is newly claimed.

Reproduce with the baseline `release.py` extracted from `c10c163`, the preserved
initial CPU profile, and a new output directory:

```sh
CUDA_VISIBLE_DEVICES='' python3 -m scripts.benchmark_assets \
  --baseline data/asset-audit-20261107/baseline-source/release.py \
  --profile-before data/asset-audit-20261107/profile-before.json \
  --output data/asset-audit-20261107/reproduction
```

**Remaining:** tensor/device-transfer and full training costs remain unmeasured
under the GPU hold. GPU checkpoint/cache-policy comparisons remain paused. The
existing Noul transfer and Score numerical failures remain unresolved. Further
CPU provenance work would need its own measured change and adversarial parity
checks; none is launched here. No new data or training is queued.
