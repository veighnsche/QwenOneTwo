# CPU-only manifest access improvement — 2026-11-07

The user paused GPU work because of noise and authorized removal of repeated
manifest parsing. This slice runs only ordinary Python data validation,
serialization, tests and timing. **No GPU tools or model workloads ran.** No
weights, services, training data, annotations or reviews changed; no agents or
training were used.

## Change and acceptance

`ReviewedRelease` stores immutable `purpose` and `selected_ids` fields derived
from the fully validated loader result. The latter is a frozenset for membership
checks; it does not define sampling order. Component/reference input builders,
native-plan/episode admission, supervision admission and receiver-purpose checks
use these fields instead of repeatedly decoding the complete manifest.

The loader's validation body is AST-identical to the prior version, excluding
the final constructor call that now supplies the validated selected set. Hashes,
reviews, approval binding, exclusions, lineage, splits, sources and record-byte
checks remain enforced. Admission guards retain their order/error messages.
Canonical record comparisons, role whitelists, answer blindness and the separate
Reason exception remain unchanged. Input construction after admission and model
execution arithmetic are unchanged.

The public `manifest`, `records`, `record` and `summary` APIs still return fresh
JSON copies. Returned-copy mutations and later disk edits cannot change the
snapshot or admission index. Catalogue/sampling iteration continues to use the
original manifest order, including the receiver's deterministic four-record
wrap. Startup catalogue reads remain; this does not cache a mutable manifest.

Acceptance requires identical validated snapshot bytes, purpose/selection
membership, model-input bytes and provenance, rejection/mutation regressions,
and CPU-only timing against literal prior sources.

## Measured results

`scripts/benchmark_manifest.py` compares literal sources from commit `8459e84`
against the new implementation. It loads only the existing training/development
releases; locked labels stay unopened. CUDA visibility is empty and imports of
Torch, Transformers, Safetensors, Triton and CuPy are explicitly forbidden.
No such runtime was loaded.

Both versions yield identical complete immutable snapshots:

| Release | Manifest bytes | Records | Assets | Selected |
|---|---:|---:|---:|---:|
| Training | 4,072,795 | 9,728 | 17,809 | 8,000 |
| Development | 670,388 | 1,600 | 2,832 | 1,000 |

All manifest, canonical record, asset and summary bytes match. The index agrees
with every manifest selection descriptor. **110 existing representative input
cases pass exact normalized-byte/provenance parity**: 52 reference-role inputs,
46 component views and 12 native plans, covering 71 training and 39 development
strata. Comparisons include permitted content/excerpt byte hashes, reference hashes,
dependencies, binding coordinates, provenance nodes/sources, indices and native
plan fields. This checks inputs, not model tensors or semantic competence.

Five alternating pairs measure preparation and materialization of the same eight
existing receiver records from historical window 105, after one warmup per path.
Record bytes are preloaded identically; normal per-call record/snapshot comparisons
and source lookups remain inside the measured builder. All observations:

| Pair | Legacy seconds | Current seconds | Execution order |
|---|---:|---:|---|
| 1 | 0.904687 | 0.015810 | Legacy first |
| 2 | 0.909318 | 0.014614 | Current first |
| 3 | 0.908300 | 0.015923 | Legacy first |
| 4 | 0.913464 | 0.014751 | Current first |
| 5 | 0.910475 | 0.015866 | Legacy first |
| **Median** | **0.909318** | **0.015810** | |

The measured CPU preparation saving is **0.893508 seconds per eight views**.
Instrumented full-manifest property reads fall **16→0** for that batch. The CPU
view-preparation path is 57.51× faster; **this is not a measured training
speedup**. Model execution, backwards, optimizer work and full training timings
were not run. The prior audit's estimated one-second overhead is superseded by
this direct measurement for this fixed CPU batch, not for every possible batch.

## Checks and evidence

**51 model-free tests pass in 16.263 s**, with model-runtime imports blocked.
They cover release evidence/lineage rejection, fresh-copy mutation protection,
frozen admission metadata, absence of full-manifest reads in the hot builders,
locked/unselected rejection before model imports, unchanged role whitelists,
Reason isolation, sampling/resume and all 130 existing E input views. Test CLI
subprocesses run with CUDA invisible and use only the model-free release loader.

The initial run had one new fixture error: a missing operation field. The
fixture was corrected; the final full 51-test run passes. The failed log is
retained. No product error was hidden or semantic annotation repaired.

Compilation, whitespace, executed-source hashes, raw snapshot identities,
loader-validation AST equality, timing-median recomputation and evidence checks
pass. The benchmark exits **0**. Existing GPU numeric/gradient reports remain
historical evidence; no new GPU or full-flow qualification is claimed.

Raw evidence: `data/manifest-access-20261107/`, including literal baseline
sources, executed source snapshots, all case hashes/timings, both test logs and
the CPU receipt audit. [Portable evidence](evidence/manifest-access-v1.json)
contains the measurements, release/source identities and raw-file hashes.

Reproduce with a new output directory and baseline `release.py`,
`component_producer.py` and `producer.py` extracted from `8459e84`:

```sh
CUDA_VISIBLE_DEVICES='' python3 -m scripts.benchmark_manifest \
  --baseline data/manifest-access-20261107/baseline-source \
  --output data/manifest-access-20261107/reproduction
```

**Remaining:** full training/GPU speed and checkpoint/cache-policy comparisons
are unmeasured and paused by the user's noise constraint. Repeated asset lookup
and audit serialization are possible independent CPU profiling targets, not
changes made or speedups claimed here. Noul transfer and Score numerical failures
remain unresolved; no new training or data is queued.
