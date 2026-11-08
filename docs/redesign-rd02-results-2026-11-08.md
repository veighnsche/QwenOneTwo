# RD-02 — explicit receiver preparation and resumable identities

RD-02 extends the existing `scripts/train_receiver_probe.py` runner with a
model-free `--prepare` action and named input profiles. The
[portable evidence](evidence/receiver-preparation-v1.json) binds the local plans,
dependency hashes and exact check results. The [queue ledger](redesign-progress.md)
records delivery; RD-03 and later remain unstarted. No model or fit ran.

RD-01 was verified before implementation: its recorded source/test, raw artifact,
reused-audit and validation-log hashes were unchanged, both delivered commits
were ancestors of clean `main`, and remote `main` was `47c58a3`. Its existing
23-test result was reused. The [RD-01 limits](redesign-rd01-results-2026-11-08.md)
remain constraints on later learning; no missing annotation was invented.

## Preparation contract

The runner now requires exactly one action, `--prepare` or `--execute-gpu`.
Preparation requires an explicit `--input-profile`. It rejects execution,
resume and historical loss-balance options combined with preparation. No GPU
selection or model library is needed to prepare. Outputs must be fresh local
`data/` paths; the base snapshot is read only.

| Profile | Source / roles | Declared exposure | Execution status |
|---|---|---|---|
| `d-pooled-role-slots-v1` | Current D / existing frozen pooled reference | Historical 32-record probe, eight repeats, 256 presentations /32 windows | Existing legacy implementation; execution still requires explicit task authorization and GPU action |
| `frozen-w-pooled-role-slots-v1` | Proposed frozen W / existing frozen pooled reference | Proposed RD-08: 1,500 records, four passes, 6,000 presentations /750 windows | Pending RD-03/05/07 |
| `direct-role-states-v1` | Proposed frozen W / ordered direct roles | Same proposed RD-08 records, order and exposure | Pending RD-03/04/05/07 |

These are plans, not executed exposures. Native regime is explicitly rejected
for these component profiles; native wiring belongs to RD-06. The richer role
projection is listed as a pending RD-04 owner with unknown parameter names/shapes,
not a constructed module or fabricated initialization. A proposed full-catalogue
plan does not establish envelope coverage or authorize the RD-08 fit.

Each deterministic `receiver-preparation-v1` contract binds the reviewed
training/development releases, RD-01 inventory and ordered panels, row identities
and freshly checked permitted-input hashes, selected training order, profile and
regime, exact RTX 3090 UUID, initial checkpoint, frozen dependencies, named owner
manifest, objectives, schedule and failure policy. Complete E130/X180 panels,
subgroups and the 50-case Choice retention panel remain attached.

Preparation checks the base and diagnostic weight files, Qev initialization
files, checkpoint masters/moments/RNG files as opaque bytes. It hashes tokenizer,
template and configuration files, all package source files and runner/imported
release-constant scripts. It never imports a tensor/model library or interprets
weights as tensors. Historical numeric evidence is bound by actual file hashes;
it does not qualify a new profile or fabricate reference acceptance.

The initial state is the existing complete-window `checkpoint-0188`, retaining
its 188 scheduled windows/188 optimizer updates, Adam moments, RNG and A4 schedule
position. Its 414 receiver parameters have exactly one owner: D372,
Qev_pointer2, score1, new_matrix17 and new_other22. W, R, the budget row and the
reference remain frozen. The new profile's additional projection remains pending.
Existing status/value/feasibility/Reason alignment/structure coefficients and
eligibility masks are declared unchanged; no auxiliary count loss is added.

## Execution, cache and resume boundaries

A contract hash becomes the new checkpoint protocol identity. Different
profiles, releases, panels, owners, objectives, dependencies or exposures cannot
share a prepared checkpoint identity. `--prepared-plan` is re-created against
current metadata before execution admission, then actual release identities,
panel order and `Window.ownership` are checked. Wrong UUID, profile, native regime,
stale plan or mismatched checkpoint fails before model loading. New profiles
remain rejected while their implementation/qualification is pending.

Legacy GPU commands retain the existing loop. Bound legacy execution/resume uses
that same loop, `Window`, `checkpoint.save` and transactional
`checkpoint.restore_model`; there is no second trainer or queue framework.
`checkpoint_metadata.py` factors the existing opaque-byte identity inspection out
of the model-importing module. Tensor restore, Adam, clipping and update/window
algorithms are unchanged, checked structurally against the delivered baseline.
Real CUDA roundtrips were not run under this metadata-only submission.

Prepared resume requires both the checkpoint path and its exact manifest hash.
The checkpoint binds the full order, complete-window cursor, ownership, schedule,
reference, attempted exposures, eligibility/failure counters and completed
measurement receipts. Cursor advancement counts failed attempts; no replacement
record or automatic retry is inserted. Scheduled windows and actual optimizer
updates remain distinct. A missing/foreign/partial recovery state is rejected.
The new process reports the checkpoint prefix separately and resumes pending
measurements at their checkpoint boundary. It never treats the original parent
sampler as new-profile exposure history.

Cache contracts bind the actual input and coordinate-mapping hashes, profile,
regime, source/role policy, base/tokenizer/template/precision/code and frozen
reference where applicable. Mutable D-source and live consumer features are
ineligible for caching. Pooled reference caches cannot enter direct-role inputs.
These identity guards implement no RD-03 encoder or feature extraction.

`preparation.json` separates host preparation timing from the deterministic plan
and records zero attempts, encodings, Sessions, prefills, positions and updates.
The existing `Recorder` instruments actual source/reference encoding calls,
Session constructions and prefill calls only on explicit execution. Calls and
failed calls are separate; their host intervals are not GPU compute time.
Actual optimizer updates come from `Window`, and existing per-case causal ledgers
remain the position evidence. No timing, token or cost cutoff was added.

## Checks and reproduction

Raw plans, blocked-import launcher, logs and receipts are under
`data/redesign-rd02-20261108/verified/`. All three profiles were prepared through the
actual runner with `torch`, `numpy`, `safetensors`, `transformers` and `peft`
imports blocked. A second legacy preparation is byte-identical despite a different
output directory. The two proposed RD-08 arms share their full order, initial
state, owners and objectives; their profile/checkpoint identities differ.

```sh
python -m scripts.train_receiver_probe --prepare --input-profile d-pooled-role-slots-v1 --snapshot "$MODEL_SNAPSHOT" --output data/rd02-new-local-plan
PYTHONPATH=tests:. python -m unittest tests.test_receiver_plan tests.test_receiver_learning tests.test_training_schedule tests.test_training_driver tests.test_host_metrics.HostMetricsTests
PYTHONPATH=tests:. python -m unittest tests.test_receiver_plan
```

Use the existing local snapshot path recorded in each raw plan and a fresh output
directory. The evidence records the exact blocked launcher commands and results,
compilation command, source hashes, whitespace and local-link checks. The
40-test model-free suite passed; after final dependency/admission guard changes,
the 12 preparation tests were rerun and unchanged results for the other 28 reused.
Tests cover deterministic blocked-import preparation, ambiguous action rejection,
profile/GPU/regime checks, stale dependencies, base/checkpoint bytes, zero-work
receipts, cache bindings, opaque checkpoint integrity, ownership/cursor/counter
rejection and failed-exposure retention. No fixture performs a fit or optimizer
update. Existing driver/schedule/Recorder tests and unchanged optimizer/checkpoint
algorithm evidence are reused.

The initial broader invocation also encountered an existing
`tests.test_host_metrics.ProfilingPlanTests` setup failure: `profiling receipt
input drift`. Its 130 original rows all match; its saved view hashes all match
the historical input schema and none match current `clean-view-v2`. The relevant
profiler/input/test files are byte-identical to baseline `47c58a3`. This older
diagnostic fixture is not the Recorder test class or RD-02 runner. Its failure
and proof are retained; it was not relabeled as passing or silently repaired by
reusing stale inputs. Required RD-02 checks pass independently.

No required RD-02 technical check remains pending. New-path GPU qualification
and real tensor/resume integration remain future RD-07 work under the subsequent
slices, not a learned result from these metadata fixtures. Learning outcome is
not applicable. No model execution, training, agents, new records/reviews, locked
access, deployment, service or Operating Table changes occurred.
