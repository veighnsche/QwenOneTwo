# Backend release-boundary slices — 2026-11-05

This continues the [backend implementation slices](backend-slices-2026-11-05.md)
after `2da8e43`, under the user's instruction to continue. No semantic-data
generation, capability fitting, model execution, frontend or deployment is part
of this round. The existing 324/108 numeric matrices and 44 GPU results retain
their original measured sources and were not rerun.

## Slice 1: bind reviewed release evidence to actual bytes

Finding: record checks accepted arbitrary nonempty review-artifact hashes, while
the driver accepted an unverified pilot-review hash string. There was no loader
connecting those identities to actual annotation/source/review/release bytes.

Acceptance: a read-only, model-free loader checks exact content hashes, bound
review judgments and current annotations, named release approval, source identity,
acyclic ancestry, frozen family splits, quarantine propagation and required pilot/
batch review coverage. Changing annotations after review or selecting excluded
descendants must reject. Loaded snapshots remain immutable after local file edits.

Implementation: `training/release.py`, the shared review gate in `inputs.py`, and
the versioned [transport contract](reviewed-release-format-v1.md). A pilot requires
120 root families, 20 per specified domain, two independent review declarations
per scenario and no repeated identical inputs. Locked material also now requires
two independent reviews. The loader never creates annotation/review artifacts or
authenticates the honesty of their contents. Actual human review remains absent.

Validation: **23/23 model-free tests passed in 0.637 s**, including 13 new release
regressions and 10 existing input/provenance cases. Command:

```bash
PYTHONPATH=tests .venv/bin/python -m unittest \
  test_training_release test_training_inputs -v
```

Raw log: `data/backend-release-20261105/release-tests-final.log`. All fixtures are
temporary, explicitly marked schema-only objects; no real reviewer, annotation,
approval or pilot is claimed. The CLI pass and rejection paths were exercised.
Driver integration follows in a separate slice; this loader alone does not close
that boundary or implement task-specific model episodes.

## Slice 2: bind the driver, catalogue and resumed exposures

Acceptance: the native driver requires loaded training/pilot evidence with a
matching prerequisite identity. IDs and operation/status/domain catalogue fields
must match the approved training records exactly. Cross-release family or exact
input leakage rejects. Producer mutation cannot change later exposures. Resumed
cursors must describe the actual deterministic schedule at the claimed window.

Implementation: `Sampler.from_release` / `assert_release`, `bind_pilot`, and a
new explicit loaded-release driver signature replace the arbitrary pilot hash
and raw-record arguments. Per-window checks precede producer calls; reports pin
both releases. Failed exposures retain IDs/cursors and require explicit resume.
Sampler resume validates exact counters using identifier-only schedule replay.

Validation: **36/36 model-free tests passed in 0.820 s**: 13 release, 10 input,
5 schedule and 8 driver tests. New driver tests are explicitly simulations of
orchestration counters; no Session/model/optimizer is executed or certified.
The earlier actual Session/CUDA evidence is preserved separately. Command:

```bash
PYTHONPATH=tests .venv/bin/python -m unittest \
  test_training_driver test_training_schedule test_training_release test_training_inputs -v
```

Raw log: `data/backend-release-20261105/boundary-tests-final.log`. A fresh-process
import also confirms that release/driver/sampler preflight does not import torch.
No real reviewed pilot/review artifacts were found among local data manifests;
the existing Qev/qualification manifests are different artifact types. Actual
pilot integration and task-specific episode production remain blocked on real
reviewed annotations. These mechanics do not establish semantic competence.

## Slice 3: consume actual numeric evidence at the joint-phase gate

Finding: the joint-phase prerequisite still accepted an arbitrary pass dictionary
and hash string. Acceptance: load the existing pinned aggregate, verify all child
bytes/axes/invariants and archived sources, then require that immutable receipt
before any joint-phase producer runs. No numeric matrix rerun is needed.

`training/qualification.py` validates the recorded v3-1024-v1 bundle, all 108
unique cases, thirteen operation/status cells, parameter/tied-base evidence,
one-prefill/cache/accounting settlement and archived source content. The driver
requires its `NumericEvidence`; bare dictionaries reject. The loader does not
assert that later arbitrary source edits match the measured implementation.
The historical finite-guard delta remains documented separately. The G1 flag
remains a caller attestation, and actual A4 training memory is still unqualified.

Validation: **42/42 focused model-free tests passed in 0.950 s** (the preceding 36
plus six numeric-evidence regressions). After the final Boolean-marker guard,
the affected suites were rerun as recorded in the portable validation summary.
Test report-shaped objects are explicitly schema-only; their pass markers are
never measured model results. Actual recorded evidence was separately loaded:

```bash
.venv/bin/python -m qwen_one_two.training.qualification \
  data/codec-qualification/1024-v1-20261105 \
  --expected-hash 3387858c77ea2de9c1d9eb500a29c5991d14ede997503de18d033f81146d43cf
```

The actual bundle passed inspection: 108 cases, three matching child hashes and
28 archived source/test files. Its aggregate identity and original measurements
remain unchanged. Raw inspection receipt:
`data/backend-release-20261105/loaded-1024-evidence.json`. A fresh-process import
again confirms no torch/model runtime is loaded. No GPU workload or optimizer
step occurred in this continuation.

## Final evidence and next unfinished slice

After the final numeric pass-marker validation, **14/14 affected numeric/driver
tests passed in 0.312 s**. Commands, exact log hashes and test-source hashes are
in the [portable validation](evidence/backend-release-validation-2026-11-05.json).
The [new backend profile](evidence/backend-profile-release-v1.json) pins this
continuation's sources/contracts and verifies the same base/Qev/diagnostic
artifacts. Its SHA256 is
`929ce379f69f5510a03504aebae1d7ac57f302c3a7f6dd7dba6565865b9d9776`.
The prior backend profile and all numeric reports remain unchanged.
Final static checks recomputed all 41 current source hashes, 18 contract hashes,
four log hashes and test-source hashes; 85 local documentation links resolve.
`git diff --check` passes, and codec sources are unchanged from `2da8e43`.

Next unfinished slice: supply an actual independently reviewed pilot and named
training release, then implement task-specific episode producers against those
annotations and review semantic/promotion fixtures. No such reviews or semantic
targets were invented here. The current loader can check their bytes and
declarations, but cannot establish truthful review or derive semantic gold.
Full A4 participating-buffer/auxiliary-graph memory qualification remains separate
and requires the actual episode mix. No capability training is authorized by
these engineering results. See the [updated handoff](next-agent-handoff.md).
