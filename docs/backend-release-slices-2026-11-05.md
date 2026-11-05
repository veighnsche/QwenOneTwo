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
