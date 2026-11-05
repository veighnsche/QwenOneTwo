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
