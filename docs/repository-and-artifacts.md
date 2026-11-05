# Repository contents and local artifacts

Repository: https://github.com/veighnsche/QwenOneTwo

The repository includes the Python implementation, frontend, deployment templates,
dependency lockfile, tests, implementation specifications, training walkthroughs,
design decisions, Astra review, diagnostic repair protocols and measured G1 reports.
Start with the [documentation map](README.md) and
[latest G1-core results](g1-gate-results-2026-11-05.md).

The [GPU-only workspace rules](../AGENTS.md) apply to inference, training, feature
extraction, readout fitting and real-model tests. No CPU fallback is permitted.
Publishing this repository does not change or redeploy a running service.

## Artifact boundary

As required by the workspace rules, git excludes model caches, pretrained weights,
local trained/diagnostic checkpoints, raw generated artifacts, virtual environments
and local environment files. The ignored `data/` and `artifacts/` directories are
not part of a clone. Configuration examples and the project's license notices
are included. Do not commit credentials or a populated `.env`.

Measured reports in `docs/` record the local raw-report/checkpoint paths, SHA256
identities, hardware, protocol versions, results and limitations. Those paths are
provenance references to local files, not bundled checkpoint downloads. A fresh
clone does not contain the frozen diagnostic checkpoint used for the reported
run or the active prototype's trained artifacts.

The original failing [G1 run](g1-core-results-2026-11-05.md),
[precision-stage run](g1-precision-results-2026-11-05.md) and
[passing gate-repair run](g1-gate-results-2026-11-05.md) are preserved separately.
Their ignored raw JSON bundles remain local. The passing G1 result qualifies its
bounded engineering fixtures, not general latent semantics, full context or a
production capability checkpoint. No downstream expansion is included.

## Running from a clone

Follow the [GPU-only setup](../README.md#gpu-only-setup) to install dependencies
and obtain the pinned upstream Qwen/Qev artifacts. Downloaded artifacts retain
their own upstream license notices. Prototype training and G1 diagnostic
bootstrap/repair are separate workflows; neither replaces a reviewed A4 dataset
or capability-training run. Reproducing the exact recorded comparison requires
its matching diagnostic checkpoint and pinned numerical environment.

For G1, read the [proof protocol](small-gpu-proof-v1.md),
[compensated precision profile](g1-compensated-precision-v2.md) and
[gate-repair protocol](g1-gate-repair-v1.md) before running the corresponding
`qwen_one_two.g1_*` modules. Preserve original evidence and select a new output
directory for a new experiment.
