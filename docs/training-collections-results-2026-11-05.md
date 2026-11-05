# Training and evaluation annotation collections — 2026-11-05

Completed the authorized data-production round using bounded Muse calls,
deterministic construction and model-assisted content review. All datasets remain
local under `data/training-collections-20261105/`. The
[catalogue](../data/training-collections-20261105/catalogue.json) lists usable
releases; `*-candidate-*` directories are superseded work history. No training ran.

| Collection | Selected records | Declared parent families | Manifest SHA256 |
|---|---:|---:|---|
| Training | 8,000 | 288 | `c70d4f44135a34c00e7fa0348833b1ec56613def15f5753231589492ed35478f` |
| Development | 1,000 | 120 | `5b8f4cffdc4f33e255d3a34306cac03a68c50dcd9dab905ae8d8adcf89c81e77` |
| Locked | 2,000 | 288 | `d13594f18738056ee179b3498d54d8261f7f7654aad633fabb776810681aa1cf` |

## Contents and reuse

Training contains 500 reference, 1,500 receiver, 750 writer, 750 reader,
2,000 interleaved, 500 direct-answer, 1,000 grounded-language replay and
1,000 decision-replay records. The first two groups supply the 2,000 controlled
exercises. The six core groups total 6,000; replay brings the total to 8,000.

Development RF/W/E/R/X/F allocations are 100/70/130/100/200/400. Locked
RF/W/E/R/M/X/F allocations are 100/100/130/130/40/500/1000. An additional
40 development mechanical executions derive from 20 writer families and do not
increase the 1,000 base-case count. The source pilot remains development-only.

Muse authored 576 fresh wording contexts: 288 training and 288 locked. Code
instantiated six controlled evidence states per context. Development reuses
the pilot and adds controlled counterfactuals within 60 of its existing families.
Canonical semantic definitions supply five reviewed meanings for each of ten
reference roles in each split. Derived role views and variants are correlated
examples, not additional independent scenarios.

## Content review and exact checks

Actual review prompts withheld proposed outcomes. Each contained source semantics
and all controlled evidence states, compressed through an explicit base/override
format. Reviewers independently derived outcomes and checked role boundaries.
Exact rules compute the labels; their outputs were compared with the reviews.
One review approved the precise composition rules for roles, candidate bijections,
semantic interventions, replay and prescribed corrections. Each final record
binds its constituent source and recipe reviews to its current annotation hash.
This is compositional content review, not 11,000 separate literal-row judgments.

The first review used alternate status/eligible-set serialization. Its explicit
normalization receipt preserves the raw response, including mapping a proven
empty eligible bank to typed NoFeasibleChoice. Subsequent reviews used canonical
statuses and complete feasibility maps. One unsupported development effort
preference was masked and excluded from effort-scored free-job selection.

The final audit verifies production release loading and pilot binding, no split
family overlap or identical training/evaluation input payloads, all requested
slice counts, family/status/domain minima, five reference meanings per role,
35 identical-input hidden-fact pairs, 30,302 exact numeric/Boolean source literals,
and 40 locked lifecycle fixtures through production Runtime metadata guards.
All five production sampler phases produce valid eight-example ID windows.
**64/64 model-free tests passed in 2.198 seconds.** No numeric matrix or GPU test
was rerun for this offline data work.

## Usage and reproducibility

**31 actual Muse calls:** 12 wording-authoring and 19 review calls. Each requested
low reasoning, one model step, no tools/reminders/delegation and no automatic
retry. Actual counters report 965,889 input, 254,566 output and zero cached tokens.
The 25,010 reasoning-token counter may be included in output; do not add it
blindly. These counters exclude Codex overhead and do not establish a currency
price. Remote hardware remains unexposed; no local RTX3090 execution is claimed.

The [portable receipt](evidence/training-collections-v1-summary.json) binds release
identities, actual calls/usage, tests, source and workflow hashes. Local artifacts:

- `train-examples.jsonl`, `development-examples.jsonl`, `locked-examples.jsonl`.
- `releases/{train,development,locked}/`: immutable sources, records, model reviews
  and automated approval under `automated-content-v1`.
- `calls/`, `checks/`, `reference-definitions/`: actual inputs/results and checks.
- `verify_collections.py`, `verification.json`, `tests.log`: independent audit.
- `development-secondary-mechanical.json`: the additional execution protocol.

Per-record generator hashes bind the exact source archived inside each release.
The checkout subsequently removes trailing whitespace only; Python AST equality
was verified, the nine construction tests passed again, and dataset bytes and
manifest identities remain unchanged.

Reproduce the read-only checks without calling a model:

```sh
PYTHONPATH=. .venv/bin/python data/training-collections-20261105/verify_collections.py
PYTHONPATH=tests .venv/bin/python -m unittest \
  test_data_production test_drafts test_training_inputs test_training_release \
  test_training_driver test_training_schedule -q
```

## Limits and next work

These are approved experimental **annotation** releases. Episode producers still
need pinned-tokenizer/source-token materialization, actual semantic packets and
reference artifacts. The 400 reserved correction records contain prescribed
intervention specifications; their correction losses remain masked until an
actual runtime packet/snapshot is bound. They are not captured student failures.

Replay is a provenance-preserving synthetic preservation proxy, not recovered
original pretraining/Qev data. Family IDs, repeated templates, model review and
passing metadata tests establish neither semantic independence nor learned
invocation. Full-trainer memory qualification and the existing training/promotion
gates remain required. No fitting, frontend work, deployment or live service
changes were performed or queued.
