# Muse draft pilot — 2026-11-05

Completed the user's bounded request: repair Muse setup and generate exactly
12 draft scenarios, two per domain. These are model proposals in development,
with zero human reviews and no approvals. No training, fitting, local model
execution, frontend work, deployment or live-service changes occurred.

## Data and review copies

Local ignored directory: `data/muse-pilot-20261105T191615Z/drafts/`.

- `scenarios.jsonl`: 12 sealed scenario/annotation records with source bindings.
- `sources/`: 17 UTF8 source files with checked hashes and byte coordinates.
- `review-inputs.md`: complete prompts, supplied facts/rules and request-meaning
  proposals, with outcome/status/Reason/continuation proposals withheld.
- `outcome-proposals.md`: the separate unreviewed decision and continuation labels.
- `draft-review.md`, `manifest.json`, `validation-report.json`: review guide,
  provenance, artifact hashes, coverage and mechanical results.
- Immutable original lane handoffs and pre-repair snapshots remain available
  locally. They are history, not additional independent scenarios.

The two separate review copies were rendered mechanically by Codex from the
sealed Muse JSONL. This formatting changed no scenario annotations and counts
as no human review. Raw scenarios, sources, session logs and local configuration
remain outside git. The [portable summary](evidence/muse-draft-pilot-v1-summary.json)
records exact identities without publishing the generated dataset.

| Coverage | Draft count |
|---|---:|
| Refund eligibility / scheduling / demo preparation | 2 each |
| Access rules / resource allocation / issue prioritization | 2 each |
| Noul / Choice / Score | 5 / 4 / 3 |
| Assessed / Conflicting / Insufficient / NoFeasibleChoice | 7 / 2 / 2 / 1 |

There are **7 of 13** operation/status cells. Missing: Noul Insufficient/Unresolved,
Choice Conflicting/Unresolved, Score Conflicting/Unresolved. All twelve admit a
grounded direct route; annotated internal routes remain demonstrations. Effort
preferences are absent/masked. Score intervals remain explicitly unreviewed.

## Setup and actual producer

The supplied Muse skill entrypoint was preserved unchanged. Its four missing
references were reconstructed from that entrypoint, explicitly identified as
new guidance rather than recovered originals; the canonical skill folder and
relative reference links now validate.

The OS tmux snapshot reproduced a server exit on `capture-pane` even with an inert
pane. It was replaced user-locally with the
[official tmux-builds 3.7c release](https://github.com/tmux/tmux-builds/releases/tag/v3.7c),
whose archive matched published SHA256
`cc56bd1cc873eb6089c615f0496b072385bda8a6d944069f38564a5d49c128aa`.
The failed executable is quarantined outside PATH. No system package/service was
changed. The inert capture regression and actual Muse inspection then worked.

Muse CLI `1.4.14-R7309.1` matched the official stable channel before launch.
One task-owned interactive tmux coordinator used two native authors. Live settings
confirmed Contributor on and XHIGH; configured runs and completed provider events
confirmed `muse-spark-1.3-contributor` for the coordinator and both authors.
Native author prompts/session IDs are bound separately from the coordinator
assignment. Individual child effort fields were not exposed in retained native
records; XHIGH is recorded as requested/inherited, with the coordinator's live
TUI confirmation, rather than claimed as a separate per-child measurement.
Remote weights, sampling details and GPU hardware were unexposed and
remain null with explanations; no local RTX3090 run is claimed for remote Muse.

`automation_update` was unavailable. No heartbeat or substitute timer was created;
Codex supervised this run in the active chat through completion. Muse recorded
a clean exit; both native authors and commands released; the task-owned tmux
server exited and stale owned sockets were removed. No background work remains.

## Independent checks and repairs

Codex performed model-assisted implementation/content review, distinct from the
required human semantic/outcome reviews. Concrete findings were routed back to
Muse and repaired while unrelated authoring continued:

- Preserve absent values for every non-assessed status, keeping feasibility
  targets separate. Remove lexical censorship; enforce roles and inspect prose.
- Check actual source files within the draft root, exact Choice feasibility and
  acceptable sets, explicit unreviewed Score flags and optional point membership.
- Bind actual native author prompts and verify final canonical record hashes.
- Replace a design-exposed refund pattern; remove selection conclusions from
  candidate properties; supply full scheduling intervals; align rubric anchors.
- Remove unsupported code/pickup details; make necessary versus sufficient rules
  explicit; give conflicting claims the same observation time and align the user
  question to that assessment time. Preserve direct-answer alternatives.

Validation evidence:

- **36/36** focused model-free tests passed in **0.667 s** before the final
  Score/schema tightening: 13 draft, 10 input and 13 release tests.
- The final affected draft suite passed **14/14 in 0.014 s** after that tightening.
- Independent final artifact verification checks 12 unique families, two per
  domain, 17 sources, 28 manifest artifacts, author prompt/session bindings and
  all final record hashes. Swapping an in-memory loss label leaves actual request
  reference-input bytes unchanged. Original data remain untouched by that test.
- **12/12** draft records reject at the approved-record gate. No human reviewers,
  approvals, model rollouts or semantic competence results were fabricated.

Reproduce the model-free validation against the local artifacts:

```bash
PYTHONPATH=tests .venv/bin/python -m unittest test_training_drafts -v
.venv/bin/python -m qwen_one_two.training.drafts \
  data/muse-pilot-20261105T191615Z/drafts/scenarios.jsonl
```

The existing numeric matrices and GPU evidence were not rerun; this task changed
no codec arithmetic or learned weights. Mechanical validation establishes draft
consistency and declared identities, not semantic approval or a learned flow.

## Usage and remaining work

Observed usage was substantial: **194 unique provider-completion records**,
28,894,817 reported input tokens including 28,079,085 cached tokens, and 188,759
reported output tokens. The 100,501 reasoning-token counter may be a subset of
output and must not be added blindly. These totals include coordinator, author
and automatic-observer traffic; they are not unique dataset tokens or a currency
estimate. Deduplication uses actual source-run record IDs across runtime streams.
The local usage receipt preserves exact counters and log snapshot hashes. This
orchestration overhead should be assessed before scaling the generation process.

Next work: obtain the required independent human reviews, adjudicate/revise these
proposals, then decide whether/how to complete the 120-scenario pilot. These twelve
are not an approved full pilot or training release. Preserve their model-proposal
origin in any later release; the existing authored-root pilot admission rule
needs explicit reconciliation before accepting a reviewed model-proposed pilot.
Task-specific episode producers, semantic/promotion fixtures and actual A4
training-memory qualification remain separate unfinished work. No training starts
from these draft artifacts.
