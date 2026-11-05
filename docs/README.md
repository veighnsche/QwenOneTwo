# QwenOneTwo documentation map

For current measured status, artifact locations and remaining tasks, start with
the [next-agent handoff](next-agent-handoff.md). The v3 324-case numeric matrix
through 512 positions and the separate 108-case 1024 extension pass. Backend
training infrastructure is implemented; learned semantic capability remains untrained.

The [repository and artifact guide](repository-and-artifacts.md) explains what
is included in GitHub, which evidence remains local and how to run from a clone.

Start with the [implementation specification](architecture-and-training-plan.md),
then the [implementation handoff](implementation-handoff.md). The first owns
cross-component decisions; the second owns work order, migration gaps and
unresolved items. Neither authorizes coding, training or deployment by itself.

The [final decision audit](final-decision-audit.md) records reconciled ambiguities,
remaining interface/configuration freeze items and coding-readiness assessment.

A1-A4 and B1 initial design profiles are recorded and consolidated in the
specification. The full codec/runtime, engineering vectors and v3 numeric
qualification through 1024 positions now have scoped implementation/evidence.
Role-specific input/clean builders and training infrastructure are implemented.
Approved semantic datasets, task-specific episode integration, full A4 training
memory qualification, reference/capability training and frontend migration remain outstanding. A recorded design is not trained
capability; use the current handoff and measured reports for completion status.

The [Astra review](review-astra-2026-11-05.md) is a historical independent report,
not a live claim about subsequently edited sources. B1 section 11 records the
role/target mapping response to F2/F4; the resource owner records the F1 admission
response. F1-F8 responses are recorded; independent review of the later
implementation, reviewed training artifacts and further empirical validation
remain outstanding.

## Authority and status

User instructions and AGENTS.md take precedence. Within these docs, the
implementation specification resolves cross-document conflicts. Detailed topic
contracts below own their respective fields; examples illustrate them and do
not override them. Future changes must update the owner and affected examples.

- **Requirement:** behavior requested by the user or necessary to preserve it.
- **Design default:** selected starting design; version changes and verify them.
- **Review draft:** numbers/annotations needing review before use as gold data.
- **Unverified:** capability or technical hypothesis requiring evidence.
- **Prototype:** historical local implementation, not the complete target flow.

## Topic owners

| Document | Owns |
|---|---|
| [Implementation specification](architecture-and-training-plan.md) | Requirements, component ownership, lifecycle, contract summaries |
| [Implementation handoff](implementation-handoff.md) | Dependency order, implementation gaps, remaining freeze decisions, evidence needed |
| [Final decision audit](final-decision-audit.md) | Current design/validation status, recorded profiles and remaining Astra findings |
| [Independent Astra review](review-astra-2026-11-05.md) | Historical full-document/math review, original source hashes, findings and readiness verdicts |
| [Latent representation](latent-representation-v1.md) | Semantic field roles, shapes and exact-data requirements |
| [Packet codec](latent-packet-codec.md) | Typed request/result layout, role encodings, exact-value bodies and structural position costs |
| [Typed-value ABI](typed-value-abi.md) | Exact body costs, bounded per-type writer heads and all-status return admission |
| [Model configuration](model-configuration.md) | A2 dimensions, pooling/projections, normalization, adapters, precision and frozen/live codec boundaries |
| [Exact modules and optimizer ownership](module-and-optimizer-contract-v1.md) | F6 block equations, shared trunks, heads/pointers, initialization, parameter groups and per-component eligible-example accumulation |
| [Neural input encoding v1](neural-input-encoding-v1.md) | B1 bits/helpers/counters/Noul initialization; section 11's per-role field/tag/owner/length/absence profiles, target grammar and historical/handle rules |
| [Writer schedule](latent-writer-schedule.md) | Native preparation steps, parallel payload heads, learned pointers, draft feedback and completion |
| [Control flow v1](control-flow-v1.md) | F5 state/stop policy, zero extra consumption, exact chat tokens/masks, bounded notices, cancellation and terminal outcomes |
| [Semantic supervision](semantic-supervision.md) | Answer-blind reference encodings, frozen semantic targets, reconstruction/alignment and invariance tests |
| [Training-input rules v1](training-input-rules-v1.md) | F7 per-role/branch whitelists, permitted outcome-bearing Reason supervision, actual-status masks and teacher-forcing exposure |
| [Clean test construction v1](clean-test-construction-v1.md) | F3 permitted-view tensor lineage, fresh source/state construction, hidden-fact twins and separate clean/transfer/whole-flow protocols |
| [Promotion matrix v1](promotion-matrix-v1.md) | F8 exact slice allocations/family minima, 13 status cells, intervention groups, all-attempt denominators, unforced coverage, effort gates and phase/regression protocols |
| [Small GPU proof v1](small-gpu-proof-v1.md) | G1-core protocol and versioned passing scoped execution: one-cache traces, targeted gradients, accounting and required pass evidence |
| [Original G1-core results](g1-core-results-2026-11-05.md) | RTX3090 execution: cache/connected-gradient/accounting evidence, failed cross-precision thresholds and blocked continuation; expansion remains gated |
| [G1 precision-stage results](g1-precision-results-2026-11-05.md) | Complete 72-case numeric pass with unchanged thresholds; cache/gradient/accounting/optimizer pass; overall G1 blocked by released Noul Choice |
| [G1 compensated precision v2](g1-compensated-precision-v2.md) | BF16 truncation diagnosis, controlled ablations, compensated GPU products and explicit FP32 zones; unchanged thresholds and no expansion |
| [G1 gate measured results](g1-gate-results-2026-11-05.md) | Latest bounded G1-core PASSED: 72 numeric cases, 36 released continuations, 144 gate labels and 612 unforced decisions; no expansion |
| [G1 gate repair](g1-gate-repair-v1.md) | Undertrained readout and supervision-counter mismatch, bounded GPU-only head fit, unchanged four-way argmax and stronger anti-collapse checks |
| [Full codec implementation](full-codec-implementation.md) | Full typed transport, learned writer interfaces, immutable memory, resource reservations and one-cache runtime; no training/frontend |
| [Original full codec qualification](codec-qualification-results-2026-11-05.md) | Historical component/reference results and failed 512-position BF16 arithmetic gate; retained alongside the later v3 pass |
| [Feedback precision v3](codec-feedback-precision-v3.md) | Serial feedback diagnosis, projection ablations and explicit BF16-storage/FP32-feedback arithmetic revision; unchanged tolerances |
| [Feedback v3 qualification](codec-feedback-results-2026-11-05.md) | Follow-up matrix evidence, cache/gradient/accounting results and the remaining scope limits |
| [1024 qualification](codec-1024-results-2026-11-05.md) | Separate 108-case numeric pass, exact source provenance and memory limits |
| [Backend slices](backend-slices-2026-11-05.md) | Contract fixes, F7/F3 builders, losses/accumulation, reference isolation and export/resume evidence |
| [Release-boundary slices](backend-release-slices-2026-11-05.md) | Content-bound releases, pilot/sampler/driver identities, deterministic resumed exposures and actual numeric-evidence inspection |
| [Reviewed release format v1](reviewed-release-format-v1.md) | Local source/review/approval bytes, immutable release snapshots, pilot/batch checks and driver ingestion |
| [Post-proof expansion plan](post-proof-expansion-plan.md) | Conditional codec/runtime qualification, reviewed data, training phases, same-job frontend integration and release order after measured proof success |
| [Losses and training schedule](loss-configuration-and-training-schedule.md) | Starting weights, eligibility masks, replay, forcing dimensions and joint-training advancement |
| [Scalar rows](scalar-row-handling.md) | Private row indices, tied-table overrides, gradient ownership, vocabulary masking and precision/export |
| [Numeric verification](numeric-verification.md) | Exact invariants, forward/gradient tolerances, control parity and export/update checks |
| [Numeric profile v1](numeric-profile-v1.md) | A3 near-zero rules, ordered budget arithmetic, explicit GPU kernel flags and numeric fixtures |
| [Training run profile v1](training-run-profile-v1.md) | A4 phase lengths, optimizer schedule/groups, exact loss subtypes, correction allocation and development promotion |
| [Memory and snapshots](memory-and-request-snapshots.md) | References, immutability, pending requests and one-time consumption |
| [Result statuses](result-statuses.md) | Semantic status/value-presence semantics and runtime-outcome distinction |
| [Budget accounting and resource admission](position-limits-and-budget-accounting.md) | Position counters; position/handle/immutable-record holds, frozen K, allocation manifest, release rules and exhaustion |
| [Cache and gradients](cache-and-gradient-strategy.md) | Alternating state history, differentiable training and hard boundaries |
| [Data and readiness](training-data-and-readiness.md) | Counts, annotation/review, splits, scoring and acceptance targets |
| [Data generation](training-data-generation.md) | Scenario construction, variations and pilot production |
| [Data operations](data-operations.md) | Annotation instructions, independent review/adjudication, provenance and release lifecycle |

## Worked examples

The [example index](complete-training-examples.md) retains shared conventions
and the draft labels. Each full walkthrough owns its example-specific steps:

- [Noul: workshop access](noul-training-walkthrough.md).
- [Choice: printer selection](choice-training-walkthrough.md).
- [Score: packing-shift disruption](score-training-walkthrough.md).

These are supervised review drafts. They are not model runs, approved gold
data, mandatory invocation routes or evidence that the system is ready.

## Original discussion prompts

The [natural prompts](natural-invocation-prompts.md) and their
[refund](refund-noul-flow.md), [meeting](meeting-choice-flow.md) and
[demo](demo-score-flow.md) flows preserve the original semantic discussion.
Their general protocol questions are resolved by the topic contracts above.
Keep these prompts and close paraphrases out of training/tuning. They were
discussed extensively during design, so report them as illustrative cases,
not fresh independent evidence of generalization.

## Current versus target behavior

The [project README](../README.md) describes the legacy prototype and its setup.
Its commands do not train the complete target architecture. Documentation
consolidation changes no service, checkpoint, model weights or execution path.
