# Implementation handoff and remaining decisions

Status: documentation consolidation and implementation plan. No implementation,
model workload, service change or deployment was performed or authorized by
writing this file. Follow the [specification](architecture-and-training-plan.md)
and [document ownership map](README.md).

Read the [final audit](final-decision-audit.md) for closure assessment. All seven
recent topics and the [typed-value ABI](typed-value-abi.md) are recorded. A1 is
closed as a design. The [model configuration](model-configuration.md) now records
A2; [numeric profile v1](numeric-profile-v1.md) records A3 and
[training run profile v1](training-run-profile-v1.md) records A4. These are initial
designs, with implementation and GPU evidence still absent. The recorded
[B1 encoding](neural-input-encoding-v1.md) fixes the exact neural input map;
matching reference vectors and codec interoperability remain to be implemented
and validated.

The [Astra review](review-astra-2026-11-05.md) identified additional contract gaps.
B1 section 11 now records the F2/F4 role/target-presentation response and fixes
startup value meaning. The [current audit](final-decision-audit.md) tracks the
other findings; no all-choices-closed or all-G0-passed claim is justified.

The [resource owner](position-limits-and-budget-accounting.md#admission-and-reservation)
now records F1's position/handle/Ref contract, allocation manifest and boundary
example. Implementation and independent re-review are still required; do not
turn a recorded capacity argument into a passing runtime/GPU test.

The [exact module/optimizer contract](module-and-optimizer-contract-v1.md) records
F6's blocks, complete head/pointer inventory, shared parameter owners and
per-component eligibility-based accumulation. G1 must verify ownership,
inactive-gradient behavior, combined gradients and actual GPU memory use;
the recorded design does not establish these results.

The [training-input rules](training-input-rules-v1.md) record F7's role/branch
whitelists, permitted outcome-bearing Reason supervision and exposure/mismatch
rules. Implement field builders and provenance/invariance checks; F3 clean
tensor construction is now recorded in the [clean-test contract](clean-test-construction-v1.md),
with independent re-review/implementation/GPU validation absent. The
[promotion matrix](promotion-matrix-v1.md) now records F8's slice/coverage,
denominator, effort and phase/regression rules. Actual reviewed artifacts and
independent re-review/implementation/GPU validation remain unperformed.

The [G1-core proof protocol](small-gpu-proof-v1.md) now fixes two <=256-position
traces, path-specific gradients/negative controls, exact accounting and a bounded
comparison bundle. The [original RTX3090 run](g1-core-results-2026-11-05.md) failed.
The precision-stage rerun passed numerics but remained blocked on released Noul.
The subsequent [gate repair/rerun](g1-gate-results-2026-11-05.md) PASSED G1-core:
72 numeric cases at unchanged tolerances, 36 released continuations, raw gate,
cache/gradient/accounting/optimizer and frozen-base checks. The isolated child
changes only the existing gate weight/bias, not the live capability checkpoint.
A later instruction authorized the [full codec/runtime](full-codec-implementation.md)
and next qualification only. Its [original expanded BF16 profile](codec-qualification-results-2026-11-05.md)
failed; component/reference successes did not override that result. The later
[feedback precision v3](codec-feedback-precision-v3.md) revision and
[rerun evidence](codec-feedback-results-2026-11-05.md) are separately versioned.
Training, reviewed-data and frontend expansion remain excluded.
The [post-proof expansion plan](post-proof-expansion-plan.md) records the requested
conditional work order after measured success: full codec/component qualification,
reviewed data, A4 phases, same-job frontend integration and release preparation.
G1-core cannot substitute for full A3, 512/1024 qualification or semantic promotion.

## Consolidation decisions

| Issue found | Resolution |
|---|---|
| Protocol decisions repeated in exploratory flows | Topic contracts own details; specification owns cross-component decisions; flows remain semantic examples |
| Early documents call agreed representation/status choices undefined | Link their recorded definitions; retain GPU/semantic feasibility as unverified |
| Gate sometimes described before writing and sometimes at dispatch | One operation-intent gate begins a native draft; learned completion seals it; dispatch is a mechanical admission step |
| Prototype accounting confused with planned accounting | Label prototype separately; target B pays for reasoning and all later latent work, including answering |
| Base frozen versus learned actual LM-head rows | Freeze pretrained parameters; scalar rows are explicit learned extensions, with tied-matrix handling validated |
| Arbitrary example budgets look like operation budgets | Keep .04/.05/.06 as review-only preferences; no operation-to-budget rule |
| Repeated full examples risk diverging from walkthroughs | Example index links the three complete walkthroughs as their owners |
| Data scale-up precedes proof of trainable switched state | Short GPU feasibility gate precedes large data production and full-flow training |
| Numeric readiness targets lack denominators | Data/readiness owner defines scoring units, failed calls and coverage separately |
| Original prompts were discussed during design | Retain their exclusion from tuning; do not market them as fresh blind tests |
| Latent packet layout | [Recorded codec](latent-packet-codec.md) fixes typed records, role-aware framing, question/rubric echoes and differentiable scalar delivery |
| Native writer schedule | [Recorded schedule](latent-writer-schedule.md) emits record payloads in parallel per native step and learns pointers/completion within bounded preparation |
| Semantic supervision | [Recorded design](semantic-supervision.md) freezes answer-blind reference encodings/decoder and retains semantic alignment plus behavioral interventions |
| Loss configuration | [Recorded defaults](loss-configuration-and-training-schedule.md) define normalized losses, reference-validity masks, valid-set Choice loss and forcing/joint stages |
| Scalar rows | [Recorded handling](scalar-row-handling.md) isolates rows 248077/248078 over the frozen tied table, excludes them from text and matches train/export arithmetic |
| Numeric checks | [Recorded plan](numeric-verification.md) separates exact invariants, float tolerances, discrete parity and targeted gradient checks |
| A3/A4 integration | Explicit numeric kernels/near-zero/budget rules and 7000-window training profile; qualify 1024 positions before joint training; development promotion remains separate from locked tests |
| B1 exact encoding | [Recorded map](neural-input-encoding-v1.md) pins enum IDs, metadata offsets, helper formulas, counter fractions and Noul token rows 3721/1802; reference-vector/GPU validation remains unperformed |
| Post-review role/target mapping | B1 section 11 defines per-role fields/tags/owners/lengths/absence, request-only target attrs, historical remapping and sealed-render traversal; other Astra contracts remain open |
| Data operations | [Recorded procedure](data-operations.md) fixes annotation/review, lineage, provenance and releases; actual reviews remain unperformed |

## Dependency order for future authorized work

The recorded [control flow](control-flow-v1.md) resolves F5's initial policy:
zero extra consumption/hold, attempt ceiling 352, explicit delimiter IDs/masks,
at most one notice per closed attempt, code12=WriterAborted and H<=22. This is
a documentation change; source code/legacy prototype behavior remains unchanged.

| Package | Deliverable | Must pass before moving on |
|---|---|---|
| G0: contract freeze and pilot review | Recorded A1-A4/B1 profiles, typed records/layout, action/completion contract, actual versioned manifest and small reviewed fixtures | Reference examples agree with schema/ledger; G1 fixtures explicit; B1 reference vectors produced/checked before G2 interoperability; numeric calibration follows in G1 |
| G1: short GPU feasibility | [G1-core](small-gpu-proof-v1.md) differentiable W+D -> W+R -> W+D -> W+R traces, then broader qualification | Core accounting/return-path gradients/continuation pass first; full applicable A3, expanded operations/statuses/length qualification remain distinct |
| G2: request/memory/runtime | Immutable store, learned reference interfaces, serial dispatcher, budget admission and cancellation | Static/mechanical invariants, malformed/stale/duplicate handling, no gold/domain-rule substitution |
| G3: capability training | A4 reference/component blocks, writer/receiver alignment, typed heads, native reader and replay | Reference/component promotion fixtures pass; reference artifacts frozen; actual updates to native R |
| G4: complete trajectories | A4 guided/transition/free-control blocks, actual packets/results, multi-call and answer-phase use | A3 qualification at 1024 first; bindings/full gradients/failure continuations; final-two-checkpoint development promotion; distinguish forced/free routes |
| G5: free generation and UI integration | One Send, normal streamed answer, observational decision events, targeted cancel | Locked functional/semantic tests pass without hints, forced gates or supplied gold results |
| G6: length expansion and release preparation | Longer histories, reproducible checkpoints/reports and reviewable deployment diff | Claims bounded by tested lengths; later deployment requires appropriate user authorization |

Small hand-checked fixtures and static bookkeeping can precede G1. Do not generate
thousands of reviewed trajectories or tune a full curriculum before the core GPU
path is feasible. G2 work may be designed earlier; its integration depends on G1.
No CPU model execution, silent second prefill or second full model is a fallback.

## G0 items that must be explicit

| Item | What must be specified | Verification |
|---|---|---|
| Packet codec | [Codec](latent-packet-codec.md) and [typed ABI](typed-value-abi.md) fix layout/body costs and bounded value heads; instantiate model profile and validate | Construct example/worst-case packets and verify all-status return admission under 256/64 ceilings |
| Model configuration | [Recorded profile](model-configuration.md) and [exact module contract](module-and-optimizer-contract-v1.md) fix dimensions, block equations, R/D/T adapters, complete heads, initialization and optimizer owners | Instantiate exact shapes; verify reference/live isolation, parameter ownership, gradients and GPU feasibility before capability claims |
| Writer schedule | [Recorded schedule](latent-writer-schedule.md), typed ABI and model profile fix start/record/finish, bounded value heads, feedback and object counters | No uncharged feedback loop, invented candidates or silent truncation; finite completion/failure path within 32 native positions |
| Semantic supervision | [Recorded method](semantic-supervision.md), [input rules](training-input-rules-v1.md) and [model profile](model-configuration.md) fix role-whitelisted targets/pooling/decoder, Reason exception and branch exposure | No current verdict in question targets or desired score in rubrics; Reason/status targets valid for actual carrier; behavioral interventions establish meaning beyond a decoding probe |
| Loss/run configuration | [Loss defaults](loss-configuration-and-training-schedule.md) and [A4](training-run-profile-v1.md) fix weights/masks/forcing, blocks, optimizers, typed losses, corrections and promotion; instantiate actual hashes/fixtures | Unit/eligible-count accounting and gradients; development tuning only, no backprop-through-argmax claim or invented run |
| Neural input map B1 | Implement the [recorded encoding](neural-input-encoding-v1.md), not independently chosen train/serve maps; preserve A1 costs and A2 dimensions | Matching reference neural-input vectors in train/serve; reserved features zero; unsupported fields rejected; validate before G2 interoperability |
| Scalar-row handling | [Recorded design](scalar-row-handling.md) fixes private IDs, separate FP32 masters, BF16 effective rows, FP32 scalar accumulation and vocabulary exclusion; implement/validate shared provider | Unrelated pretrained rows remain unchanged; no scalar-ID sampling; no weight mutation inside an active graph |
| Numeric checks | [Recorded limits](numeric-verification.md) and [A3 profile v1](numeric-profile-v1.md) fix initial tolerances, near-zero rules, arithmetic, GPU kernels/flags and fixtures; implement and calibrate on development runs | Exact controls and gradients must pass alongside tensor comparisons; never tune tolerances on locked outcomes |
| Data operations | [Recorded procedure](data-operations.md) fixes labels/review/provenance and model-free pilot; assign reviewers and create reviewed artifacts | Independent review and split manifest before scaling/locking; no invented reviews or model runs |

G0 freezes initial methods and verification procedures. Numerical tolerances
may be calibrated on the small G1 reference/development runs, then frozen before
locked evaluation. Learning rates, loss weights and capacities may be revised
on development evidence with versioned changes; contract freeze does not claim
optimal hyperparameters or prohibit this declared development work.

Freeze work is scoped to the package: G1 fixtures need their concrete model/
numeric profile. A1-A4/B1 initial designs are recorded; general codec work follows
that map and still requires matching reference-vector/interop evidence. A minimal
G1 state proof uses declared fixtures without claiming full codec/checkpoint
compatibility.
Scoped proof readiness does not complete every G0/pipeline item.

These items are real implementation preparation, not already validated choices.
Do not fill gaps with an undocumented one-vector bridge, a second prompt or a
teacher supplying the answer in production. A GPU teacher, if used for offline
annotations, is separate from the one-backbone deployed inference contract.

## Training objectives and boundaries

Keep named losses and their eligibility masks visible in configuration/reporting:

| Objective | Applies where |
|---|---|
| Action and writer completion | Annotated native states/drafts; identify selected demonstration routes separately from sets of valid routes |
| Source/reference and exact-attribute fidelity | Fields with evidence-grounded annotations; exclude arbitrary UUID prediction |
| Question/rubric/candidate semantics | Separate semantic annotations and controlled interventions; not only final verdict accuracy |
| Status | Every semantic decision example with a valid status label |
| Noul truth | Assessed Noul cases with supported probability/truth supervision |
| Choice feasibility and selection | Reviewed bank/criterion; selection over acceptable set only for assessed cases |
| Continuous task Score | Assessed cases with reviewed rubric-grounded labels; no fabricated values on abstention |
| Initial effort preference | Examples with reviewed effort annotations; independent of task Score and not an optimality objective |
| Native continuation and replay | Appropriate answers, grounded reasons, direct paths, ordinary language and original decision tasks |

For ties, selection supervision must not punish another acceptable winner solely
because one reference trajectory chose a different entry. An acceptable-set
probability objective is the [recorded default](loss-configuration-and-training-schedule.md).
Loss weights and hyperparameters are development choices, not facts established
by these documents. No policy-gradient stage is assumed.

## Prototype migration map

Read-only inspection found these gaps; this is not a code or security audit:

| Existing location | Reuse or change for the target |
|---|---|
| `qwen_one_two/model.py` | Retain one CUDA backbone/pinned artifacts; replace trainability-changing adapter selection with explicit train-safe modes and functional state; support aligned D |
| `qwen_one_two/decision.py` | Retain useful pointer/readout initialization; replace one-vector query/feedback assumptions with typed fields, statuses and bound returns |
| `qwen_one_two/thinking.py` | Retain single submission/cancellation structure; integrate draft lifecycle, full B/A/H ledger, admission and explicit abort handling |
| `qwen_one_two/policy.py` | Retain pinned context and S-curve; use one position-policy owner for codec and runtime limits |
| `qwen_one_two/curriculum.py` | Legacy synthetic examples are prototype material, not the new reviewed dataset |
| `qwen_one_two/train.py` | Prototype detaches native hidden state on a latent-call path; not proof of complete gradient-connected training |
| `qwen_one_two/train_feedback.py` | Bridge-only result fitting with native path frozen must not stand in for training native R to interpret results |
| `qwen_one_two/server.py` and UI | Preserve single user submission and targeted cancel; expose automatic phase/events without requiring a second generation action |
| Existing tests/artifacts | Useful regression evidence only for their recorded prototype scope; new contract needs independent tests/manifests |

Introduce a new versioned protocol/checkpoint format for the new field/weight
layout. Preserve legacy artifacts unchanged. Reject unsupported manifests rather
than pretending old bridge-only checkpoints contain the new skills. Pin base,
Qev, native/decision adapter, heads/codec, tokenizer and dataset/split hashes.
Do not change the live service or `current` checkpoint pointer during design.

## Evidence bundle for each milestone

Record configuration and artifact hashes, GPU UUID/dtype, dataset/split lineage,
teacher-forcing flags, mode/phase trace, one-prefill/state progression, request
bindings, position debits/reservations, status/value presence, relevant gradient
checks, failures and semantic outcomes. Do not fabricate numerical evidence
from documentation or count duplicate scenario variants as independent proof.

The UI observes the same running inference; telemetry is not a second decision
endpoint. User-facing answers need no internal vectors, request IDs or scores
unless showing them serves an explicit inspection purpose.
