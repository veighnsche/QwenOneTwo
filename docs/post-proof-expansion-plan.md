# Expansion after the small GPU proof

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

Owner: dependencies and deliverables after [G1-core](small-gpu-proof-v1.md).
The [specification](architecture-and-training-plan.md) and detailed topic owners
retain authority over implementation behavior, shapes, data and acceptance.
This plan does not replace [A3](numeric-profile-v1.md), [A4](training-run-profile-v1.md)
or the [promotion matrix](promotion-matrix-v1.md).

## 1. Accept measured proof evidence first

Entry requires the complete G1-core report: one logical prefill/state lineage,
targeted through-value gradients and negative controls, train/serve agreement,
exact admission/debits/releases, actual native continuation and measured GPU
memory. All required cases must pass. Missing measurements are not an assumed
pass; prescribed calls are not learned invocation. Fix/version failed mechanisms
and repeat affected checks before expansion.

Keep bootstrap weights and fixture traces as explicitly diagnostic artifacts.
Do not silently promote them to capability checkpoints. Capability training uses
the recorded initialization/reference preparation with known provenance.

## 2. Expand the codec/runtime, then qualify the larger path

Implement the full recorded [codec](latent-packet-codec.md), [typed ABI](typed-value-abi.md),
[B1 role map](neural-input-encoding-v1.md), [F6 modules](module-and-optimizer-contract-v1.md),
[memory](memory-and-request-snapshots.md), [admission](position-limits-and-budget-accounting.md)
and [control flow](control-flow-v1.md). Avoid independent train/serve encoders.

| Expansion | Required evidence |
|---|---|
| Choice | Candidate construction/reuse, frozen bank, feasibility/acceptable-set selection, selected contents delivered; no claimed derivative through argmax |
| All13 operation/status cells | Assessed values versus non-assessed Reason; genuine zero distinct from absent; no fake value/Reason on mechanical failures |
| All typed bodies | Exact Boolean/int64/decimal/ref/source-string/UTF8, active lengths/absence, units/provenance, legal clipping and rejection |
| All semantic roles | Q/T/candidate/rubric/key/Reason counts, presence, owner/tag/metadata mappings and ordered projections |
| Immutable memory | New versions, historical reuse, sealed snapshots, frozen K, no stale/self-supporting/foreign refs |
| Resource/lifecycle boundaries | Request256/257, return64, candidate16, writer32, attempts16, records32, handles256, independent holds, abort/cancel/duplicate/late handling |
| Both phases/multiple calls | Continuous state and gradients across serial invocations; answer calls spend original B; zero allowance and forced/natural closure |
| Scalar/export behavior | Private rows, tying, inactive grad=None, frozen hashes, save/load and matched behavior |

Generate matching exact metadata/body fixtures and GPU neural vectors under the
same profile. Keep clean components, production-tensor transfers and whole-flow
fixtures distinct; apply [F3](clean-test-construction-v1.md) and
[F7](training-input-rules-v1.md) throughout. No original-context hidden states
may masquerade as clean evidence omissions.

Qualify complete component paths up to512 total positions, without truncating
packets or detaching gradients. Measure memory including component buffers and
optimizer state. G1-core alone does not qualify this path. Before joint training,
extend applicable numeric/gradient/state/accounting checks to1024 total positions
as A4 requires. C remains262144; neither guard establishes full-context operation.
If GPU memory/backward fails, fix/version the GPU implementation or report the
blocker; do not substitute CPU, another prefill or a detached call boundary.

## 3. Build and review the pilot before data scale-up

Use [data operations](data-operations.md) and [generation](training-data-generation.md):
120 distinct authored fictional scenarios,20 in each of six domains. Small
hand-checked engineering fixtures may precede proof; bulk semantic production
waits for core feasibility and pilot acceptance.

Keep human prompts natural and operation-name-free. Store separate canonical
semantics, exact grounded values/refs, admissible routes, statuses/value labels,
Reason/continuation targets, effort preferences and provenance. No handwritten
latent vectors or outcome columns in request inputs. Reason's permitted outcome
content follows F7. Label actual views/packets, not withheld world truth.

Every pilot scenario needs two independent semantic/outcome reviews and any
required adjudication; assigned reviewers and approvals must be real. Review
request semantics before proposed outcomes, then outcomes, Reason and native
continuation for the actual result. Treat subjective effort/Score examples as
drafts until reviewed. Quarantine critical errors and affected descendants.

Pilot exit: approved versioned scenario/annotation manifests, valid source
coordinates, consistent branch masks, demonstrated annotation instructions and
coverage gaps explicitly recorded. Mechanical validity is not semantic review.
No fictional reviewer names, automatic self-certification or invented approvals.

## 4. Produce versioned training/development/locked releases

After pilot review and applicable component feasibility:

| Release | Quantity |
|---|---:|
| Core training | 6000 |
| Reviewed replay | 2000:1000 language,1000 decision |
| Development | 1000, exact promotion slices |
| Locked | 2000, exact promotion slices |

Keep underlying scenario/rule/template/rubric families together and separate
across splits before making variants. Reserve400 existing core examples for A4
corrections, not400 extra independently authored cases. Preserve rejected rows,
lineage, release hashes, reviewers, label scopes and rights/provenance of replay.
Locked tests enter neither teacher prompts nor reference fitting, contrasts,
corrections or tuning. Design-exposed original prompts/walkthroughs cannot be
marketed as fresh blind evidence.

Instantiate the [promotion matrix](promotion-matrix-v1.md) exactly:13 status
cells, role/intervention groups, family minima, effort bands and separate
protocols. Insufficient coverage is incomplete; never force model calls or invent
effort labels to fill it. Model assistance, if useful and later executed, is
GPU-only and supplies proposals, not approval. Existing8000 counts are a starting
release; version changes on development evidence, never silently relabel locked
cases after predictions.

## 5. Train in A4 order with explicit exits

| Phase | Scheduled windows | Exit |
|---|---:|---|
| Reference semantic codec | 1000 | Reviewed recovery/contrast gates; freeze/version reference pooler,codec,T |
| Components | 2000 | Writer/receiver/reader, status/value, effort, replay and applicable numerical gates |
| Guided joint | 1000 | Actual predicted slots/returns; complete connected trajectories under declared guidance |
| Transition | 2000 | Compatible prefix guidance shrinks to zero; no overrides; actual rollout-state supervision |
| Free-control joint | 1000 | No forced decisions/gold returns; unforced promotion separately generates native text too |

Seven thousand windows of eight scheduled examples gives56000 presentations,
including repetitions/reference records. It does not mean56000 independent
scenarios. Use A4 mixture/schedule and F6 eligible-example component buffers,
unique optimizer ownership and complete-episode updates. Initialize live codec
from frozen compatible reference weights with independent storage/fresh moments;
carry live optimizer moments across subsequent live phases. W stays frozen.

Keep continuous initial effort separate from task Score; both are actual scalar
rows. Teach invocation/writing with explicit discrete supervision and result
interpretation through applicable connected losses; no derivative through hard
choices is invented. Preserve semantic/grounding objectives and replay throughout.
Gold return exercises remain isolated readers, never replacements for joint
Qev output. Outcome-bearing prefixes and overrides remain separately logged.

Evaluate every100 updates, pass at BOTH final checkpoints U-100/U, and satisfy
all applicable promotion rows before advancing. If a block fails, preserve its
report and make a versioned development revision; no automatic extra epochs,
earlier favorable checkpoint or widened thresholds. Recheck cache/gradients/
numeric invariants as capabilities grow, including1024 qualification before
joint training. Passing diagnostic G1-core is not any phase's semantic promotion.

## 6. Integrate the frontend with the same inference job

Read-only inspection found the legacy browser starts one generate request and
polls run state (static/app.js/qwen_one_two/server.py). Adapt that single-job
contract; do not assume an existing SSE/WebSocket transport. Incremental answer
display may use the existing polling mechanism, provided updates are authentic
outputs of the same continuous inference. A transport change is not a new
decision or generation job.

| User interaction | Target behavior |
|---|---|
| One Send | Start budget prediction, automatic learned decisions and normal answer in one job/cache |
| Progress/inspection | Observe actual budget/phase/call/status events from that job; no decision-first confirmation or second Run button |
| Normal output | Incrementally display Qwen's answer from the same run; diagnostic data is optional inspection, not the answer |
| Budget | Show actual continuous prediction/requested grant; keep override explicit and bounded; zero thinking still permits answer |
| Stop | Cancel the owned run; ignore late events/deliveries; settle GPU ownership before reuse |
| Reconnect/polling | Reconcile existing run events/state without resubmitting prompt or restarting inference |
| Completion/failure | Respect Completed/Limited/Cancelled/Failed; never show stale output as a new run |
| Unavailable checkpoint | Report actual readiness; never silently fit a head, use diagnostic weights, CPU or another model |

Backend run identity, monotonic event sequence, terminal state and immutable
checkpoint/profile identity must make stale/duplicate updates detectable. Preserve
existing access validation/run ownership. Telemetry reads do not invoke the
decision model, spend model positions or trigger second prefill. API transport
may serialize metadata for inspection; deployed latent calls remain textless.

UI work can use explicitly labelled recorded/synthetic fixtures while training
is incomplete. Such fixture-based checks establish UI behavior only. Integration
exit requires real model traces with same-prefill counters, learned unforced
calls, actual returned values and normal answer/cancel behavior. No mock event
stream or forced example can certify learned capability.

## 7. Locked evaluation and release preparation

Evaluate the final development-promoted model on the fixed locked release under
F8. Keep clean/transfer/guided/free/runtime/replay reports separate. Include every
scheduled failure/attempt, coverage, family uncertainty, effort quality and exact
execution invariants; a recovered answer does not erase an incorrect judgment.
If locked failures drive tuning, retire that material to development and create
a fresh locked release for a fresh claim.

Prepare versioned model/codec/head/optimizer-compatible manifests, numerical
reports, frontend/backend integration evidence and reviewable deployment changes.
Do not swap live current pointer, publish, push or deploy through this plan.
Any later deployment follows the user's applicable authorization and preserves
The Operating Table. Claims and UI limits remain bounded by tested lengths;
longer-context/G6 qualification is a separate step. Full C/max B, optimal effort
or answer-quality improvement cannot be claimed from short pilot success.

Current position: G1-core PASSED; the full codec/runtime candidate is implemented
and its initial failed qualification is preserved. See the separately versioned
[v3 results](codec-feedback-results-2026-11-05.md) for the current numeric gate.
The separate [1024 numeric extension](codec-1024-results-2026-11-05.md) now passes
108/108 cases. [Backend slices](backend-slices-2026-11-05.md) implement role inputs,
provenance, clean construction, reference/live separation, losses/accumulation
and checkpoint/export/resume. Next are genuinely reviewed pilot integration,
task-specific episode producers and semantic/promotion fixtures. Full A4 actual
training-memory qualification is also outstanding. No reviewed-data release,
capability fitting, frontend or deployment gate is passed. Live services and
production capability checkpoints remain unchanged.
