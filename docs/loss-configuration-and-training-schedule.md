# Loss configuration and training schedule

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

Owner: losses, masks, normalization, gradient ownership, sampling and forcing.
Follow [semantic supervision](semantic-supervision.md),
[writer](latent-writer-schedule.md), [codec](latent-packet-codec.md),
[cache/gradients](cache-and-gradient-strategy.md),
[data/readiness](training-data-and-readiness.md) and the
[implementation specification](architecture-and-training-plan.md).

The recorded [A4 run profile](training-run-profile-v1.md) specializes phase
lengths, optimizer groups/schedule, exact field losses, correction allocation
and development promotion. It is not a trained checkpoint or executed manifest.

## 1. Normalize before weighting

For each component, average eligible tokens/fields/decisions within an example,
then eligible examples, then apply its coefficient. Normalize semantic losses
across roles so many alternatives cannot automatically dominate one question.
Log eligible counts and weighted gradient checks. No eligible targets means
zero contribution, not an invented target. Across accumulation, use eligible
counts for the whole window rather than equal weighting of unequal microbatches.
The [exact module/optimizer contract](module-and-optimizer-contract-v1.md) fixes
the executable accumulation design: separate component VJPs/buffers, eligible
EXAMPLE denominators, one combined gradient and clip/step, no count-first replay
or retained multi-episode graphs, and unused gradients kept as None.

Semantic alignment/reconstruction and receiver input use the same declared
canonical payload normalization, not a loss-only normalization with an unchecked
raw-amplitude path. Keep this separate from scalar-row dot-product arithmetic.

## 2. Starting joint/component coefficients

| Component | Loss | Weight |
|---|---|---:|
| Native answers, isolated reading and language replay | Next-token cross-entropy | 1.0 |
| Operation gate | Supervised-action CE or permitted-set probability loss | 1.0 |
| Writer actions, including finish/abort | Cross-entropy | 1.0 |
| Counts, types, modes and presence flags | CE/BCE as appropriate | 0.5 |
| Object/source/span pointers | Pointer cross-entropy | 1.0 |
| Exact attributes | Type-specific classification; exact bytes/scale, not numeric regression | 1.0 |
| Semantic slot alignment | MSE of normalized predicted/reference slots | 1.0 |
| Semantic reconstruction | Token CE through frozen decoder | 0.5 |
| Semantic contrasts | Contrastive loss; starting temperature 0.1 | 0.1 |
| Result status | Cross-entropy | 1.0 |
| Candidate feasibility | Per-entry classification | 0.5 |
| Noul probability | Binary cross-entropy | 1.0 |
| Choice selection | Acceptable-set probability loss | 1.0 |
| Task Score | Reviewed-point MSE or interval-distance loss | 1.0 |
| Initial effort preference | Reviewed-target MSE | 0.5 |

Non-assessed reasons use applicable semantic, binding and continuation losses.
Reason annotations may contain result status/supporting evidence; request-
question/rubric inputs remain answer-blind. Reference artifacts must support a
reason role before freezing; do not silently fit a moving decoder during joint
training. There is no extra reward for calls, avoiding calls, answer improvement
or token minimization. No policy-gradient/discrete straight-through stage is assumed.

The [training-input rules](training-input-rules-v1.md) own role/branch exposure.
Reason-specific targets must match the actual non-assessed carrier and evidence;
mask incompatible reasons without suppressing valid raw-value/status losses.
Guided Reason exercises remain separate from actual joint returns. Log outcome-
bearing native prefixes; they cannot demonstrate independent outcome derivation.

Distinguish selected demonstration routes from all task-valid routes. Imitating
a demonstrated Noul action does not prove a direct answer invalid. Where policy
annotations permit multiple actions, supervise their combined probability mass.
For Choice with acceptable selections V:

    L_choice = -log(sum(p(i) for i in V))

Do not penalize another acceptable tie winner or predict opaque UUID labels.

## 3. Numeric losses and clamping

Train raw Score/effort readouts against reviewed targets; clamp delivered values
to [0,1]. Raw regression retains corrective gradients outside the range even
when downstream clamp gradients vanish. With a reviewed interval [a,b] and no
approved point target:

    L_interval = max(a-r, 0)^2 + max(r-b, 0)^2

If an approved point exists, use its MSE instead; no invented midpoint or
unrecorded double objective. Example .30 severity and .04/.05/.06 effort labels
remain masked until reviewed. Noul uses stable BCE on its truth logit or
true/false logit difference with reviewed binary/probability targets. A low
probability is not an uncertainty status or automatic calibration guarantee.

## 4. Eligibility masks

Use reference validity for the applicable packet, not predicted status. Otherwise
abstention could evade numeric supervision.

| Situation | Supervision |
|---|---|
| Reference Assessed; model abstains | Status plus applicable raw-value losses |
| Reference non-assessed; model emits a value | Status loss; no fabricated numeric target |
| No feasible Choice bank | Status/feasibility; no winner loss |
| Several acceptable Choice winners | Probability mass over the valid set |
| Attribute/reference lacks valid annotation | Mask target and report missing support |
| No valid contrastive pair | No contrastive term for that group |

Do not give malformed/cancelled/exhausted runtime cases fake semantic labels.
Native token loss excludes source prefill, latent inputs, descriptors and
runtime-inserted controls. It covers approved continuation and model-predicted
termination where annotated. Semantic decoder text targets are offline
reconstruction supervision, not extra inference prompts/native answer tokens.

## 5. Rollout divergence

If the writer omits Pine, do not train its pointer to select an absent entry.
Retain formation/binding losses and report the failure. Decision labels must
describe the actual bound packet where its semantics are known; otherwise mask
inapplicable downstream targets without removing failures from reports. A new
rubric cannot inherit the old scalar label.

Correction continuations must suit the actual state: revision, contradiction
recognition, qualification or exhaustion. Reusing the original happy-path answer
blindly teaches ignoring results. An unvalidated decoded gloss is not a truth
oracle for relabelling arbitrary model-generated packets.

## 6. Forcing dimensions and phases

Log decision forcing (operation/writer/reference), semantic-slot substitution,
gold-return substitution, native-token teacher forcing and budget overrides
separately. Zero decision forcing does not mean fully free text. Guided cases
never count as autonomous invocation/result-use evidence.

| Phase | Decision guidance | Semantic packets | Decision returns |
|---|---|---|---|
| Reference codec | Separate component exercise | Role-whitelisted reference encoder outputs | No request outcome labels; permitted Reason status/support is semantic content, not an evaluation return |
| Components | Reviewed actions/references | Reference packets for receiver; predicted slots for writer | Gold allowed in isolated reader exercises |
| Guided joint | 100% reviewed decision prefix | Predicted writer slots | Actual Qev result |
| Transition | Reviewed prefix shrinks to zero | Predicted writer slots | Actual Qev result |
| Free-control joint | 0% forced decisions | Predicted writer slots | Actual Qev result |
| Free evaluation | No forcing or gold substitution | Actual outputs | Actual outputs |

Approved native continuations can still use next-token teacher forcing during
training. Free evaluation generates native text too, without budget override,
supplied boundary prefixes or gold returns.

## 7. Transition and advancement

For a declared transition block of N>1 updates, k=0 through N-1:

    q = 1 - k / (N-1)
    guided_events = floor(q * T)

Guide the first guided_events of a reviewed trajectory with T decision events,
then continue with model choices. Use a contiguous reviewed prefix, not gold
actions pasted back after incompatible divergence. A4 fixes a 2000-update
transition and the other phase lengths; instantiate them in the run manifest.
Advance only when development checks pass as well as the block ending;
decreasing training loss alone is insufficient. Keep locked tests untouched.

Early component/guided training may use an explicit bounded budget override so
an untrained effort row does not prevent call learning. Record it and remove it
before autonomous budget/free-flow evaluation. Changes based on development
evidence require new configuration lineage, not retroactively edited reports.

## 8. Trainability and gradients

Reference preparation freezes W and trains reference pooling/reconstruction
components with reconstruction weight 1.0 and contrastive weight 0.1. Validate
meaning, then freeze/version encoder and decoder.

Component phases update the relevant writer, receiver or native reader paths.
Joint phases update R, aligned D, runtime protocol components and applicable
readouts together. Frozen decoder computation still propagates gradients to
predicted latent inputs. Continuous task Noul/Score returns stay attached;
hard choices get explicit supervision.

The startup budget has a deliberate stop-gradient on its control-feedback value:
its row learns from effort labels rather than answer loss inventing another
scalar meaning. This does not detach task-result values or shared causal state.
Shared upstream representations can still change through other losses, so effort
supervision remains necessary.

Keep pretrained weights frozen and adapters registered appropriately through
joint unrolls. Step optimizers after complete episode backward; no accidental
detach/global trainability changes at mode transitions. The
[scalar-row design](scalar-row-handling.md) records row isolation, pilot
optimizer settings, precision, text exclusion and export handling.

## 9. Sampling and monitoring

Start with 25% replay and 75% new tasks. Balance operations/non-assessed examples
in component batches using sampling instead of large automatic class weights.
Retain direct-answer trajectories and delegated cases across all three operations.
These are training proportions, not production invocation quotas.

A4 fixes native/decision windows to two replay cases out of eight, with separate
component/joint recipes. The reference-codec phase is a separate role-whitelisted
semantic task without ordinary-language replay. Native teacher forcing is
allowed in free-control TRAINING, while free promotion/testing forces neither
native text nor decisions. Preserve separate reports and denominators.

Keep semantic objectives through joint training. Monitor individual failures,
eligible counts, forcing flags, weighted gradients and regression evidence, not
only summed loss. Tune only on development data, version changes and preserve
the distinction between guided skills, connected-flow training and autonomous
functional verification.
