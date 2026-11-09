This is a conditional design. The first gate is whether Qwen3.5-0.8B can switch Qev on and off while preserving one correct causal state, including any recurrent state and attention caches. That must be established before semantic fitting. Model support, adapter-switching behavior, long-context support, and GPU kernel compatibility are assumptions to verify.

1. Requirements and chosen components

| Required behavior | Proposed component and freely chosen mechanism |
|---|---|
| One user prompt, one inference job, one causal history | A single chronological model execution with mode-labelled positions; no prefix replay or independently prefilling a decision context |
| Qev active for allowance prediction and decisions; inactive for native text | Explicit execution modes controlled by the runtime, with independently versioned native and decision extensions |
| Learned invocation timing and request contents | Native action head plus variable-length latent request formation, trained with autonomous trajectories |
| Vector boundary with recoverable task details | Typed vector sequences, immutable source-span handles, stable alternative identities, explicit scalar metadata, and bounded lossless-content checks |
| Noul, Choice, and Score semantics | Separate typed output heads sharing a decision representation; operation-specific assessed values and common semantic-status heads |
| Missing, conflicting, unresolved, and infeasible outcomes | Explicit status enum and bound reason slots; legitimate absence masks prohibit fabricated value targets |
| Result-sensitive task continuation | Native result-consumption training with matched result interventions and source-access restrictions |
| Immutable requests and safe delivery | Sealed request envelopes, monotonically scoped identities, job epochs, exactly-once consumption, and structural validation |
| Continuous initial allowance in an LM-head row | One extension-owned LM-head regression row, independent of the task Score head |
| Exact accounting and cancellation | Reservation ledger and position-by-position execution ledger, coupled to request/job lifecycle |
| One selected RTX 3090 | GPU-only model execution with measured memory and throughput gates |
| Finite, auditable training | Fixed datasets, ordering, exposure, owners, checkpoints, and acceptance thresholds established before each stage |

The architecture, adapter placements, loss functions, proposed thresholds, and stage ordering below are design choices. They are not facts about existing artifacts.

2. Architecture and data flows

**Owners and modes.** Preserve the pretrained base and original Qev release byte-for-byte. Store all extensions separately:

- A native adapter and native action/request heads, active only in native mode.
- An explicitly named aligned Qev derivative, containing the original release identity plus separately stored trainable deltas and decision heads.
- Input/output vector codecs.
- An extension LM-head row for the allowance regression.
- Runtime schemas and resource-policy versions.

A checkpoint manifest identifies each owner and its frozen or trainable status. “Qev inactive” means none of its adapter operations execute during native positions. It does not mean erasing causal state previously produced during decision positions.

**One history.** Prefill the user prompt once with Qev active to predict the allowance. Continue through native and decision segments in one chronological state. Previous state is preserved when adapters change. This defines a dynamically routed causal model; it does not claim equivalence to running either fixed adapter over the entire prefix.

Before fitting, compare the intended production engine against a simple, GPU-executed reference implementation that advances one position at a time with explicit mode selection. Verify all attention caches, recurrent states, adapter routing, and cancellation behavior. Engine support for this exact execution model is unverified.

There is also a first-answer bootstrap issue: an activation produced during Qev prefill cannot simply be called an inactive-Qev native answer computation. The proposed path uses a fixed native answer-entry control position processed with Qev inactive. Its cost is recorded and reserved. At zero *extra* thinking allowance, the model enters the ordinary answer path immediately.

This is feasible only if mandatory answer-entry/closure control positions may occupy the expressly reserved control capacity. They must remain visible in total internal-position accounting. If “zero thinking” forbids even this mandatory transition, the bootstrap requires a different demonstrated mechanism; this is a requirement conflict to resolve before fitting, not an accounting exemption.

**Native requests.** In native mode, a learned action head selects continuation, Noul, Choice, or Score. Request formation emits latent slots, typed pointers, scalars, and termination metadata. No generated question strings or textual tool calls cross the boundary.

Requests contain:

- Operation, request identity, and schema version.
- Proposition/criterion/target/rubric representations.
- Evidence references and relevant condition representations.
- For Choice, an immutable ordered bank with stable candidate identities.
- Source-span handles for names, quoted contents, dates, quantities, and other exact details.
- Explicit evidence and rubric versions.

Use variable-length source-grounded representations rather than assuming a small fixed vector can encode arbitrary exact content. Original token tapes remain immutable on the native side. Stable handles allow exact selected contents to be recovered without asking the decision model to regenerate them. Decision evaluation receives vector representations of relevant content; source storage is not itself a free model evaluation.

A new vector processed by the model consumes a position. Reusing a stored reference without model execution consumes storage and host work, but not a model position. Copying referenced information into a new model segment consumes the actual positions executed.

**Decision execution.** Seal the request before activating Qev. Append its vector segment to the existing history and execute Qev-active decision positions. Heads return:

- Semantic status.
- An assessed value only when status is Assessed.
- Bound reason vectors and references to relevant evidence or constraints.
- For Choice, a stable selected identity plus a bound handle to usable contents.

The result envelope also carries request identity, request digest, bank/rubric versions, job epoch, and execution metadata. These are structural bindings, not semantic corrections.

Noul predicts the proposition’s truth probability. Negating a proposition changes its meaning; approval is not the universal interpretation of truth. Score predicts the exact supplied rubric’s continuous value. Choice distinguishes any-feasible questions from certified-best questions and does not equate ranking with feasibility.

**Native consumption.** Return to native mode and inject the typed result through the result codec. The native model chooses whether to answer, qualify, clarify, request alternatives, continue reasoning, or make another serial request. One request may be pending; nesting and asynchronous calls are unsupported.

The native model receives no evaluation gold answer or withheld fact. Its response must depend on delivered result contents, status, reasons, and accessible source facts. Runtime deduplication prevents duplicate delivery of a result; learned behavior must prevent repeating unchanged requests as if repetition generated evidence.

**Lifecycle.** Requests move through formed, sealed, running, completed, delivered, or cancelled states. Delivery requires matching job epoch, request identity, immutable digest, and an unconsumed result. Cancellation atomically revokes delivery and resumption rights. GPU work already launched may finish, but remains charged and its output is discarded. New input creates a queued job or explicitly cancels the current one.

Structural errors, semantic statuses, cancellation, and execution/resource failures use separate types.

3. Allowance and accounting

The budget row produces a raw scalar from the Qev-active prompt representation, without softmax or categorical averaging. Its weights belong to an extension-owned actual LM-head row; the original vocabulary rows remain preserved.

Cast the raw value to FP64, clamp it to obtain \(s\), and evaluate the written expression with explicit ordered operations. For left-associated multiplication:

```text
a = fp64(3) * s
a = a * s
b = fp64(2) * s
b = b * s
b = b * s
u = a - b
v = u * fp64(131072)
B_pred = floor(v + fp64(0.5))
```

Disable reassociation and fused substitutions that change this operation order. Boundary tests must establish the intended arithmetic, including rounding transitions.

Report independently:

- Predicted allowance, `B_pred`.
- Any explicit override and its source.
- Granted extra-thinking allowance.
- Mandatory control reservation and actual use.
- Answer reservation and actual use.
- Total executed positions and additional processing.

For context capacity \(T=262144\), reserve prompt positions, answer capacity, necessary controls, and any already committed work before granting optional thinking. General task Score never changes this grant.

All reasoning, request formation, decision evaluation, and feedback positions debit the thinking ledger; failed and aborted attempts do too. Mandatory control costs are separately declared, never hidden. The answer ledger is separate and the thinking ledger does not reset at answer start.

A call may start only after reserving a conservative, finite upper bound covering its complete request, evaluation, feedback, and required closure. The generation protocol imposes maximum segment lengths and enforced termination. A call that cannot fit is unavailable; the native model must complete using its remaining answer resources. Any recomputation is separately measured and charged rather than disguised as cache reuse.

4. Milestones and proof obligations before fitting

**Milestone A: execution feasibility.** With engineering fixtures only, demonstrate:

- Correct single-history switching, including recurrent-state handling.
- No adapter leakage into inactive positions.
- Native bootstrap and normal termination at zero extra thinking.
- Exact accounting, reservation failure, duplicate/stale return rejection, and cancellation.
- The actual LM-head row and ordered FP64 allowance calculation.
- GPU-only execution on the explicitly selected device.

Forced fixtures establish machinery, not learned autonomous capability.

**Milestone B: semantic interface adequacy.** Establish recoverability of source identities, order, names, quantities, evidence scope, negation, conditions, and rubric direction. Test long and confusable alternatives. A bridge that reconstructs topic while losing a date or constraint fails.

**Milestone C: supervision readiness.** Audit reviewed labels and splits. A schema is not evidence that labels exist. Missing status, reason, rubric-interior, feasibility, budget-utility, or autonomous-trajectory supervision blocks the corresponding stage.

**Milestone D: resource feasibility.** Measure representative GPU steps and full required inference lengths. Freeze the implementation, memory strategy, and exposure plan only after these measurements. Short-context training does not prove 262144-position inference or semantic retention.

5. Staged training

Before every stage, freeze its dataset manifest, family split, ordering, exposure, objective weights, owners, maximum lengths, checkpoint cadence, and acceptance test. A conservative initial exposure rule is one predetermined traversal of the approved examples or trajectory windows, with no automatic extra pass. Dataset size and maximum positions become explicit numeric caps before authorization.

Save complete-window model, optimizer, scheduler, gradient-scaler, RNG, sampler, accumulation, and exposure-ledger state. Recover only from a consistent completed window, preserving deterministic ordering. Hash frozen owners before and after each stage. Execution completion and learning acceptance are separate outcomes. Evaluate the predetermined terminal checkpoint; do not select a best checkpoint.

**Stage 1: vector interface and decision semantics.**

Needed supervision includes reviewed:

- Request slots, evidence references, exact bindings, and legitimate absence masks.
- Status labels and relevant missing/conflicting/interpretation reasons.
- Noul probability targets with an explicit interpretation and provenance.
- Choice feasible sets, selected acceptable identities, unknown feasibility, no-feasible cases, and any-versus-best criteria.
- Score values attached to exact targets and complete rubrics, including meaningful interior anchors.

A deterministic truth label alone does not provide a calibrated probability target for ambiguous evidence. Specify the target construction—such as reviewed evidence-conditioned probability assessments—and its limitations before training.

Matched examples vary negation, operators, irrelevant missing facts, material conflicts, alternative order, availability knowledge, rubric anchors, and rubric direction.

Train the codecs, aligned Qev deltas, and typed decision heads. Freeze the base, original Qev artifact, native extension, and allowance row. Use status losses, assessed-only value losses, binding/pointer losses, and reason grounding losses. Mask all absent values. Choice loss may accept a reviewed set of equally valid identities; it must not invent a total preference order.

Gradients may flow through trainable codecs and the aligned decision extension. No downstream native answer target supplies the decision label. Require semantic acceptance before dependent native training.

**Stage 2: initial allowance.**

Train only the actual allowance LM-head row using continuous regression. Freeze the decision extension and all other owners.

Budget labels require reviewed utility-versus-cost evidence from bounded GPU trials or another explicitly justified source. They must distinguish useful thinking from mere output length and include zero-useful-thinking cases. This supervision is currently an assumption, not an available dataset. Its acquisition requires separate future authorization.

The deployed calculation remains the required scalar transformation. There are no budget bins, and task Score is not an allowance label.

**Stage 3: native request formation and result consumption.**

Freeze the accepted decision system and allowance row. Train only the native adapter and native action/request/consumption heads against the accepted interface.

Required supervision includes reviewed trajectories with:

- Legitimate no-call behavior.
- Invocation timing and request contents.
- Correct operation and source bindings.
- Meaningful serial calls with changed evidence or different questions.
- Responses to each semantic status.
- Result-conditioned continuation without inaccessible facts.

Train action timing, latent request slots, pointers, and native continuation. Decision execution is frozen and gradients stop at its boundary. Native request losses explicitly align outputs with the fixed request space; final answer gradients alone cannot establish request meaning.

Use matched result interventions: change an assessed value, selected candidate contents, a material missing fact, conflict reason, or rubric direction while preserving the accessible context. Review which continuation changes are justified. Include interventions where the model should question an inconsistent result using visible evidence. Do not manufacture a gold answer from the injected value.

Teacher-forced and forced-call examples can bootstrap behavior, but cannot satisfy autonomous acceptance.

**Stage 4: finite autonomous trajectories.**

Run a preregistered finite rollout schedule with native decisions enabled and the decision system frozen. Each trajectory has fixed context, call, and position limits. Reviewed supervision must cover the branches actually reached; unreviewed branches cannot silently receive guessed labels.

Any reinforcement-style objective needs independently reviewed task utility and cost terms. Rewarding agreement with the system’s own decision head is circular. If trustworthy branch supervision is unavailable, this stage is blocked.

Do not add passes, search architectures, or deploy after failure. Report the failure and propose a separately authorized next experiment.

6. Held-out acceptance and regressions

Hold out semantic families, not just random paraphrases. Include unseen wording, operators, numerical ranges, names, bank lengths, and rubric formulations. Separate development diagnostics from the final test. Never fit on final cases or choose checkpoints using them.

Predeclare numerical thresholds and sample sizes before fitting. Suggested starting gates, subject to a justified statistical plan, are:

- At least 95% joint status-and-delivered-content correctness for typed decisions.
- Noul Brier score better than training-defined constant baselines by a declared margin, with calibration and negation-consistency results.
- Score assessed-value MAE at most 0.10, alongside status accuracy, interior-anchor tests, and rubric-direction interventions.
- Choice identity/order invariance and feasibility-retention performance at least 95%.
- At least 90% fully correct autonomous trajectories across no-call, one-call, and serial-call families.
- Ordinary answering noninferiority within a preregistered margin, such as two percentage points.

These numbers are proposed design gates, not evidence of achievable performance. Report confidence intervals and family-level results. If label reliability or sample size cannot support a claim, mark it untestable.

Acceptance must cover:

- Joint semantic status and value/content; never assess values only on convenient successful-status subsets.
- Noul constants fitted only from training labels, including prevalence-aware baselines; behavior near 0.5 and preservation of proposition polarity.
- Score zero, 0.5, continuous interiors, anchors, conditions, and direction.
- Choice stable identity under permutation, usable selected contents, ties, unknown availability, no-feasible banks, and certified-best requirements.
- Bound reasons, irrelevant missing information, conflicts, and unresolved interpretations.
- Native autonomous request formation, legitimate absence of calls, and useful serial calls.
- Content-sensitive result consumption without invented facts or blind obedience.
- Exact arithmetic, total accounting, cancellation, stale returns, closure, and answer retention.

Keep every case, timeout, refusal, malformed request, and resource failure in the denominators. For matched interventions, report both-correct pairs, not merely average performance on each side. Engineering fixtures and forced requests appear separately from autonomous semantic results.

7. GPU design, measurements, and decisive blockers

Select the RTX 3090 explicitly by stable device identity and assert every model parameter, activation, model input, and fitting operation is on it. CPU orchestration, storage, tokenization, and static checks remain allowed; no CPU inference, embedding extraction, probes, or model fallback.

Start with a frozen reduced-precision base and small trainable extensions. Verify actual dtype/kernel support rather than assuming it. Use GPU activation checkpointing where required, appropriate fused kernels, packed training windows, pinned asynchronous transfers, and preprocessed training data. Microbatch size and accumulation are chosen from measured memory before fixing the stage.

Memory accounting includes:

\[
M_{\mathrm{peak}} =
M_{\mathrm{base}} + M_{\mathrm{extensions}} + M_{\mathrm{gradients}}
+ M_{\mathrm{optimizer}} + M_{\mathrm{activations}}
+ M_{\mathrm{cache/state}} + M_{\mathrm{workspace}}.
\]

Measure allocated and reserved peaks plus device-level usage. For attention caches, calculate actual layer counts, head dimensions, dtype, and sequence lengths; include recurrent-state storage separately. Long-context training may be impractical even when short-window extension training fits.

Profile representative steady-state complete windows with CPU/GPU timelines. Record meaningful examples and supervised positions per second, model-executed positions, transfer time, kernel time, host dispatch gaps, synchronization, checkpoint overhead, and end-to-end elapsed time. A saturated host thread that leaves GPU idle gaps fails the objective. Address measured causes through batching, preprocessing, synchronization removal, or dispatch changes—not utilization cosmetics or reduced workload.

For every optimization, compare outputs, losses, gradients, masks, and updates against the accepted reference within declared tolerances. Language changes alone are not a performance argument.

The smallest decisive experiments, in order, are:

1. **Mode/state experiment:** a short GPU trace alternating native and Qev positions, compared against explicit stepwise reference state. Failure blocks the architecture.
2. **Zero-thinking bootstrap experiment:** produce a normal native answer after budget prediction with all transition costs exposed. Failure identifies the exact allowance/control interpretation or engine capability requiring change.
3. **Capacity experiment:** one bounded full-required-length inference job, measuring actual cache/state memory and accounting. Failure means the 262144 capacity requirement needs a different verified GPU execution strategy or an approved product change.
4. **Interface experiment:** reviewed contrast pairs differing only in exact identity, negation, quantity, condition, or rubric direction. Failure blocks semantic fitting at scale.
5. **Small decision-learning experiment:** a finite reviewed set containing all statuses and assessed interiors, compared with constant baselines. Failure blocks native training.
6. **Small autonomous experiment:** held-out no-call and serial-call tasks plus result interventions. Forced success does not pass.
7. **Throughput experiment:** representative steady-state training windows with a complete timeline and memory ledger.

The main unknowns are switching correctness, bootstrap semantics, maximum-context memory, sufficient reviewed supervision, latent-interface fidelity, and whether this foundation has enough capacity for reliable autonomous formation and consumption. None should be assumed away. Any proposed relaxation—shorter context, changed mode semantics, textual boundaries, external judges, fewer operation semantics, or different hardware—requires explicit user approval.

Provenance: I used only the supplied blind assignment and general technical knowledge. I used no tools, repository files, implementation history, external sources, or other agents.
