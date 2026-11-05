# Training data scope, annotation procedures and readiness criteria

See the [current handoff](next-agent-handoff.md) for implementation status and
measured evidence. Counts and thresholds below are proposed acceptance targets,
not evidence that the architecture works. Resolve annotation ambiguities before
locking the test set; never choose thresholds after observing its outcomes.

Owner: data quantities, splits, scoring and semantic acceptance. Exact annotation,
review/adjudication and provenance rules belong to [data operations](data-operations.md);
the overviews below do not define a competing procedure.
The [implementation specification](architecture-and-training-plan.md) owns
cross-component requirements and the [handoff](implementation-handoff.md) orders
contract freeze and short GPU feasibility before data scale-up.

The recorded [promotion matrix v1](promotion-matrix-v1.md) specializes all
evaluation allocations, coverage, denominators and gates below. In particular:
development W/E=70/130; locked components=RF100/W100/E130/R130/M40; explicit
13-cell status minima/floors, point-Score acceptance intervals, successful
unforced/family coverage and effort-error gates apply. Its aggregate status-
success gate counts unscorable attempts as failures; conditional status accuracy
still reports adjudicated cells. These settings override earlier unspecified
minima/operational details, not evidence that any fixture or gate has passed.

## Scope and starting quantities

Start with fictional, self-contained tasks in refund eligibility, scheduling,
demo preparation, access rules, resource allocation and issue prioritization.
All required facts and policies must be supplied. Each domain should support
multiple operations and direct answers; do not learn domain-to-operation rules.

| Training material | Initial examples | Purpose |
|---|---:|---|
| Controlled decision exercises | 2000 | Question semantics, operation outcomes and statuses |
| Latent writer and native reader exercises | 1500 | Meaningful requests and bound-result interpretation |
| Complete interleaved trajectories | 2000 | Startup through request, evaluation, continuation and answer |
| Direct-answer trajectories | 500 | Continuing without a decision call |
| Ordinary language and decision replay | 2000 | Preserve native and original decision capability |

The first four groups total 6000 core examples; replay brings training to 8000.
These are starting quantities, not a guarantee of adequacy. Expand based on
development failures, with lineage and provenance preserved. Review a pilot of
about 120 distinct scenarios across the domains before producing the full set.
Pilot variants are correlated examples, not additional independent scenarios.

The recorded [A4 run profile](training-run-profile-v1.md) splits replay into
1000 language/1000 reviewed decision examples and allocates 400 EXISTING core
examples to reviewed corrections (100 writer/reader, 300 trajectories). It adds
no new independent scenarios by repeating examples across 7000 update windows.

Include negation, changed criteria, irrelevant/missing/conflicting evidence,
valid/invalid/uncertain alternatives, ties, permutations, exact attribute changes,
rubric direction and anchor changes, multiple calls, earlier-result reuse,
answer-phase calls, zero allowance, rejected calls, exhaustion and cancellation.

Keep the original three natural prompts and close paraphrases out of training
and tuning. Operation hypotheses for those prompts are not required routes.
They were discussed extensively during design and are illustrative rather
than fresh independent held-out evidence.

## Annotation record

Keep the ordinary human prompt separate from inspectable training annotations:

- Intended semantic question, independent of the desired verdict.
- Exact evidence/source bindings, omissions and material conflicts.
- Candidate identities, recoverable attributes and acceptable selections.
- Rubric dimension, conditions, direction, anchors and version.
- Expected semantic status and valid decision payload, if one exists.
- Acceptable continuation: conclusions, qualifications or focused questions.
- Acceptable operations and invocation windows where their prerequisites exist.
- An illustrative continuous initial effort preference.

Annotate acceptable answers and selections rather than one exact sentence or
winner in a tie. Valid direct answers remain acceptable. A supervised selected
route is one training trajectory, not proof that every other route is wrong.
English annotations are offline review material, not generated inference-time
tool text or a second prompt. Latent vector coordinates are learned, not manually
invented labels.

Budget labels are effort preferences, not minimum sufficient or optimal budgets.
The fixed S-curve maps the scalar to allowance. General task Score annotations
remain separate from this startup port.

## Annotation production and review

Apply the [data-operations procedure](data-operations.md): mechanical checks and
one separate outcome-blind model-assisted content review per record, with
automated release approval. No human reviewers or second-review quota are required. No model generator
is required for the 120-scenario pilot; reviewed model-proposed roots retain
their origin under the same review/approval rules. Quarantine critical errors
and preserve provenance/rejected records. No review execution is claimed here.

For deterministic tasks, derive outcomes from explicit scenario rules and
independently check the rules and examples. A current Qev prediction or prototype
answer is never automatically ground truth.

For subjective scores, review against explicit anchors and record an acceptable
interval where appropriate. Fix unclear rubrics or mark genuinely unresolved
assessments; do not invent exact decimal certainty. Before locking evaluation,
set interval-width limits so broad labels cannot make the metric meaningless.
The starting interval-width default is at most 0.20 on a normalized [0,1] scale;
review and freeze it before locked evaluation. Revise unclear rubrics rather
than widen labels after observing predictions. The .30 severity and .04/.05/.06
effort labels in the examples remain review drafts and do not bypass this process.

Use a separate outcome-blind model review for locked semantic tests before
testing; exclude unresolved disagreements. Keep locked families isolated. Record
which model/source produced and reviewed each annotation and which checks ran.
This experiment does not require human review or claim human-certified labels.

Review question fidelity and continuation fidelity separately. A correct final
answer cannot excuse an incorrect question, wrong evidence binding or unrelated
score. Teacher outputs are proposals subject to review, not an oracle.

## Splits and evaluation composition

Reserve 1000 development examples and 2000 locked test examples, separate from
training. Partition underlying scenario/rule/template/rubric families before
creating paraphrases and counterfactual variants. Keep related contrasts together
to prevent leakage. Freeze split manifests, labels, versions and metrics.

The locked set contains 500 component cases, 500 semantic intervention cases and
1000 ordinary human prompts for free generation. Report slice sizes, failure
counts and uncertainty. If inspected test failures drive training, that set has
become development material and needs a replacement for a fresh locked claim.

A4 partitions the 1000 development examples into 100 reference recovery, 200
writer/receiver, 100 isolated reader, 200 intervention cases (100 pairs) and 400
free jobs. Its phase-specific development gates do not replace the locked-set
composition or final readiness gates here. All related variants remain in their
split; no forced coverage or favorable-subset selection is allowed.

## Establishing semantic use

| Test | Required distinction |
|---|---|
| Same evidence, different question | Evaluate the requested proposition rather than a memorized verdict |
| Same question, changed facts | Respond to evidence rather than a verdict encoded in the query |
| Candidate order and identity changes | Preserve selected contents and exact attributes |
| Rubric direction/anchor changes | Interpret the scalar under its actual scale |
| Isolated reader, evidence withheld | Continuation depends on the supplied bound return |
| Full evidence plus inconsistent return | Notice or reconsider conflict, not blind obedience or universal ignoring |
| Stale/mismatched identities | Do not silently accept a result for another request/target/rubric |

Reader interventions are explicitly isolated tests; full-flow tests use Qwen's
own requests and actual Qev results. Teacher-forced actions/returns teach skills
but do not establish autonomous invocation.

Apply [clean test construction v1](clean-test-construction-v1.md) before interpreting
withheld-evidence results. Every supplied tensor/lookup must derive only from its
declared permitted view; contextual production tensors are reported separately
as transfers. Construction failures invalidate cleanliness and remain reported.

## Proposed acceptance gates

Use the [numeric verification owner](numeric-verification.md) for float,
gradient and exact-control criteria. The semantic thresholds below do not
substitute for backend numeric agreement.

| Gate | Initial acceptance target |
|---|---|
| Execution | Zero observed violations of one prefill, continuous state history, GPU-only execution and resource accounting |
| Binding enforcement | Every deliberately mismatched, stale or duplicate return rejected or handled correctly |
| Gradient connectivity | Verified effective gradients through request formation, decision computation and native result consumption |
| Training/serving forward agreement | Within numerical tolerances fixed before locked testing |
| Complete semantic flow | At least 90% acceptable overall; no operation below 85% |
| Result statuses | At least 95% correct; fabricated assessed values in at most 1% of non-assessable cases |
| Continuous Score | At least 90% within independently annotated acceptable intervals |
| Isolated native reader | At least 95% correct on result-interpretation interventions |

Count rejected and failed calls. A rejected malformed request can pass runtime
enforcement while still failing request formation. Count cases requiring an
assessment but wrongly abstaining as semantic failures too.

The 1000 free-generation prompts must demonstrate at least 50 valid, unforced
calls of each operation across multiple domains, and at least 100 successful
direct-answer runs. These are coverage requirements for the test collection,
not mandatory routes for individual prompts. Score all calls, not a cherry-picked
subset. No operation hints, forced gates, teacher-forced reasoning prefixes or
runtime domain routing in the free-generation test.

The promotion matrix fixes scoring units, interval defaults, coverage, effort
and protocols. Before evaluation instantiate approved case IDs/hashes and
case-level acceptable continuation/route judgements under those rules. Numerical
forward tolerances remain A3's. Do not retrofit any rubric to favorable results.

## Scoring units and denominators

Freeze the valid-route catalogue and case-level scoring instructions on
development material before opening the locked test. These definitions prevent
selecting only favorable predictions for the targets above.

| Metric | Scoring unit and denominator |
|---|---|
| Complete semantic flow | All scheduled free-generation jobs, including implementation-caused errors/cancellations and rejected requests; external/user cancellations are a separately declared stratum |
| Operation floor | All scheduled component cases for that operation, plus all actual free-run attempts of that operation reported separately; no pooling forced cases to hide free-run failures |
| Status accuracy | All semantic test calls with adjudicated status labels, including wrongly abstained assessed cases |
| Fabricated assessed values | Non-assessable reference cases reporting a numeric/selected value, divided by all non-assessable reference cases |
| Score interval agreement | All reference-assessed Score cases with reviewed intervals; missing, refused or nonfinite results fail rather than disappear |
| Isolated reader | All scheduled reader cases, including unsupported-detail invention and unusable outputs |
| Intervention success | Predefined contrast groups, with all required changes/invariances satisfied; do not count only the easier member |
| Execution/binding enforcement | Every scheduled positive and deliberately invalid lifecycle fixture; report counts and any violation, not only a percentage |

A successful complete flow requires a grounded acceptable answer, a valid route,
correct binding/meaning for its decisions and appropriate native result use.
A correct answer after an incorrect internal judgment is reported as a recovered
answer, not a fully correct decision flow. Report final-answer and internal-call
success separately so recovery remains visible.

Direct answers need not invoke any operation to pass. Alternate latent routes
must have evaluable semantics under the predefined route rubric; unscorable
calls are unresolved, not silently accepted. Use binding checks and semantic
interventions for fidelity; a decoded gloss/reconstruction probe alone does not
establish latent meaning.

Coverage counts valid unforced calls and successful direct jobs, together with
distinct jobs/scenario families. Repeated calls or paraphrases are not independent
evidence. Report uncertainty by scenario family where cases are correlated.
Underrepresented status/operation slices or missing unforced coverage yield an
incomplete readiness result; never force gates to fill them.

Mechanical-only malformed/cancelled fixtures have no fabricated semantic gold
status; score them in the declared runtime stratum. Preserve ordinary-language
and original-decision replay checks as regression evidence, not a new answer-
improvement study. Do not combine component/reader/guided/free denominators into
one flattering readiness percentage.

## Meaning and limits of readiness

Report component skills, complete supervised trajectories, free invocation and
result use, and longer-context operation separately. Only free invocation plus
verified result use establishes the intended autonomous complete flow.

Passing supports claims within tested domains, lengths and limits. Short
trajectories do not validate the full 262144-position context or 131072-position
maximum reasoning allowance. This is functional verification, not a study of
answer-quality improvement, token savings or optimal budget selection.

All model generation, training, encoding, fitting and verification use the
explicitly selected RTX 3090. No CPU fallback. No workload runs by recording
this plan.
