# Promotion matrix v1

Status: recorded design response to Astra F8, not implemented, independently
re-reviewed, populated with approved cases or GPU-validated. Counts/thresholds
are starting acceptance settings, not measured performance or optimal values.
No coding, model execution, training, service change, commit or deployment is
authorized. Future model workloads remain on the explicitly selected RTX 3090.

Owner: evaluation slices, minimum coverage, denominators, effort metrics, phase
promotion and regression protocols. Specializes [data/readiness](training-data-and-readiness.md)
and [A4](training-run-profile-v1.md). Follow [input rules](training-input-rules-v1.md),
[clean construction](clean-test-construction-v1.md), [numeric verification](numeric-verification.md),
[data operations](data-operations.md) and [specification](architecture-and-training-plan.md).

## 1. Fixed collections

Slice labels below reserve names; actual approved IDs/hashes belong in a future
manifest. Family counts across slices cannot be added as independent evidence.
Related variants stay in one split. Keep existing 1000/2000 base-case totals.

| Slice | Development | Locked | Minimum distinct families, dev/locked |
|---|---:|---:|---:|
| RF: reference semantic recovery | 100 | 100 | 25/25 |
| W: writer | 70 | 100 | 20/30 |
| E: receiver | 130 | 130 | 26/26 |
| R: isolated reader | 100 | 130 | 26/39 |
| M: mechanical lifecycle | Secondary protocol below | 40 | 20/20 |
| X: semantic interventions | 200 | 500 | 50/100 |
| F: free complete jobs | 400 | 1000 | 100/250 |
| Total base cases | 1000 | 2000 | |

Development W+E=200. Locked component total is RF100+W100+E130+R130+M40=500.
Development M uses 40 predefined views derived from 20 W cases. These are extra
test EXECUTIONS, not extra independent examples. Record actual view definitions
and assertions before evaluation. Mechanical cases exercise the named control,
resource and invalid-binding contracts; they carry no fabricated semantic gold.

## 2. Operation/status and role allocation

Thirteen supported cells:

| Operation | Supported statuses |
|---|---|
| Noul | Assessed, Insufficient, Conflicting, Unresolved |
| Choice | Those four plus NoFeasibleChoice |
| Score | Assessed, Insufficient, Conflicting, Unresolved |

E: exactly10 cases/cell=130 in each split, each cell >=5 families/>=2 domains.
Development R: seven/cell plus three extra Assessed/operation:13*7+9=100;
each cell >=3 families/>=2 domains. Locked R:10/cell=130, each >=5 families/
>=2 domains. Reference cells are determined by adjudicated truth, not predictions.

W Noul/Choice/Score counts are24/23/23 development and34/33/33 locked. Each
operation covers >=10 families/>=2 domains. W scores formation, never receives
its intended outcome as input. No per-status writer quota is implied.

RF has10 cases for each of10 semantic fields: Question, Target, Alternative,
five Rubric fields, Attribute key, Reason. Each field has >=5 distinct reviewed
meanings. Absent midpoint is checked separately as structure, not pretend gloss.

## 3. Intervention allocation

X contains100 development pairs/250 locked pairs. Each category below has10/25
pairs, >=5/10 families and >=2 domains:

1. Question polarity or quantifier scope.
2. Changed criterion.
3. Changed evidence, including material omission/conflict.
4. Hidden-fact twins.
5. Incidental candidate order/identity changes.
6. Candidate facts, feasibility, ties, unknown versus infeasible.
7. Rubric direction.
8. Rubric anchors or conditions.
9. Clean reader return interventions.
10. Full-evidence conflicting-return interpretation.

Each pair requires both members AND intended relation. Applicable reference
contrasts reuse categories1/2/5/7/8:50 development groups under reconstruction
protocol,125 locked counterparts if evaluated. Reference-phase contrasts test
recoverable meaning, not untrained Qev interoperability. Construction violations
invalidate a clean claim even when its behavioral answer happens to be correct.

## 4. Acceptance matrix

Apply rows separately to named protocols, never pool clean/transfer/guided/free.

| Gate | Acceptance |
|---|---|
| Reference recovery | >=95% overall; >=90% each semantic field |
| Reference contrasts | >=90% groups overall; >=80% each applicable category |
| Writer structure/exact attributes | >=95% completely correct cases |
| Writer meaning / receiver semantics | >=90% overall; >=85% each operation, separately |
| Status success | >=95% overall; >=90% each operation/status cell |
| Fabricated assessed values | F<=floor(N/100) for N reference-non-assessable cases, separately per operation/protocol |
| Assessed Score agreement | >=90% within frozen acceptance interval |
| Isolated reader | >=95% overall; >=85% each operation/status cell |
| Semantic interventions | >=90% groups overall; >=80% each category |
| Guided/transition complete flow | >=90% under declared guidance |
| Free complete flow | >=90% all scheduled eligible jobs |
| Free internal-call semantics | >=85% each operation, all actual attempts |
| Execution/binding/accounting/clean construction | Zero observed violations |
| Numeric/gradient checks | Applicable A3/F6 checks independently pass |

Success needs ceil(threshold*N) correct. Thus10-case status cell needs9, five-
case cell needs5; fabrication denominator1..99 permits zero,100 permits one.
Zero denominator means incomplete, never a passing rate. Mechanical invariant
enforcement can pass on rejected malformed input while formation still fails.

For point-labelled task Score freeze interval[point-.05,point+.05] clipped[0,1].
Approved interval-only labels keep their reviewed interval, maximum width.20.
No post-prediction widening. Approved point .30/effort examples remain drafts
until independent review. Predictions are delivered clamped values, required
finite/present/correctly bound; missing/refused/non-assessed output fails an
Assessed Score case. No fabricated exact targets are added for non-assessed cases.

## 5. Units and failure handling

| Metric | Denominator/failure rule |
|---|---|
| Complete flow | Every scheduled eligible job; implementation error, overlength, rejected request, unusable answer fail |
| Writer | Every scheduled writer case; missing/malformed request fails |
| Receiver | Every scheduled receiver case; wrong abstention/value/unusable output fails |
| Free operation | Every actual attempt selected by that operation gate, including abort/reject/malformed |
| Status success | Scheduled E cases or actual semantic attempts in named protocol; unscorable request meaning fails aggregate |
| Per-status accuracy | All cases with adjudicated reference status, including missing output/wrong abstention |
| Fabrications | All reference-non-assessable cases; delivered numeric/selected value counts regardless of claimed status |
| Score | Every reference-Assessed Score case; missing/nonfinite/refused/non-assessed output fails |
| Reader | Every scheduled reader case; invented support fails |
| Interventions | Every scheduled group, all members/relation required |
| Mechanical | Every scheduled case, individual assertions reported |

Unscorable latent meaning is an explicit bucket, not an invented status. It
fails aggregate status success/call semantics instead of disappearing. Conditional
per-status rates retain their adjudicated denominators; report unscorable counts.

A correct final answer after wrong internal decision is a recovered answer, not
complete-flow success. Report final-answer and full internal success separately.
Valid direct answers need no call. Freeze acceptable route criteria beforehand;
unvalidated reconstructed gloss alone cannot establish actual latent meaning.

Verified external interruptions have a separate stratum; unresolved interrupted
fixture prevents complete promotion. Implementation-caused failures remain
failures. Diagnostic reruns cannot replace them. Completion after an external
user stop needs resumed authorization; do not silently replace the fixture.

## 6. Unforced coverage

Require these in the SAME fixed F collection:

| Coverage | Minimum |
|---|---|
| Successful unforced calls | 50 Noul,50 Choice,50 Score |
| Each operation's diversity | >=25 jobs,>=20 families,>=3 domains;>=10 successful calls in each of three domains |
| Each operation/status cell | >=5 adjudicable attempts from>=3 families/>=2 domains |
| Successful direct jobs | >=100 from>=50 families/>=3 domains |
| Multiple-valid-call jobs | >=25 development/>=50 locked |
| Correct prior-result-reuse jobs | >=10 development/>=20 locked |
| Valid answer-phase calls | >=20 development/>=30 locked;>=10 families,two operations,two domains |

Successful coverage calls require correct request meaning, decision, binding
and native use. All unsuccessful attempts stay in accuracy denominators.
These are collection-level requirements, never per-prompt mandatory routes.
Missing coverage is incomplete; do not force gates or append favorable cases.

Use documented domain order: refund, scheduling, demo preparation, access,
resource allocation, issue prioritization. F domain counts are67/67/67/67/66/66
development and167/167/167/167/166/166 locked. No operation hints, runtime domain
routing, supplied reasoning, gold returns or budget overrides in F evaluation.

## 7. Effort metrics

Effort is separate from task Score. Reserve200 development/400 locked F prompts
with reviewed POINT effort targets,50/100 in each band: t=0;0<t<=.10;
.10<t<=.50;.50<t<=1. These are reporting bands, not model budget classes.
Use >=50/100 distinct families overall and>=10/20 per band. Review genuine
preferences before predictions; never invent/relabel targets to fill coverage.

Let s=delivered clamp(raw),t=reviewed effort target,B(s)=exact A3 S-curve.

| Metric | Initial gate |
|---|---|
| MAE(s,t) | <=.05 overall;<=.075 each band |
| P90 absolute error | <=.10 overall |
| abs(mean(s-t)) | <=.02 |
| abs(mean((B(s)-B(t))/131072)) | <=.03 overall;<=.05 each band |
| t=0 behavior | >=90% get B=0 exactly |
| Availability | Valid finite readout for EVERY scheduled effort case; missing prevents pass |

P90 is nearest-rank sorted error at index ceil(.9*N). Compute B(t) using same
canonical FP32-scalar/ordered-FP64 contract as prediction, recording canonical
target scalar bytes. Effort training remains raw regression; these evaluate
delivered clamped behavior, with raw range/clamp saturation separately reported.

Report requested allowance, effective grant, consumed thinking positions; mean/
median/P90/max allocation; positive excess/shortfall versus target; thinking-
phase/answer-phase decision expenditure; and unused allowance separately from
executed computation. Actual B counts include all charged latent work; native
answer tokens are A and control overhead H. Use requested B for preference error,
not context-capped grant that can conceal excessive requested allowance.

Compare fixed always-zero and training-label-mean baselines diagnostically; no
locked-label baseline tuning. Unused allowance is not automatically wasted
compute. These gates measure adherence to reviewed preference, not minimal
necessary reasoning, optimality, answer improvement or token savings.

## 8. Phase protocols and regressions

| Phase | Required development protocols |
|---|---|
| Reference | RF, applicable reference contrasts, structural/numeric checks |
| Components | W,E,R,M,effort,replay |
| Guided joint | Component regressions,X,declared guided complete flow,effort,replay |
| Transition | Same, with checkpoint's declared shrinking guidance |
| Free control | Unforced complete flow,all free coverage,X,component regressions,effort,replay |

Guided/transition complete-flow evaluation reuses400 F cases under separate
protocol, not more independent cases. Use predicted slots/actual Qev returns,
generate native text. Log exact guidance and permitted early guided override.
Transition uses recorded prefix fraction without budget override; no incompatible
guidance pasted back after divergence. F protocol always fully unforced.

Native replay NLL_current<=1.05*NLL_baseline. Use fixed100 direct-answer D-F cases
under matched explicitly guided native-token diagnostic, separate from free
evaluation. Baseline is after reference freeze/before component updates,R at
initialization. Decision replay uses fixed130 E cases/matched reference-packet
interface:>=90% after components, subsequently>=max(.90,component_accuracy-.02).
Component reference is promoted final component checkpoint. Freeze cases,
interfaces/masks; claims do not imply complete original-Qev preservation.

Evaluate every100 updates, finish prescribed block and pass at BOTH final
checkpoints U-100/U. These repeat the same development families, not independent
confirmations. Final promoted checkpoint proceeds to locked evaluation. Failure
requires versioned development revision, not extra epochs, favorable earlier
checkpoint or relaxed gate. Retire locked material to development if used to tune.

Report family counts and95% family-bootstrap intervals with2000 resamples,seed17.
Resample eligible families with all their group members together, within each
reported slice. Gates use point estimates; report uncertainty and correlation.
Do not imply confidence-bound gates or independent calls from repeated variants.

Actual reviewed fixtures, baseline captures, independent re-review and GPU
validation remain absent. Short-context promotion does not establish full C or
maximum B reliability; G6 length qualification remains separate.
