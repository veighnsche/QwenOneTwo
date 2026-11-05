# Independent GPT-6 Astra document and mathematical review

Review date: 2026-11-05. Reviewer: the explicitly requested GPT-6 Astra agent, performing the substantive review directly without further delegation.

**Verdict: the documented arithmetic is substantially correct, but the specification is not yet a complete interoperable codec or executable training contract.** A narrowly scoped GPU feasibility implementation is a reasonable next engineering step after its fixture-specific choices are recorded and implementation is authorized. General codec interoperability needs the interface corrections below. Full training additionally needs the differentiable-state proof, approved data and concrete evaluation/run artifacts. None of those empirical gates has passed through this review.

All 33 source documents were read: project README and AGENTS, plus all 31 Markdown documents originally in `docs/`, including every walkthrough and original flow. Total coverage was 5,724 lines. The workspace instructions were also read. Source specifications were not edited. No neural model, tokenizer runtime, training, inference, feature extraction, optimizer, GPU test, service or checkpoint was executed or modified. Arithmetic used disposable standard-library scripts, integer arithmetic and `Fraction`; static JSON and installed source files were inspected read-only.

## Severity-ranked findings

Priorities describe what must be resolved before the affected milestone, not a claim that an unimplemented system has already failed in production. There is no manufactured list of bad sums: the published packet constants, example budgets, phase totals, bit interval coverage and B1 numeric examples recompute correctly.

| ID | Priority | Classification | Finding / affected milestone |
|---|---|---|---|
| F1 | P1 | Missing resource invariant; overstrong guarantee | Position/handle admission does not specify a worst-case **new Ref allocation hold** for result/reason creation. Resolve before general runtime admission. |
| F2 | P1 | Underspecified interface | B1 covers all 128 bits, but does not fully define which fields/tags apply to each role. Two implementations can produce different legal-looking inputs. Resolve before interoperability. |
| F3 | P1 | Evaluation-design gap / empirical risk | “Isolated” packets can retain withheld facts in contextual source states or writer-derived slots. Resolve before interpreting isolated tests as evidence of semantic use. |
| F4 | P2 | Underspecified interface | Target attributes are supported, but their exact request placement and return omission/presentation rule are not explicit. Existing return arithmetic is valid only without extra target attributes. |
| F5 | P2 | Missing control contract | Up to 32 additional consumption positions have no specified selection/stop policy; control emission and startup feedback semantics also need a concrete trace. |
| F6 | P2 | Incomplete reproducibility contract | A2 leaves block equations, some head wiring and optimizer ownership unspecified; A4's dynamically eligible accumulation needs an explicit implementation strategy. |
| F7 | P2 | Wording contradiction | Blanket answer-blind/no-outcome reference preparation conflicts with the explicitly required outcome-bearing reason role. Scope the prohibition by role. |
| F8 | P2 | Incomplete promotion contract | Evaluation totals add up, but status coverage minima and several metric slices/eligibility rules remain manifest decisions, not closed acceptance gates. |

### F1 — Reserve new Ref allocations for every permitted completion

Sources: [32-object interpretation](/home/vince/QwenOneTwo/docs/architecture-and-training-plan.md:252), [writer version allocations](/home/vince/QwenOneTwo/docs/latent-writer-schedule.md:162), [A1 admission envelope](/home/vince/QwenOneTwo/docs/typed-value-abi.md:117), [A1 guarantee](/home/vince/QwenOneTwo/docs/typed-value-abi.md:138), [B1 handle reservation](/home/vince/QwenOneTwo/docs/neural-input-encoding-v1.md:275).

A1 explicitly reserves request positions, maximum return positions and 32 consumption positions. B1 explicitly reserves local handle capacity for result/reason objects. Neither gives the corresponding admission inequality for the separate 32-new-Ref limit. General references to checking counts do not specify how much capacity must remain after dispatch, when semantic status is unknown.

Concrete counterexample using only allowed writer actions:

1. WRITER_START creates a question and target: two new Refs.
2. Twenty-eight question revisions create 28 immutable versions: 30 allocated Refs.
3. FINISH seals an allocated RequestSnapshot Ref: 31 allocated Refs, with only 30 preparation positions used (`1+28+1`). B1 has an explicit RequestSnapshot object kind.
4. A minimal Noul request is 13 positions. With sufficient B, `13+50+32=95` positions are reserved; the position and handle caps can pass.
5. An assessed result needs one Result Ref and reaches 32. A non-assessed result additionally creates its Reason Ref and reaches 33, violating the cap after an otherwise admitted call.

If snapshots are not allocated Refs, the same counterexample uses 29 revisions and 31 preparation positions. Defining that distinction is itself necessary. No assumption about candidate/evidence allocation is needed for this example.

**Correction:** define an allocation manifest and require, before dispatch,

`allocated_new_refs + maximum_new_refs_needed_by_any_legal_completion <= 32`.

Reserve that capacity independently of B/C and the 256-handle table. Charge actual allocations once and release unused holds. State whether attachments can create new referenced objects or only point to existing ones; if they can create them, include their maximum too. Also fix when bank/snapshot Refs are allocated and whether bank assembly is one final allocation or creates intermediate immutable versions. The current guarantee of bounded model positions remains arithmetically correct; the stronger all-resource completion guarantee is not established by the written admission envelope.

### F2 — Complete the role-to-field and role-to-tag schema

Sources: [B1 role-selected layouts](/home/vince/QwenOneTwo/docs/neural-input-encoding-v1.md:94), [common fields](/home/vince/QwenOneTwo/docs/neural-input-encoding-v1.md:101), [descriptor applicability](/home/vince/QwenOneTwo/docs/neural-input-encoding-v1.md:196), [tag order](/home/vince/QwenOneTwo/docs/neural-input-encoding-v1.md:261), [closure claim](/home/vince/QwenOneTwo/docs/final-decision-audit.md:78).

All three layouts cover exactly 128 unique bits without a gap or overlap. That is necessary and correct. It does not specify the semantic applicability of every field or tag. “Only their applicable fields,” “sum applicable tag embeddings” and “fields with no defined use” leave implementers to decide applicability.

Examples of unresolved choices:

- On EVALUATE or RETURN_END, does the port repeat the enclosing operation, or equal None? What owner/reference and following-content length apply?
- On CANDIDATE_DESCRIPTOR, is the owner the bank or the candidate? If a bank has handle 3 and its first candidate handle 4, owner=3 and owner=4 yield different metadata and owner embeddings. Primary reference can identify the candidate in both versions.
- Which descriptors/boundaries receive a slot tag? Which payloads inherit an owner or attribute-index tag? The ordering of a sum does not answer which summands exist.
- Header “following content length” coexists with “complete packet length.” Is the first always total minus one, zero, or a role-specific nested length? The descriptor examples do not define the header case.
- Two midpoint slots remain physically present when `rubric_midpoint_present=false`. Their canonical payloads, loss eligibility and tags are not fixed. Zero payload, learned empty-field payload and unconstrained predicted payload are different wire inputs; the last also creates an unnecessary unsupervised channel.
- Historical header/descriptor fields need a field-by-field rewrite rule: which attempt ordinal, sealed flag, lengths, owner and historical flags are preserved versus rewritten? Exact sidecar identity alone does not resolve their neural presentation.

**Correction:** add one normative table for all roles 1–40, specifying payload encoder, every metadata field as constant/derived/absent, tag presence/index source, length definition, and zero/absent behavior. Include one complete Noul, Choice, Score, startup, non-assessed and historical presentation vector specification. First-encounter handle assignment also needs a fixed traversal order through descriptor fields and attachments, including whether unpresented sidecar-only references enter the table. This is a finite schema completion task, not a need to reconsider all role IDs.

Do not label B1 an independently implementable “exact map” until that table and matching fixtures exist. The docs correctly say no golden vectors have passed; the additional issue is that prose still permits different input maps before any numerical test.

### F3 — Remove contextual information from isolation fixtures, not merely from visible text

Sources: [cached source-state presentation](/home/vince/QwenOneTwo/docs/latent-packet-codec.md:81), [source strings](/home/vince/QwenOneTwo/docs/typed-value-abi.md:53), [isolated-packet requirement](/home/vince/QwenOneTwo/docs/semantic-supervision.md:122), [workshop isolated reader](/home/vince/QwenOneTwo/docs/noul-training-walkthrough.md:157), [packing isolated reader](/home/vince/QwenOneTwo/docs/score-training-walkthrough.md:178).

Production deliberately retains the full shared causal history. That is not an error. The issue is the stronger interpretation of an isolated component test.

A cached source hidden state is contextual. If the original prompt says “the badge expired” before an orientation clause, the hidden state selected for the later orientation clause can contain information about the expired badge. Moving only that state into a fresh component context does not make the badge fact unavailable. Likewise, question/target slots produced by the live writer after seeing the full prompt can carry case facts even if the isolated reader is not given the original evidence text. A frozen answer-blind question target is a different artifact from those writer-produced slots. Existing cautions about possible hidden channels are correct, but the isolation construction must operationalize them.

**Correction:** distinguish three fixture types and provenance explicitly:

1. A clean component fixture whose states/slots are encoded solely from its allowed inputs in a separate offline component context, with forbidden facts never present in the encoder context.
2. A realistic production-packet transfer fixture containing contextual source states/live writer outputs, reported as such.
3. The actual one-prefill whole-flow fixture, retaining the intended full history.

For clean isolated readers, use approved answer-blind question/rubric representations and target representations containing only permitted identity/content. Inspect all supplied tensors, including source states, target slots, question slots, candidate slots and reason attachments—not just text columns. Pair clean variants with fact-order permutations and return swaps. Component fixture preparation is already allowed as a separate offline exercise; it does not authorize another prefill inside production.

This is a demonstrated information-path risk, not evidence that an unexecuted model has exploited it. Without the distinction, a passing isolated test would not support the claimed “only declared evidence was available” conclusion.

### F4 — State the target-attribute grammar and return omission rule

Sources: [eight target/candidate attributes](/home/vince/QwenOneTwo/docs/typed-value-abi.md:64), [request grammar](/home/vince/QwenOneTwo/docs/latent-packet-codec.md:66), [return grammar](/home/vince/QwenOneTwo/docs/latent-packet-codec.md:151), [header target count](/home/vince/QwenOneTwo/docs/neural-input-encoding-v1.md:153), [presented counts](/home/vince/QwenOneTwo/docs/neural-input-encoding-v1.md:201).

A1 allows up to eight inline target attributes. The request grammar shows TARGET_DESCRIPTOR, two target slots, then evidence; it does not show where the target attributes occur. Candidate attributes are explicitly included in candidate records. Returns repeat the target descriptor and two slots, while the fixed return formulas have no term for target attributes.

For a target with one int64 attribute, the request needs `13+4=17` positions before evidence. Its exact location and descriptor following-content length need to be fixed. For eight 128-byte UTF-8 attributes, the attribute cost is `8*(3+16)=152`; a Noul request can still fit at `13+152=165`. Echoing those attributes in its return would require `14+152=166`, exceeding 64.

The documented return formulas therefore imply **no automatic target-attribute echo**. This is a coherent design, not a numerical mistake. State it directly: requests place target attributes at a specified point; ordinary returns present zero target attributes unless an explicitly charged, bounded mechanism is defined. Set the presented counts/lengths accordingly while preserving exact target sidecar binding. If full target-attribute echo is desired instead, all return formulas, caps and admission rules must change; it cannot be added as an implementation convenience.

### F5 — Define the optional consumption/control trace

Sources: [additional 32 consumption positions](/home/vince/QwenOneTwo/docs/position-limits-and-budget-accounting.md:82), [gate resumption](/home/vince/QwenOneTwo/docs/latent-packet-codec.md:173), [roles 37–40](/home/vince/QwenOneTwo/docs/neural-input-encoding-v1.md:65), [control layout](/home/vince/QwenOneTwo/docs/neural-input-encoding-v1.md:206), [startup return](/home/vince/QwenOneTwo/docs/latent-packet-codec.md:223).

The 32-position allowance is a cap, not a policy deciding whether to execute 0, 1 or 32 CONSUME_TICK positions. The four-operation gate and seven writer actions contain no explicit consumption-step/finish decision. The docs do not identify another head, a fixed count, or a text/phase rule that decides this. “After return delivery” and “not midway through consumption” also need an exact boundary if consumption includes these extra ticks.

The startup BUDGET_VALUE is described as a continuous feedback value without one definitive statement of whether it is the raw score, clamped score or a transformed granted/overridden allowance. The predictor, override and grant can legitimately differ. Structured remaining counters do not remove the need to define this payload.

**Correction:** specify one minimal transition table, including the source of each decision, the consumed input and its B/A/H debit. A simple initial choice is zero additional consumption ticks unless an explicit learned mechanism is selected; do not silently implement a new controller. Pin BUDGET_VALUE semantics and the effect of overrides. Define the normal/control termination tokens and phase-close mechanics, including which actions require a model position and which are out-of-band. H=64 is a safe upper reserve only because exhaustion terminates; it is not presently a proof that every desired failure/closure sequence fits. The documented out-of-band stop when H is exhausted is a sound safeguard.

### F6 — Finish the model/run instantiation contract without mistaking routine choices for evidence

Sources: [A2 pooler](/home/vince/QwenOneTwo/docs/model-configuration.md:76), [draft reader](/home/vince/QwenOneTwo/docs/model-configuration.md:113), [heads](/home/vince/QwenOneTwo/docs/model-configuration.md:167), [reason heads](/home/vince/QwenOneTwo/docs/typed-value-abi.md:114), [A4 accumulation](/home/vince/QwenOneTwo/docs/training-run-profile-v1.md:19), [optimizer groups](/home/vince/QwenOneTwo/docs/training-run-profile-v1.md:100).

The published dimensions are compatible, but they do not determine a unique module:

- “Cross-attention block” needs residual/norm order, normalization type/epsilon, projection biases and attention mask/scaling rules. The protocol scale is naturally `1/sqrt(32)` per head, distinct from the 256-wide pointer's `1/16`; the exact block equations should make that explicit.
- Source/catalogue key construction, draft tensors' packing/projection, non-assessed reason-slot and attachment heads, and clipping-boundary heads need exact input/output wiring. The listed status head alone does not generate reasons.
- The inherited Qev pointer Q/K matrices are trainable under the profile but are neither the D LoRA nor newly initialized projections. Assign their LR/group explicitly. New query/prototype/embedding parameters are excluded from decay, but their positive LR group membership should also be recorded.
- “Xavier initialization” needs uniform/normal and gain. Pooler queries and the exact separation of frozen/live copies need a parameter inventory. These are modest choices, but shape equality is not checkpoint interchangeability.

A4 correctly requires whole-window eligible-example normalization. During an on-policy window, however, eligibility can become known only after later episodes diverge or fail their length guards. Immediate backward on episode 1 cannot in general know a component's final denominator across all eight episodes. An implementation must choose a sound strategy: precompute/replay deterministic traces with no side effects, retain necessary graphs, or accumulate per-component gradient sums before final normalization. Each has memory/compute costs. Simply divide every microbatch by eight is wrong when only some examples are eligible; see the independent calculation below.

**Correction:** record a fixture-specific module/parameter inventory for G1, then a complete one for G2/G3. Choose and test the accumulation strategy, including zero-eligible windows and failed exposures. This does not require proving all architecture choices optimal before experimentation. It does require avoiding silent train/serve or optimizer differences.

### F7 — Scope “no outcomes” to request roles; permit justified reason targets explicitly

Sources: [reference phase prohibition](/home/vince/QwenOneTwo/docs/training-run-profile-v1.md:28), [reference encoder exclusion](/home/vince/QwenOneTwo/docs/semantic-supervision.md:33), [reason annotations allowed](/home/vince/QwenOneTwo/docs/model-configuration.md:90), [reason support before freeze](/home/vince/QwenOneTwo/docs/loss-configuration-and-training-schedule.md:51).

Reference preparation is described as receiving “no decision outcomes,” and the reference encoder as never seeing a reference decision. But the same frozen artifact must support reasons, whose annotations are expressly allowed to include justified status/evidence. A reason such as “no feasible candidate because every alternative violates the required constraint” necessarily contains an outcome. Taken literally, the blanket prohibition prevents preparing the required reason role.

**Correction:** define a role-specific whitelist for question, target, alternative, rubric, attribute key and reason. Keep outcome blindness strict for question/rubric targets and inappropriate winner labels out of alternatives. Permit reviewed status/evidence in result-reason targets only. This is a wording/role-contract contradiction, not a reason to weaken the main protection against outcome-bearing request targets.

### F8 — Complete evaluation slices and operational promotion rules

Sources: [development allocation](/home/vince/QwenOneTwo/docs/training-run-profile-v1.md:207), [phase promotion](/home/vince/QwenOneTwo/docs/training-run-profile-v1.md:222), [replay baselines](/home/vince/QwenOneTwo/docs/training-run-profile-v1.md:236), [metric denominators](/home/vince/QwenOneTwo/docs/training-data-and-readiness.md:174).

The main denominators are substantially better specified than a single pooled accuracy, and failed/rejected jobs are correctly retained. Remaining gaps matter before actual promotion:

- “Insufficient status coverage” has no numeric minimum. The 13 supported operation/status combinations are Noul 4, Choice 5 and Score 4. Overall 95% status accuracy does not establish any minority-status capability: in 100 cases, all five cases of one status could fail while overall accuracy still passes. Specify coverage and report per-status failures, without imposing runtime routing quotas.
- The 1% fabricated-value rule is a discrete count rule. For a non-assessable denominator from 1 to 99, it permits zero fabrications; at 100 it permits one. A zero denominator is incomplete, not 0% error. Freeze whether the denominator is per operation, collection, phase or combined, and which label-valid failed calls enter each metric.
- Guided/transition promotion needs a declared complete-trajectory slice distinct from the unforced report. It can reuse the 400 jobs under a separate evaluation protocol, but that reuse and guidance schedule should be explicit. Reference contrasts can reuse an applicable subset of the 100 pairs, provided its roles and group denominators are fixed.
- Score acceptance is interval-based, while training may use approved point labels. Require an acceptance interval/tolerance for every evaluated point-only case before testing; otherwise the interval metric can omit valid point-labelled cases.
- No promotion metric directly measures effort-target error or whether predicted budgets are systematically excessive. Free-flow success can pass with large budgets and early stopping. This is not a contradiction because the docs disclaim optimal budgeting, but it leaves the trained effort-preference objective without a reported quality gate.
- The paired final checkpoints are repeated measurements on the same development families, not two independent demonstrations. Likewise, 50 calls can come from correlated jobs; the docs properly require family reporting but should instantiate the coverage minimum in the manifest.

**Correction:** create one evaluation matrix with collection/slice IDs, eligible units, minimum counts, label-validity handling, applicable phases, thresholds and family-based uncertainty. Include the existing replay NLL/decision baselines, effort error/allocated-budget summaries, and explicit mechanical-only strata. There is no need to increase the published 1,000/2,000 totals merely to remove ambiguity.

## Independent arithmetic and interface audit

### Packet/body/return costs

The following calculations are correct under the documented presentation without extra target attributes in returns:

| Item | Independent derivation | Result |
|---|---|---:|
| Common request/result framing | header + 8 question + target descriptor + 2 target + boundary | 13 |
| Rubric | descriptor + 5×2 slots | 11 |
| Candidate before attributes | descriptor + 4 slots | 5 |
| Assessed Noul | 13+1 | 14 |
| Assessed Score | 13+11+1 | 25 |
| Assessed Choice | 13+5+A_i | 18+A_i |
| Non-assessed Noul/Choice | 13+reason descriptor+4 reason slots+R | 18+R |
| Non-assessed Score | 13+11+1+4+R | 29+R |
| Extra reason maximum | aggregate, including all attachment framing | R≤32 |
| Worst non-assessed Noul/Choice | 18+32 | 50 |
| Worst non-assessed Score | 29+32 | 61 |
| Candidate attribute cap for returns | 64−18 | 46 |

Consequently `R_max = 50 / max(50,18+max_i A_i) / 61` for Noul/Choice/Score is correct. The Choice maximum is at most 64. Reserving only the expected assessed winner's return would be incorrect; the docs correctly require all candidates/statuses.

Attribute costs independently reproduce A1: absent 3; Boolean/int64/reference 4; decimal 5; source string `3+m` for 1≤m≤32; UTF-8 `3+ceil(b/8)` for 0≤b≤128. Empty UTF-8 and absent both cost 3 but have distinct presence/type semantics. Max UTF-8 attribute cost is 19; max source-string attribute cost is 35. Eight attributes do not imply fit: eight max source strings cost 280, already beyond a request. One source attachment of 32 states costs 33 and therefore cannot fit within R=32; a 31-state evidence-style attachment costs 32 and can. Eight four-position Boolean attribute attachments also exactly consume R=32. These are aggregate caps, not contradictory individual limits.

Source evidence costs `1+m`, maximum 33; a prior-result presentation costs `1+L`, maximum 65. Four maximum prior-result presentations cost 260 before common framing and cannot fit. Eight maximum source records plus Noul framing cost `8*33+13=277`, so the individual maxima cannot all be used together. The documents explicitly allow this fact.

For the printer encoding, literal UTF-8 bytes are Cedar=5, Birch=5, Pine=4, and `self-adhesive shipping labels`=29. All names cost 4 positions; format costs `3+ceil(29/8)=7`; rate and Boolean availability cost 4 each. Thus attributes=19, candidate=24, assessed return=37, and three-candidate request=`13+3*24=85` before actual evidence. Reserve 50, not 37. Source-span alternatives have tokenizer-dependent presentation lengths; no measured token counts were invented.

Request exact-fit examples can be constructed without violating independent limits. A Noul target with seven 128-byte UTF-8 attributes (133 positions), three 32-token evidence records (99) and one ten-token evidence record (11) gives `13+133+99+11=256`. Increasing the last span to eleven tokens gives 257. This assumes the explicit target-attribute placement proposed in F4. An assessed Choice return reaches 64 with attributes of costs `19+19+4+4=46`. These are structural fixtures, not generated/validated model outputs.

Preparation counts recompute as Noul `1+8+1=10`, Choice `1+8+16+1=26`, Score `1+8+1+1=11`. The total position ceiling is `32+256+64+32=384`; sixteen attempts can account for at most 6,144 associated positions, excluding other reasoning/answer/control positions. Revisions consume both writer positions and new immutable versions. FINISH/ABORT need an actually executed writer step; do not silently append an uncharged 33rd step.

Maximum positions are not the same as maximum objects. Inline slots/attributes create no independent Refs, but questions, targets, bank/snapshot/result/reason objects and new versions do. A full Choice construction using separate eight evidence Refs, 16 candidates, question, target, bank, snapshot, result and reason would total 30 new Refs without revisions. This count is conditional on those allocation choices, not proof that every memory implementation uses 30. Incremental immutable bank versions or newly allocated attachments can change it substantially. F1 requires a normative allocation plan.

### Budget curve, arithmetic order and ledger

For `B(s)=131072*(3s²−2s³)`, the derivative is `B′(s)=786432*s*(1−s)`, maximized at s=1/2 with slope 196,608. Therefore a 10⁻⁵ score drift can change the unrounded budget by approximately 1.96608 near the center. The warning that integer allocations can differ by about two positions is correct. Any positive error tolerance can cross a rounding boundary, so exact allocation parity is a distinct requirement.

The exact control sequence correctly validates finite FP32 raw values, promotes that value exactly to FP64, clamps, evaluates separate ordered multiplies/subtraction, then applies nonnegative half-up rounding. It must not use FP32 polynomial evaluation, FMA, a reassociated polynomial, Python-style ties-to-even `round`, or a decimal display string as its input. Signed zero maps to zero; finite out-of-range scores clamp to 0/1; NaN/Inf are rejected before clamp. Task Score does not alter B.

| Displayed target | Actual FP32 value promoted to FP64 | Rounded budget |
|---|---:|---:|
| 0 | 0 | 0 |
| .04 | 0.03999999910593033 | 612 |
| .05 | 0.05000000074505806 | 950 |
| .06 | 0.05999999865889549 | 1359 |
| .25 | .25 | 20480 |
| .5 | .5 | 65536 |
| .75 | .75 | 110592 |
| 1 | 1 | 131072 |

The exact decimal polynomial values for .04/.05/.06 are respectively 612.368384, 950.272 and 1358.954496; FP32 input conversion changes the unrounded result slightly but not these integer examples.

**Exhaustive transition audit performed:** for every n=0…131071, integer/rational comparisons located the adjacent FP32 inputs bracketing exact mathematical budget n+1/2. A separate rational software binary64 round-to-nearest-even implementation evaluated every intermediate in A3's specified order at both neighbors. All 262,144 boundary inputs agreed with the ordered scalar implementation and transitioned from n to n+1. Mismatches: zero. The closest mathematical neighbor was 1.1641532182693481×10⁻¹⁰ budget units from a half boundary (transition n=65534). This is verification of a deterministic scalar specification, not of a trained budget head or CUDA implementation. The fixture-sequence SHA256 is `d03a821158f38babc3add587022f56fdf4ed9524a6538e25fe6bd2a16940588c`.

`C=262144`, `Bmax=131072` and `B_granted=min(B_pred,C−P−A−H)` are consistent after validating `P+A+H≤C`. A tight example is P=262016, A=64, H=64: available B=0; one more prompt position must be rejected. No negative allowance should be concealed by clamping. Overrides need an explicit integer range including the same 131072 maximum and context admission; the docs' qualitative bounds should become schema constraints in the manifest.

Startup delivery spends three H positions, leaving 61. A and H remain protected through B reservations. Reservation does not spend or create positions; actual work debits B/C and releases unused capacity. For Noul, the minimum 13-position request reserves `13+50+32=95` after already charged preparation. With B=95 admission is exact-fit; B=94 fails even if the model would happen to produce a 14-position assessed return. If the admitted call actually spends 13+14+4=31 positions, 64 of the held 95 are released. The example `8+32+16+4=60`, plus 150 visible answer positions, correctly grows context by 210. Answer phase does not replenish B; B=0 still allows ordinary answer text from A.

### A2 shapes, private rows and static artifact identity

Pinned static JSON confirms hidden width 1024 and context 262144. It also confirms 24 text layers: 18 linear-attention and six full-attention layers. Backbone attention uses eight 256-wide query heads and two KV heads; it is **not** the protocol's eight 32-wide heads. The protocol product `8*32=256` is correct, and its 256-wide pointer scale is `1/sqrt(256)=1/16`.

The R/T rank/alpha pairs 8/16 and D's 16/32 all have ordinary LoRA alpha/rank scale 2. All twelve listed target suffixes exist in the pinned text architecture. Qev's local adapter configuration matches rank 16, alpha 32, original dropout .05 and those targets. Setting aligned D dropout to zero is an explicitly documented change; it is not an artifact mismatch. The local Qev configuration also contains a calibration temperature; the new profile explicitly adopts its own pointer scaling and disclaims inherited calibration, which is appropriate.

Under exactly the listed 24 text-layer targets (excluding vision/MTP), ordinary A/B LoRA parameter counts are 5,411,328 for rank 8 and 10,822,656 for rank 16. This is a static dimension derivation, not a loaded-parameter measurement. The pointer Q/K each map 1024→256 without bias. Concatenating query, key, product and difference gives `4*256=1024` feasibility features as specified. Writer/gate input width is `1024+8=1032`. Atom, metadata and value input widths are respectively `64+3=67`, 128, and 3. Listed 9-way attribute counts, 129-way UTF-8 lengths and 37-way exponents cover 0…8, 0…128 and −18…18 correctly; source-length class zero must remain structurally masked for a present source span.

The vocabulary result requires **both** tokenizer files. Raw `tokenizer.json` has 248044 base entries and 26 added entries, ending at ID 248069. `tokenizer_config.json` adds seven more IDs through 248076. Together the pinned static definitions cover 0…248076 without gaps. The configured tied table has 248320 rows, leaving `248320−248077=243` originally unused rows. IDs 248077/248078 are free; after assigning two private rows, 241 other tail rows remain. Both top-level and text configs enable tying. This supports the row choices; actual runtime installation, masks and gradient/export behavior remain untested. Hash the whole tokenizer artifact, not only raw tokenizer.json, when validating the vocabulary boundary.

Canonical `N(x)=x/max(RMS(x),10⁻⁶)` and the documented wire scaling are mathematically consistent, with N(0)=0. Within the floor region its derivative is I/10⁻⁶. A literal autograd implementation of sqrt followed by clamp can encounter a zero-norm derivative pitfall; use a mathematically equivalent implementation with a safe zero branch or a squared floor and verify zero/near-zero derivatives on GPU. This is an implementation precaution, not an observed bug in unimplemented code. `sigma_E` has not been computed in this review and cannot be inferred from its formula. The [1,v,v²] encoders with biases avoid a purely homogeneous one-scalar normalization by construction, but do not prove useful scalar sensitivity after training.

Zero-initialized scalar rows and LoRA B factors legitimately suppress some upstream gradients initially. Direct raw regression can update a zero row; post-bootstrap connectivity fixtures are still necessary. The stated separation of FP32 masters, BF16 effective values and FP32 scalar accumulation is coherent. Excluding private IDs from softmax denominators, and keeping inactive rows `grad=None`, are necessary and correctly specified.

### A3 numeric gates

The documented absolute RMS floors are correctly derived:

- Forward FP32: `10⁻⁵ * 10⁻⁶ = 10⁻¹¹`.
- Forward BF16/logits: `10⁻² * 10⁻⁶ = 10⁻⁸`.
- Gradient FP32: `10⁻⁴ * 10⁻¹⁰ = 10⁻¹⁴`.
- Gradient BF16: `.05 * 10⁻¹⁰ = 5×10⁻¹²`.

Pointwise tests and normalized RMS tests are conjunctive, so a large pointwise atol does not override a near-zero RMS bound. Cosine is undefined at a zero gradient, and the documented threshold-based exclusion is appropriate. Interpret the “implied absolute” bounds by the reference-RMS floor; do not replace the full E_g inequality merely because the candidate is small. A zero gradient matching zero is not connectivity evidence. The required nonsaturated fixtures and initial-LoRA caveat address this correctly.

An exact simplification is available. For a non-near-zero reference r, `||g−r||≤ε||r||` implies `cos(g,r)≥sqrt(1−ε²)` for ε<1. Thus E_g≤10⁻⁴ implies cosine at least approximately .999999995, stronger than .99999; E_g≤.05 implies at least .998749218, stronger than .995. On the same groups, the current cosine gates are redundant whenever the RMS-relative gate passes. They remain useful diagnostics. The same inequality bounds the gradient norm ratio to [1−ε,1+ε]. This proof does not apply if the metrics silently use different groupings, reinforcing the need to keep group definitions fixed.

The distinction between cross-precision tolerance checks and exact discrete parity at paired deployed precision is correct. Exact traces do not follow merely from deterministic flags or close tensors. Greedy ties, near-ties, state ordering and scalar export still need paired tests. FP64 finite differences with epsilon=10⁻⁶ and stated atol/rtol are documented starting tolerances, not results; clamp corners, argmax, byte choice and BF16 rounding are properly excluded from smooth finite differences.

Static installed source inspection supports the profile's feasibility concerns and flag syntax: Torch source reports 2.15.0+cu132 and accepts `(False,False)` reduction settings; Transformers source reports 5.22.1, includes a token-ordered FP32 recurrent reference, and cache update routines use `copy_`; FLA fused recurrent backward raises `NotImplementedError`. These observations do not certify the selected CUDA kernels. Mode-captured checkpoint recomputation and functional cache updates are still required. Sharing equations and deterministic settings does not itself prove prefill/continuation batching parity.

### A4 phase, sampling, loss and promotion arithmetic

The phase totals are correct: `1000+2000+1000+2000+1000=7000` windows; eight presentations each gives 56,000. Phase exposures are 8,000 / 16,000 / 8,000 / 16,000 / 8,000. The reference phase is separate, so native/decision training has 48,000 exposures, comprising 12,000 replay and 36,000 non-replay—exactly 25%/75%. Components contribute 4,000 writer, 4,000 receiver, 4,000 reader and 4,000 replay exposures. Joint phases contribute 16,000 interleaved, 8,000 direct and 8,000 replay exposures. Repetitions do not enlarge the authored dataset or number of independent scenarios.

For component c, the required window loss is `λ_c * sum_e(I_ce * mean_loss_ce) / sum_e I_ce`, or zero when no example is eligible. Within each example, apply the declared token/field/role normalization first. Example: only two of eight examples have eligible component losses 1 and 3. The correct contribution at weight 1 is (1+3)/2=2, not (1+3)/8=.5. For semantic roles, averaging a question together with 16 alternatives as 17 ungrouped fields would violate the role-balancing intent; average within role, then across eligible roles. F6 describes why dynamic eligibility needs a deliberate accumulation algorithm.

Warmups `ceil(.05U)` are 50 / 100 / 50 / 100 / 50. The LR multiplier reaches 1 at j=w, .55 halfway through the cosine segment, and .1 at j=U. R peak 10⁻⁴ therefore ends at 10⁻⁵; D 2×10⁻⁵ ends at 2×10⁻⁶; projections 2×10⁻⁴ end at 2×10⁻⁵. Resetting the phase schedule while carrying Adam moments and per-parameter step counters is coherent, though it creates deliberate LR jumps at phase boundaries. Inactive parameters must not step from momentum/decay. A zero-eligible window must still advance the scheduled-window counter as defined; distinguish scheduled windows from successful parameter updates.

The transition q=1−k/1999 has endpoints 1 and 0. For T=10 annotated decision events, guided counts at k=0/999/1000/1999 are 10/5/4/0. The floor creates a staircase, not a continuously changing number of forced events. Counts can be evaluated as integer arithmetic `((1999−k)*T)//1999` to avoid an unnecessary floating-point boundary issue. Teacher-forced native text remains distinct from forced decisions; the docs consistently preserve that distinction in the current owners. Actual incompatible packet semantics must mask unsupported labels; a fixed correction bank cannot certify arbitrary new latent meanings.

Data sums also pass: `2000+1500+2000+500=6000` core; +2000 replay=8000 training; +1000 development+2000 locked=11,000 records before any question of repeated presentations. Pilot `6*20=120` scenarios is separate in purpose and must have explicit lineage/placement rather than silently adding to totals. Corrections are `5*80=400`, split `5*20=100` writer/reader and `5*60=300` trajectories, already included in core. Development is `100+200+100+200+400=1000`; locked is `500+500+1000=2000`. The 200 intervention examples form 100 pairs, so a 90% pair-success threshold requires 90 fully successful pairs, not 180 individually convenient examples.

In 400 free development jobs, 90% complete-flow success requires at least 360 successes. The requested 150 valid calls (50/operation) plus 100 successful direct jobs can fit within 400 even under distinct-job allocation; there is no arithmetic impossibility. At n=50, an 85% operation floor requires at least 43 successes, because 42/50=.84 and 43/50=.86. At n=100, 95% permits five failures. The no-fabrication and status gates are independent and may be tighter; F8 addresses their missing slice/coverage details. Two final checkpoints are at 900/1000 or 1900/2000 within their phases. Seventy 100-window evaluation points are scheduled across the five phases.

### B1 IDs, metadata, atoms, helpers and counters

All 40 named nonpadding roles are unique and consecutive. Role 0 is padding without a position; 41–63 are reserved/rejected. Head class counts/order are consistent: operation 4, writer 7, status 5, attribute type 6, source mode 3, feasibility 4, Noul 2. Choice-only NoFeasibleChoice and numeric-only ParseNumeric masks remain necessary. Status/source metadata codes are head index+1, while operation metadata excludes Continue and adds Budget; blindly serializing gate indices would be wrong, and B1 correctly does not prescribe doing so.

Prefix widths sum to 64: `4+3+3+3+2+4+13+8+8+9+7`. Header-body widths sum to 64: `8+8+8+8+4+5+4+5+9+2+3`. Descriptor-body widths sum to 64: `8+8+8+4+2+2+16+16`. Control widths sum to 128. Independent expansion of every interval produced exactly bits 0…127 once for each complete layout. F2 concerns applicability, not interval overlap.

Capacity checks pass: 8-bit handles cover 0…255; 5-bit candidate/attempt fields include 16; 4-bit evidence/attribute counts include 8; 9-bit packet lengths include 256; 7-bit slot indices include 127. Control B uses 18 bits and includes 131072, A/context use 19 and include 262144, H uses 7 and includes 64, and writer/object counters use 6 and include 32. Previous action uses three bits for None plus seven actions. Outcome codes 0…11 fit four bits with 12…15 rejected. Ranges, not merely bit widths, must reject values such as 300-position requests or 17 candidates.

Big-endian int64 extrema are `8000000000000000` (−2⁶³) and `7fffffffffffffff` (2⁶³−1). Decimal exponent −2 is `00000000000000fe`, not sign-extended `fffffffffffffffe`. The allowed exponent range −18…18 fits signed int8 and 37 classes. Decimal scale is intentionally preserved: (120,−2) and (12,−1) have equal numeric value but different exact scale/bytes. Do not normalize away trailing zeros or an explicitly retained exponent for a zero coefficient.

The helper formula is correct. For 120, k=6 and remainder=56, giving `(7+56/64)/64=63/512=.123046875`; exponent helper is FP32-rounded −1/9 = −.1111111119389534. Unsigned magnitude of −2⁶³ is 2⁶³, giving k=63, sign −1 and M=1 without overflowing signed abs. For max positive int64, exact M=1−2⁻⁶⁸ and the FP32 helper rounds to 1. That collision is expected: exact bytes, not helpers, preserve the value. Both decimal atoms share helpers from the complete coefficient/exponent pair; Boolean/reference/UTF-8 helpers remain zero. The FP64 remainder/division/add/division order must remain as written; converting a large coefficient to FP32 first would be a different feature map.

All seven counter denominators are powers of two and every allowed integer numerator is exactly representable in FP32. The example vector recomputes to `[.00389862060546875, .00048828125, .953125, .99945068359375, .96875, .9375, 1, 0]`. Its resource state can be consistent: after startup three H positions and one writer position, context 262000 corresponds to a 140-position prompt. Timing after charging the current position and before applying its action is implementable by debiting before the forward; the writer attempt is already counted. Holds must not be confused with actual spent positions or silently permit overspending.

Source encoding has a concrete Unicode complication: the pinned tokenizer JSON uses **NFC normalization**, followed by ByteLevel encoding. Original `e` plus combining acute (`65 cc 81` in UTF-8) and precomposed `é` (`c3 a9`) can share normalized tokens while having different original bytes. The docs correctly demand exact original sidecars and refusal/revision if mapping is not faithful. Implementations still need fixtures for normalization contractions, multibyte characters, substrings cutting across tokens, template/prompt boundaries and special-token literals. A token-byte clipping offset is not automatically an original-source byte offset. The longest raw vocabulary token string is 128 characters, so the 16-bit clipping fields are ample for the pinned ordinary byte-token sizes; that capacity does not solve mapping semantics.

Noul vocabulary IDs independently match false=3721 and true=1802, with tokenizer.json SHA256 exactly the one documented. Independent FP32 clones preserve frozen vocabulary storage. The 256-dimensional projected dot products divided by 16 and `sigmoid(logit_true−logit_false)` are consistent with a two-way softmax true probability. They do not establish semantic truth, calibration or useful initialization behavior. No word-row values were loaded or inspected as model tensors in this review.

## Training feasibility and empirical limits

The conceptual gradient design is coherent: frozen W can propagate input gradients; R and D must remain registered trainable while explicit modes select contributions; task scalars stay connected; budget feedback is intentionally detached; hard gates/status/pointers require supervised losses. Frozen reference targets and a frozen T/codec can stabilize meaning while still differentiating reconstruction loss with respect to predicted slots. Different offline reference/component contexts do not violate the single serving prefill. The current prototype's detach, bridge-only and adapter-switch behavior is properly labeled insufficient for the target.

There are nevertheless substantial unmeasured feasibility risks:

- With the pinned dimensions, one FP32 recurrent state across 18 linear-attention layers contains `18*16*128*128*4=18,874,368` bytes, exactly 18 MiB. Retaining one such full state per position for 1024 positions alone totals 18 GiB, before other activations, weights, gradients, optimizer state or workspaces. This is an illustrative naive-retention cost, not a measured peak. Pure functional updates plus practical checkpoint boundaries are important on the 24 GiB RTX 3090; G1 must measure them.
- Full-attention KV storage in BF16 is 12 KiB per position across six layers, or 3 GiB at C=262144. An unchunked dense FP32 score matrix for eight heads at that length is 2 TiB for one attention layer. The current math-SDPA feasibility profile is not evidence that maximum-length prompt prefill is practical. A later chunking/length strategy needs its own numerical checks. Short G1/A4 lengths are appropriate and the docs correctly disclaim full-context qualification.
- Eight/four/ten semantic slots are capacities, not proven sufficient bottlenecks. Frozen reconstruction can succeed on templates while failing contrasts. Live codec drift can alter presentation despite stable canonical targets. Retain behavioral interventions and split by rule/rubric family.
- Validity-masked joint training is honest but can starve downstream supervision after arbitrary writer divergence. The 400 correction examples are useful but do not solve online adjudication of unknown latent meanings. Log eligible fractions and failure modes; do not mistake a low loss caused by masking for learning. The existing staged training order is sensible.
- Hard call/gate decisions have no answer-loss derivative. Given the explicit exclusion of policy gradients and straight-through estimators, learned invocation depends on sufficient reviewed action supervision and transfer. Free-control continuation loss alone does not discover an optimal call policy.
- Successful final answers can arise from re-solving the prompt or ignoring returned information. Clean reader interventions, full-history contradictions and actual free calls remain complementary, not interchangeable evidence.

No GPU numeric agreement, recurrent backward, model instantiation, VRAM/time fit, trained semantic recovery, calibrated probability, learned native interpretation, autonomous gate, export equivalence or long-context behavior was tested here. No draft severity/effort label was promoted to gold. The workshop/refund logic and printer/meeting feasible selections are consistent with the supplied rules and stated meeting assumptions. Packing/demo scalar judgments remain subjective; .30 and [ .20, .40 ] are not mathematically derivable from a two-minute jam, and the docs correctly require review.

## Useful consolidation and simplification

1. **Keep one executable contract source per concern.** Replace repeated numeric/profile summaries in the architecture, audit and handoff with compact version references and milestone checklists. Retain the architecture's ownership/lifecycle narrative. A small schema-derived table could eventually generate role maps, body lengths and bounds, avoiding separately edited copies.
2. **Add two small tables rather than another broad design document:** the role-to-field/tag matrix from F2 and an allocation/position transition ledger from F1/F5. These close more real gaps than another overview of A1–B1.
3. **Treat target-attribute return omission as an explicit v1 simplification.** Preserve the existing return counts, retain exact sidecar binding and test the native reader with only the actually presented contents. Do not accidentally expand all stored fields.
4. **Start G1 with prescribed minimal tensors, no optional extra consumption ticks, and one continuous Score return.** Demonstrate the complete R–D–R gradient/state path before implementing every parallel byte head or producing the large curriculum. Extend to Noul, Choice, status/failure and historical returns after that proof. This is a scoped experiment, not an altered claim that a one-vector prototype implements the full codec.
5. **Make RMS-relative gradient error the effective norm/angle gate; retain cosine as a diagnostic.** Its current threshold is mathematically redundant on the same non-near-zero groups. Do not relax the effective bound merely to simplify the report.
6. **Use integer arithmetic for transition-prefix counts and a single eligibility equation for accumulation.** This removes avoidable floating-point and microbatch-weighting ambiguity.
7. **Preserve the walkthroughs as semantic examples, but remove stale implementation-status wording.** Examples still say objectives/weights need specification and that scalar handling is “validated,” while the current owners record defaults but no validation. See [Noul lines 17–18 and 95–97](/home/vince/QwenOneTwo/docs/noul-training-walkthrough.md:17), [Score lines 104–106](/home/vince/QwenOneTwo/docs/score-training-walkthrough.md:104), and [example-index costs](/home/vince/QwenOneTwo/docs/complete-training-examples.md:51). Static byte/position counts do not require a GPU model run; actual neural outputs and trained performance do. Update these links/phrases in a later authorized edit, without duplicating the owners again.

## Readiness decision

| Milestone | Review verdict | Concrete prerequisites |
|---|---|---|
| Scoped GPU feasibility implementation | **Reasonable to begin after authorization and a small fixture profile.** No need to pretend the whole pipeline is frozen first. | Pin exact module/state equations, parameter groups, selected GPU/software flags, supplied trace and scalar/gradient oracle. Keep source artifacts immutable; demonstrate functional cache and complete connected return path. |
| General codec/runtime interoperability | **Not yet a complete contract.** | Resolve F1/F2/F4/F5; define the remaining bounded head/record wiring; implement independent bit/length/handle/allocation fixtures and matching neural vectors; pass G1-dependent integration. |
| Full training and promotion | **Not ready.** | Passing GPU feasibility/1024 qualification, approved pilot and split data, frozen semantic reference artifacts including reasons, actual-state supervision/accumulation strategy, exact optimizer manifest, and completed evaluation matrix/baselines. Resolve F3/F6/F7/F8 before claiming training or semantic readiness. |
| Full context / maximum-budget release | **Not established.** | Separate length/VRAM/time work, long-history semantic and numerical testing, and later deployment authorization. |

The arithmetic results support proceeding with a disciplined experiment. They do not support treating “all A1–A4/B1 initial choices are closed” as proof that independent implementers would presently construct the same system.

## Coverage and revision identity

The repository has no existing committed source revision for these documents. The review used the pre-report source manifest with canonical manifest SHA256 `d83d6b8a2abd2b3454cd14ac01983d138db6e1403d73dc37920b23314b3dd118`. All 33 source file hashes were checked unchanged during the audit. The report itself is intentionally outside that source manifest.

Every file below was read in full. Hashes identify the reviewed bytes, not a claim of validation of their implementation.

| Source | SHA256 |
|---|---|
| [README.md](/home/vince/QwenOneTwo/README.md) | `63ca4a7914804abeb6197269de58db4d9e2aba90599c8a6fd61e861c3c4f1680` |
| [AGENTS.md](/home/vince/QwenOneTwo/AGENTS.md) | `027b1aaf928f8bbdb6a054b15ed625f8585516e192e0445ee8831c630e38a435` |
| [docs/README.md](/home/vince/QwenOneTwo/docs/README.md) | `3e630d5921059af3efaeec14b3ec0f7077ea4fb8567b1741be6ab26ca97fcf87` |
| [docs/architecture-and-training-plan.md](/home/vince/QwenOneTwo/docs/architecture-and-training-plan.md) | `2c3654d21dd37a65a11cf28e9ef546ba2a4738192ab12167d14eb35bc09df446` |
| [docs/cache-and-gradient-strategy.md](/home/vince/QwenOneTwo/docs/cache-and-gradient-strategy.md) | `021a63fc8acba41b69743f22183284986261fb0265634d8ea9abc45f549799c9` |
| [docs/choice-training-walkthrough.md](/home/vince/QwenOneTwo/docs/choice-training-walkthrough.md) | `39e34583cee087beb77522434ee2f875abfe81fc7bf4ef888e2f2fa9b36b3725` |
| [docs/complete-training-examples.md](/home/vince/QwenOneTwo/docs/complete-training-examples.md) | `38a96ecb5873256813b91ffe7fcc5726e5645607e35bf620d6b2853b920f241e` |
| [docs/data-operations.md](/home/vince/QwenOneTwo/docs/data-operations.md) | `22d93031f39f313f452c83daa2b3e8bf816a86effcf6ba8da0df0e593762785d` |
| [docs/demo-score-flow.md](/home/vince/QwenOneTwo/docs/demo-score-flow.md) | `55f46ad06034e39e6c132c5f11f78ae7ba0d303c985f9df3ae7ff8a82042b571` |
| [docs/final-decision-audit.md](/home/vince/QwenOneTwo/docs/final-decision-audit.md) | `75de24ab05297b75b1500279bb8476bb0b3fdc9eb9266a11db725cfa2d79ba5f` |
| [docs/implementation-handoff.md](/home/vince/QwenOneTwo/docs/implementation-handoff.md) | `e243f9d29970064a14f48999582b3f27417ebff2961818df26f67c224dee0fa6` |
| [docs/latent-packet-codec.md](/home/vince/QwenOneTwo/docs/latent-packet-codec.md) | `b5be989d4bbe5a483d633a22b7286663743cf17679d8a8faa5b0a574fb94c755` |
| [docs/latent-representation-v1.md](/home/vince/QwenOneTwo/docs/latent-representation-v1.md) | `89c5e7fc1040b50c8406f2c5442ee8b156f125107dee0b2a91d49335c6c19b5a` |
| [docs/latent-writer-schedule.md](/home/vince/QwenOneTwo/docs/latent-writer-schedule.md) | `b12337614fd98e25184b24c70ffd8523d073e8977537c2435fe2afb564daf493` |
| [docs/loss-configuration-and-training-schedule.md](/home/vince/QwenOneTwo/docs/loss-configuration-and-training-schedule.md) | `4f7d1e142d67ed6f6849d978f4f178617903f8314586a702b4ac37c48b014846` |
| [docs/meeting-choice-flow.md](/home/vince/QwenOneTwo/docs/meeting-choice-flow.md) | `aaf2fcb5e8ff218272bed033b53e834977dabf8a8c012841198a5c32dbc1903f` |
| [docs/memory-and-request-snapshots.md](/home/vince/QwenOneTwo/docs/memory-and-request-snapshots.md) | `9d05aefd4c84fc83e594577d57ceca2e2ff2ee65027851dcd70a492cf856daa3` |
| [docs/model-configuration.md](/home/vince/QwenOneTwo/docs/model-configuration.md) | `ced99c51f1fcb67e281457aeedcc7b3721704821388045d97a105d2437b0e856` |
| [docs/natural-invocation-prompts.md](/home/vince/QwenOneTwo/docs/natural-invocation-prompts.md) | `8decb0ee2472b18f32c60882ea766d37f604b458207e4942aacdba3ffed2d36f` |
| [docs/neural-input-encoding-v1.md](/home/vince/QwenOneTwo/docs/neural-input-encoding-v1.md) | `1171562e8463f5321d56edab9d36b820596fe6feaf0a7731d9175a425e0d080e` |
| [docs/noul-training-walkthrough.md](/home/vince/QwenOneTwo/docs/noul-training-walkthrough.md) | `7042c12a7505638c924ef33b955f7f0c014ad6725e5be001bb284ebe5b1f96d6` |
| [docs/numeric-profile-v1.md](/home/vince/QwenOneTwo/docs/numeric-profile-v1.md) | `17add9f4d6b8e6069d678a1aca4651d6daa27b3f1a44f3f5f32d6c42682c6627` |
| [docs/numeric-verification.md](/home/vince/QwenOneTwo/docs/numeric-verification.md) | `36c756536eda405c0076d6f69ff48610258c75bf7633c8943dc60071c223efc0` |
| [docs/position-limits-and-budget-accounting.md](/home/vince/QwenOneTwo/docs/position-limits-and-budget-accounting.md) | `bb66b8772abdab2bf50b44877c431c8af5127a617c5cafe95e731787867d66e2` |
| [docs/refund-noul-flow.md](/home/vince/QwenOneTwo/docs/refund-noul-flow.md) | `78d0d7a8b82a1e01c98ee09abf4cef89a8906fe0d0258dad2e5c939aaa3f8343` |
| [docs/result-statuses.md](/home/vince/QwenOneTwo/docs/result-statuses.md) | `234bf3e5e07c7d2510ed14ab2e67481271c0d0f507ff8a668511dd4b05c47f9a` |
| [docs/scalar-row-handling.md](/home/vince/QwenOneTwo/docs/scalar-row-handling.md) | `9c4e0ec3d076f57ed1c0eafec8cab39356c220fd5143f295f6ef13111aace49c` |
| [docs/score-training-walkthrough.md](/home/vince/QwenOneTwo/docs/score-training-walkthrough.md) | `195e7d05b930316f24e1320dde05565ce37e3f4d302443905da44bfd9a453b2d` |
| [docs/semantic-supervision.md](/home/vince/QwenOneTwo/docs/semantic-supervision.md) | `3efdbaa65db0c33e694464d0b3d482cc29eabd98c71cc8ada2023a60341402f1` |
| [docs/training-data-and-readiness.md](/home/vince/QwenOneTwo/docs/training-data-and-readiness.md) | `e12157993f9efddb582d354c8bd03607b1b7caa159bef5e671288d4de1e8ba77` |
| [docs/training-data-generation.md](/home/vince/QwenOneTwo/docs/training-data-generation.md) | `bfe8873eb07c7c9da6763020f15b6dd1e964164a34cf0be2198f70ac1e298845` |
| [docs/training-run-profile-v1.md](/home/vince/QwenOneTwo/docs/training-run-profile-v1.md) | `2e26eb27574f84376659df406c458032453b3164cf7f27d70c5166430b8afdb0` |
| [docs/typed-value-abi.md](/home/vince/QwenOneTwo/docs/typed-value-abi.md) | `758442976da0ad229b7da35030fcb0ce43808420a638a3d1e2d44c153c0d94a3` |

Additional pinned static artifacts inspected:

| Artifact | SHA256 |
|---|---|
| `/home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/config.json` | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| `/home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/tokenizer.json` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `/home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/tokenizer_config.json` | `49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c` |
| `/home/vince/QwenOneTwo/data/qev/qev_config.json` | `27c6c1fd444462082b53a986bc93a89b5ae53b0d0519291a646252c047b00aa3` |
| `/home/vince/QwenOneTwo/data/qev/adapter/adapter_config.json` | `cbad6ebec4f92ff36c79ef6c4a94e0c2d367fa524eb8c09b995601dc695a3d03` |

Disposable arithmetic evidence was written to `/tmp/qot_astra_review_math.py` and `/tmp/qot_astra_review_math.json`. The boundary audit used no Torch/Transformers/model imports. It checked the two neighboring FP32 inputs at each of 131,072 allocation transitions, not every representable FP32 input, not a trained scalar readout, and not a CUDA mapper implementation. The report retains the method, counts and fixture digest so this scope remains explicit even if temporary files are later removed.

Source preservation was rechecked after writing this report: all original 33 document hashes still match the manifest. The Operating Table files were read only as pinned static JSON; no services, base/Qev weights, checkpoints, deployment or original documents were modified.
