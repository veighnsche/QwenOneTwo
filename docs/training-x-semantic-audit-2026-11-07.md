# Training-to-X mismatch and surviving meaning — 2026-11-07

**Audit complete. There are substantial task/input shifts, plus failures on
familiar tasks. The evidence does not isolate a single broken layer or establish
that meaning is absent from h.** No training, model execution or data generation
was needed. W/reference were never loaded or changed.

Read the [scope/acceptance](training-x-semantic-audit-v1.md) and
[portable evidence](evidence/training-x-semantic-audit-v1.json). The reproducible
checker is `scripts/audit_latent_semantics.py`; detailed per-record/pair findings
are in `data/training-x-audit-20261107/report.json`. GPU observations below reuse
the existing RTX 3090 captures and predictions; they are not fresh model results.

## 1. Actual exposure, including a historical-count correction

The audited receiver collection contains 1,500 training rows. The saved feature
cache covers all 1,003 Noul/Score rows, including non-assessed cases. The direct
readout used only the 256 assessed Noul and 254 assessed Score rows. X contains
180 applicable cases/90 pairs, including 70 assessed Noul, 40 assessed Choice
and 40 assessed Score. The separate 20 native-reader cases remain excluded.

| Assessed task | Receiver training | X |
|---|---|---|
| Noul proposition/rule | 256 positive AND; no negated proposition or OR | 50 positive AND, 10 negated AND, 10 OR |
| Noul truth | 88 true /168 false | 50 true /20 false |
| Score formula | 254 completion fractions, all higher | 20 completion, 10 remaining fraction/lower, 10 half-completion/higher |
| Score source denominator | 84 cases use 20; 170 use 100 | All 40 use 200 |
| Score units | 96 distinct unit strings | One string: `required checkpoints` |
| Score values | 85 zero, 85 one, 23 half, 61 other; 19 distinct FP32 targets | 40 distinct targets, none zero/one/half |

The controlled generator explicitly creates these differences. The same six
domain labels occur in both splits, so domain-label coverage hides task and
numeric shifts. No training/development family or permitted-view overlap was
found. No outcome, status, threshold or interval was changed for this audit.

**The reference was not trained on only 50 meanings.** That count describes the
original 500 `reference` release rows, five distinct annotations per role. The
current pass-12 artifact actually trained on the expanded **3,376-view catalogue**:
1,114 Questions, 869 Targets, 1,055 Alternatives, 102 Conditions, 205 Reasons,
seven each Dimension/Zero/One and five each Key/Midpoint. The completed coverage
run presented each view another 12 times. See its
[measured report](reference-coverage-results-2026-11-06.md).

All **6,499 role requests /1,643 distinct canonical annotations** in the receiver
training collection match that actual reference-fitting catalogue. In X:

| Role | Distinct X annotations | Distinct matches in actual reference training |
|---|---:|---:|
| Question | 82 | 0 |
| Target | 60 | 0 |
| Alternative | 7 | 0 |
| Conditions | 3 | 0 |
| Dimension | 2 | 1 |
| Zero / One | 3 each | 1 each |

Only **70/640 role requests** match, all common Dimension/Zero/One annotations.
Canonical novelty is not proof of semantic novelty: new target names, units and
wording are legitimate transfer requirements. It does mean that a successful
training readout and an X failure also differ in reference-annotation familiarity.
This confounds attributing the difference solely to the final receiver head.

The final reference's old exact-readback panel was **71/191**, including Question
4/32, Target 3/32 and Conditions 6/32. This is a diagnostic, not a semantic grade.
The earlier inspection of 14 material readback errors used the **parent**
reference, not these final weights. Neither historical result substitutes for
current connected behavioral evidence.

## 2. Separate familiar formulas from interventions

The following recomputes descriptive subsets of existing predictions. “Direct”
is the previously fitted affine diagnostic, not a deployed replacement. The
parent includes its existing scalar head. Scalar correctness here does not
include status, Reason or native delivery.

| Score subset | Cases | Parent mean distance | Direct mean distance | Exact hits, either |
|---|---:|---:|---:|---:|
| Ordinary completion | 20 | 0.269677 | **0.228196** | 0 |
| Remaining fraction | 10 | 0.194187 | **0.478984** | 0 |
| Half-completion | 10 | 0.202020 | **0.411399** | 0 |

The prior aggregate 0.233890 →0.336694 worsening hides improvement on the
familiar formula and severe degradation on changed rubrics. That supports a
distribution/interpretation contribution. It does not solve Score: error 0.228
on ordinary completion is substantial even before exact-interval scoring.
Its denominator, unit wording and target distribution still differ from training.

| Noul subset | Labels | Parent correct | Direct correct |
|---|---|---:|---:|
| Positive AND | 40 true /10 false | 10/50 | 37/50 |
| Negated AND | 10 false | 10/10 | 2/10 |
| Positive OR | 10 true | 0/10 | 9/10 |

The parent predicts false throughout. Thus its 10/10 negated-side score is not
proof it understands negation. Direct positive-AND balanced accuracy is only
**57.5%**, AUC **0.665**, with 34/40 true and 3/10 false correct; 37/50 is below
the 40-case majority-label reference. Negation pairs get both values correct in
**1/10**, criterion-change pairs **3/10**. One-class intervention sides have no
reported AUC or balanced accuracy. Existing repeats remain in the original
denominators: the 70 assessed Noul rows contain only 50 unique permitted inputs.
These correlated panels are not independent population estimates.

## 3. What reaches the receiver, and what is demonstrated

The captured learning/evaluation paths use the same fixed neutral prompt and
prepared clean packets. `ReceiverProducer` and `evaluate_case` both pass the
role slots, bound source spans, candidate bank and explicit rubric direction to
`CleanPackets.request`. The human scenario prompt, expected values and Reason
targets do not enter those requests. Labels are read after the forward pass.
This is forced component invocation, not a learned native writer/gate.

| Boundary or behavior | Supported observation | Limit |
|---|---|---|
| Source/identity binding | 3,474 source entries match original bytes and exact coordinates; 640 saved X reference entries match role hashes | Provenance is not a semantic decoder |
| Input sensitivity | All 70 changed-input pairs have different final h; the prior capture found no pooled collision in 150 changed role comparisons | Different vectors do not establish correct logical/numeric meaning |
| Withheld information | Category 4's ten hidden-fact pairs have identical permitted inputs, wire, h and q | Only 3/10 pairs return the correct Insufficient behavior; invariance alone is inadequate |
| Noul operators | Parent direction is correct on 9/10 negations and 8/10 AND→OR pairs; direct 6/10 and 7/10 | Parent gets neither side-pair fully correct; direct gets 1/10 and 3/10 |
| Choice identity/order | All ten renamed/reordered pairs select A then X correctly, 20/20 cases | A narrow, useful binding/selection behavior, not general Choice competence |
| Choice ties | All ten unique/tied-max pairs pass, 20/20 cases | Every selection stays A; A is acceptable on both sides, so passes do not establish recognition of the expanded tie set |
| Score transformations | Reversal changes h strongly; scale changes h strongly | Direct correct direction only 5/10 and 2/10; exact values fail all 40 cases |
| Reader contradiction | Category 10 inputs/wire/h/q are identical across ten pairs | `supplied_return` is excluded by the receiver builder; this cannot establish learned recognition of misleading feedback |

Category 3 removes a required fact: h changes on every pair, but only 3/10
removed-fact cases return the required status. Across the original X parent
receipts, all 180 transports pass while behavior remains 49/180. Mechanical
communication succeeds more broadly than semantic use.

### Numbers are present, but the exact-value channel is unused here

The released `exact_literals` retain numeric bytes/coordinates. These component
builders do **not** call `source_attribute` or supply target/candidate attributes.
All 180 captured request role sequences contain **zero ATTRIBUTE_DESC and zero
INT64 positions**. Counts reach Qev through independently encoded source-body
states and their projection. They are not supplied as typed numeric atoms.

This is permitted evidence construction, not evidence truncation. Source bytes
and bindings survive in the registry, but the neural receiver must recover and
combine quantities from vectors. Exact literal availability outside the model
does not establish exact numeric access inside it. The codec already supports
source-backed typed values; this specific experiment has not tested that path.
No numeric bypass or gold-derived attribute was added.

### Input hash has a narrower scope than the full packet

`CleanView.materialize` hashes canonical role annotations, source excerpts,
purpose and operation. Rubric `direction` is passed separately through
`indices['direction']` into the renderer; it is absent as a distinct field in
those hashed bytes. Thus that view hash alone cannot certify an arbitrary
direction-only change. Existing X reversal pairs change question/rubric text as
well, and full row hashes, construction receipts and captured wire comparisons
remain available. **Zero existing view-hash groups have conflicting directions**;
no demonstrated dataset collision or invalid historical score is inferred.
Closing this audit-identity gap is a concrete follow-up, not a model result.

## 4. Validation and remaining work

The audit verifies 1,003 training feature receipts, 180 X captures, all 620 saved
direct predictions, 90 boundary-comparison receipts, exact source coordinates,
reference catalogue identity and unchanged relevant implementation sources.
Feature/capture/readout tensor files are hash-checked as opaque bytes. Training
receipts retain source tensor hashes and a full permitted-view hash; individual
reference tensor hashes are available in X receipts, not the training receipts.

Five model-free tests cover forbidden model-library imports, locked-data rejection,
duplicate receipts, strict thresholds/intervals, units and reviewed formulas.
Compilation and whitespace checks pass. An initial CPU audit rejected an incorrect
assumption that training receipts contained individual reference tensor hashes;
the checker now verifies their actual schema without claiming that extra evidence.
The preliminary report that counted only original RF rows is retained locally;
the final report adds the actual 3,376-view catalogue and supersedes that exposure
interpretation. No failed or successful new GPU attempt occurred.

Accounting: **zero new Sessions, prefills, causal positions, reference/source
encodings, optimizer updates or probe fits**. No agents, data/review generation,
locked-set access, deployment, service changes or Operating Table writes.
Unchanged v3 numeric/gradient qualifications are reused.

The audit supports multiple contributing problems, not “pooling destroys all
meaning” or “just train longer.” Choice demonstrates some usable relational
information; Noul and Score exhibit weak transfer and incorrect transformations.
Training-only failure and familiar-formula error prevent blaming X alone.

**Next unfinished slice:** bind rubric direction into a complete component-input
identity, then design a matched source-quantity access comparison using existing
source-bound counts versus the current source-state-only path. Hold question,
rubric, W/reference, receiver and scoring fixed to separate numeric access from
operator interpretation. Preserve X as development; do not train on its examples
or select a fit from its results. That comparison and any input-path modification
require subsequent scope; neither is run here. Native invocation/reading, Reason
meaning and full Noul/Choice/Score behavior remain unqualified.
