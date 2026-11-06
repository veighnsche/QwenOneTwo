# Latent reference/receiver behavioral checks — 2026-11-06

**The gate replacement is implemented; the tested configuration fails the new
checks.** Exact reconstruction is now diagnostic only. Existing reviewed data,
final pass-12 reference weights and the existing diagnostic receiver were reused.
No generation, agents, fitting, optimizer updates, capability training, weight
promotion or service changes occurred. See the [protocol](reference-behavior-v1.md)
and [portable evidence](evidence/reference-behavior-v1.json).

## What changed

`load_reference_evidence` accepts only measured `reference-behavior-v1` evidence;
old exact-reconstruction reports cannot grant acceptance. The new runner measures
actual Noul truth, Choice identity/set membership, Score interval agreement,
statuses and complete intervention pairs through production latent packets. It
uses a fixed neutral Session prompt, withholds gold returns and Reason annotations,
and constructs only permitted, source-bound request contents. The reference's
English annotations are offline encoder inputs, never a textual Qev invocation.

Candidate construction is separate from acceptance: frozen, fingerprint-checked
references can initialize private codecs and supply exactly bound experimental
supervision without claiming that the receiver already works. Near tensors,
mutated targets, incomplete/foreign sources and unadmitted snapshots still cannot
receive reviewed decision labels. Diagnostic fixtures remain optimizer-ineligible.
Strict actual-value/snapshot/selection continuation compatibility is unchanged.
Reason retains its separate status/support input exception. This does not
authorize any training run.

## Actual results

All **180 applicable development cases / 90 pairs** completed on the specified
RTX 3090, using v3 FP32 transformer computation and compensated BF16 vocabulary
projection. The separate 20 category-9 native-reader cases are explicitly listed
as omitted, not scored as successes. There is no text reconstruction or gloss
judge in this evaluation.

| Operation | Correct behavioral result | Correct returned status |
|---|---:|---:|
| Noul | **13/100** | 15/100 |
| Choice | **31/40** | 36/40 |
| Score | **0/40** | 40/40 |
| Total | **44/180** | **91/180** |

**16/90 complete pairs pass.** By category (case successes / complete pair
successes): negation **2/20, 1/10**; changed criterion **1/20, 0/10**; removed
required evidence **2/20, 0/10**; withheld external-fact invariance **2/20, 1/10**;
candidate renaming/reordering **11/20, 1/10**; changed candidate/tie set
**20/20, 10/10**; reversed rubric **0/20, 0/10**; calibrated numeric scale
**0/20, 0/10**; misleading supplied return withheld **6/20, 3/10**.
Accepting an allowed selection on both tie-set members does not establish that
the receiver understood every changed candidate property or feasibility label.

There are **164 valid bound latent exchanges** and **16 blocked invalid Reason
decodes**: 13 have no structurally compatible pointer entry, three exceed the
Reason aggregate bound. These rejected outputs remain failures in denominators;
they are not admitted binding violations. Every one of the **180 Sessions**
passes the receipt audit for one prefill, one cache lineage and exact `P+B+A+H`
accounting, including failed decodes. No observed cache/accounting violation or
wrongly admitted snapshot/candidate binding is reported.

For a concrete failure, `development-X-05-000-0` selects the correct **A**;
its reviewed renamed/reordered counterpart selects **Y** when **X** is required.
Both use valid bound packets. For a limited success, the negation pair
`development-X-01-001-{0,1}` returns **0.5031856298446655** and
**0.49989643692970276**, giving the correct truth directions. Those near-half
probabilities demonstrate neither good calibration nor broad semantic competence.
Score outputs range from **0.32512447237968445** to **0.3610764741897583**;
none meets its approved interval. Intervals were not widened or relabeled.

All **320 frozen W tensors** and physical embedding/head tying pass. Reference
and receiver fingerprints are unchanged; all gradients remain None and optimizer
updates are **zero**. The verifier recomputes every grade, pair and denominator
from the reviewed release and rejects this report with
`semantic/behavioral checks did not pass`. No `ReferenceEvidence` was granted.
The final process exits **2**, the explicit unsuccessful-qualification result.

## Reproducibility and checks

Raw evidence: `data/reference-behavior-20261106/`, including command, source
snapshots, per-case construction audits, actual outputs, runtime receipts and
`final/report.json`. The portable receipt retains every case's actual result,
score and failure, pair results, hashes and validation audit. Frozen artifacts:

- Reference: `data/reference-coverage-20261106/final-pass/reference-006064`,
  fingerprint `5c6015dfedcd0667ab893025b39e8c35cb08cb04701138ecd66e310b64c45d3f`.
- Existing diagnostic checkpoint file SHA256:
  `1dd2e8527e14caa9002d672b4187253cdc174fb3c204ab56e80f257f23943640`.
- Receiver identity (all codec parameters plus live R/D/master parameters;
  W separately pinned):
  `4b51fe078f54f7c2852bdd5fb450ba234134f50f69c4dafe365ff286e4d0cd08`.
- Development manifest:
  `5b8f4cffdc4f33e255d3a34306cac03a68c50dcd9dab905ae8d8adcf89c81e77`.

The initial setup attempt rejected a metadata-tag assumption before scoring any
cases (exit 1). A later terminal run preserved 29 cases before exit 143; the
interruption cause is not established. A detached launcher attempt used a
resolved system interpreter instead of the virtualenv and exited 1 before model
loading. The final run corrects both setup issues, **reuses all 29 completed
measurements**, and measures only the remaining **151 cases**. Model-forward AST,
model sources, actual reference/receiver/profile identities, record/view hashes
and recomputed grades constrain reuse. Preserve the partial runs and source
snapshot. Unlogged work around interruption is not an exact physical-compute
claim.

Model-free checks: **10 behavioral grading/whitelist tests**, **6 historical
diagnostic tests**, **8 episode-plan tests** and **10 training-input tests** pass.
The RTX 3090 binding/gradient suite passes **7 tests in 43.426 s**; gradients
remain connected through frozen T into predicted slots, with input gradient
norm **0.25784578919410706**, no live adapter gradients and no optimizer updates.
The strengthened fixture initially failed because the test mutated a protected
registry copy, which correctly cannot rewrite admitted contents. After correcting
that assumption and adding an actually mismatched sealed request, **4 final GPU
binding tests pass in 17.467 s**. The initial failed test log is retained. Compilation
and whitespace checks pass. Unchanged expensive numeric matrices were reused.

## Remaining work

This result identifies incorrect receiver behavior, not a unique cause. A weak
reference encoder/pooler and an incompatible or insufficient receiver can both
cause failure; this comparison does not distinguish them. Reason support meaning,
Choice feasibility judgments, statuses beyond the tested Assessed/Insufficient
labels, learned unforced invocation/writing and correct native answer consumption
remain unqualified. Further isolation should use these saved failures and compare
reference versus receiver contribution before choosing more fitting. No further
training, pass 13, new data or follow-on experiment is queued or authorized here.
