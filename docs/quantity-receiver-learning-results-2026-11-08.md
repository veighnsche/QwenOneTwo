# Matched quantity/operator receiver learning — results, 2026-11-08

**Completed; neither arm is promoted.** Both receiver paths learn the existing
training values, but neither solves held-out count/operator use. Shared status
behavior deteriorates on assessed Score. INT64 has no clear overall advantage.
This is one matched seed and fixed exposure, not evidence that either format
is unlearnable or that the architecture cannot be repaired.

Read the [fixed protocol](quantity-receiver-learning-v1.md) and
[portable evidence](evidence/quantity-receiver-learning-v1.json). Implementation
commit: `57a3cf8`. The GPU run completed on November 7 UTC; this report was
finalized November 8 Europe/Amsterdam. All model work used the specified 3090.

## What ran

Literal UTF8 versus INT64 attributes, both bound to the same two original source
counts, source evidence and verbatim Key fields. Qev receives latent packets;
UTF8 atoms are data inside those packets, not textual invocation commands.
All **254 assessed Score receiver training rows /96 families**, four passes:
**1,016 presentations and 127 updates per arm**, A4 positions189..315. No failed
training exposures. Both restore identical checkpoint-0188 weights/moments/RNG.

The existing raw Score loss and an auxiliary two-count regression train the
actual receiver D/packet path. The new 2,050-parameter count readout starts at
zero and consumes normalized EVALUATE h; it never supplies an answer to Score.
Count targets copy existing source literals after prediction. No new training
records, authored annotations, reviews or external model calls.

Status/Reason/Noul/Choice losses were deliberately absent in this value-focused
comparison. Their private heads receive no updates, but shared D/packet updates
can alter their behavior. This is an explicit limitation of the tested objective.
W, R, budget row and reference remain frozen. No native/general training.

Only after **both fits** finished, each fixed final checkpoint evaluated
**130 E source-only retention cases +180 applicable X cases**. X Score uses the
arm's numeric attributes; other operations use original inputs. No development
gradient, stopping rule, checkpoint selection or promotion. Train/development
IDs and families are disjoint. Twenty native-reader cases remain excluded;
locked data was not opened.

## Training values learn, with status damage

| Measurement, all 254 training cases | Literal | INT64 |
|---|---:|---:|
| Score mean interval distance, initial → final | 0.4048570293 → 0.0405827365 | 0.4036024587 → 0.0403374007 |
| Correct Assessed status, initial → final | 253 → 89 | 254 → 75 |
| Exact Score head / head plus status | 38 /36 | 57 /41 |
| Required-count MAE, final | 5.1607753048 | 13.1335845782 |
| Completed-count MAE, final | 8.0563239772 | 6.4100401673 |
| Both counts round to the source integers | 2/254 | 0/254 |

All exact Score hits are among the 170 endpoint targets, where clipping can
produce exactly 0 or 1. There are **0/84 exact interior Score hits** in either
arm. Interior errors nevertheless improve: the 61 non-half interior cases fall
from 0.252707→0.053860 (literal) and 0.253679→0.058358 (INT64). Thus the improvement
is not solely endpoint clipping. Neither count readout produces any exactly
correct required or completed integer. Rounding is only a diagnostic; no Score
interval or exact-value contract was widened.

The zero-initialized auxiliary readout's initial MAEs were 73.543307 required and
36.708661 completed. Those are initialization measurements, not a previously
qualified count-reading baseline. All per-pass online losses and fixed initial/
final measurements are retained; online losses do not select a checkpoint.

## Held-out quantities and operators do not qualify

The frozen controls below reuse the pinned
[previous numeric-access run](numeric-access-results-2026-11-07.md); they were not
rerun. The original source-only parent mean error was 0.2338902032.

| X Score measurement | Frozen literal | Learned literal | Frozen INT64 | Learned INT64 |
|---|---:|---:|---:|---:|
| Mean interval distance, 40 cases | 0.2346320546 | 0.3374772806 | 0.2348618014 | 0.3383736139 |
| Ordinary completion distance, 20 | 0.2651124950 | 0.2383401253 | 0.2677185292 | 0.3289889189 |
| Reversal distance, 10 | 0.1966653995 | 0.5033985876 | 0.1964080982 | 0.4201030068 |
| Half-scale distance, 10 | 0.2116378288 | 0.3698302841 | 0.2076020489 | 0.2754136111 |
| Correct Assessed status /40 | 40 | 7 | 40 | 12 |
| Exact head / delivered Score /40 | 0 /0 | 0 /0 | 0 /0 | 0 /0 |
| Correct reversal direction /10 | 5 | 5 | 5 | 5 |
| Correct half-scale direction /10 | 0 | 5 | 0 | 1 |

Both learned arms have **0/20 pairs with both scalar results correct**. Literal
improves ordinary-completion error modestly and some half-scale directions, but
that does not offset its operator/magnitude and status failures.

| X raw-count diagnostic, 40 cases | Literal | INT64 |
|---|---:|---:|
| Required-count MAE | 98.5303640366 | 105.5418117523 |
| Completed-count MAE | 29.3867990494 | 48.3075797081 |
| Both source integers recovered, nearest-integer diagnostic | 0/40 | 0/40 |
| Completed-count MAE within training range, 22 cases | 8.0384372364 | 12.7307832891 |
| Completed-count MAE above training range, 18 cases | 55.4792412652 | 91.7903308868 |

Training has only ordinary higher-is-better completion, denominators 20/100 and
counts 0..100. X always requires 200, changes units/reference meanings, and adds
reversal/half-scale operators absent from assessed receiver training. INT64's
128 bit is never set in training but is set in every development denominator
and 14 completed-count inputs. This documents a coverage shift, not proof that
one bit causes the failure. Raw-count transfer itself fails, so this experiment
cannot isolate operator understanding independently from quantity/reference
transfer. No missing operator examples were fabricated.

## Retention is mixed; assessed Noul remains unsolved

Noul/Choice retention uses source-only inputs and the bound checkpoint-0188 E/X
receipts. The all-operation totals also include X's numeric Score inputs. The
audit explicitly maps historical view hashes and verifies records; it does not
silently treat old hashes as current input identities.

| Delivered behavior | Historical parent | Literal | INT64 |
|---|---:|---:|---:|
| Assessed Noul, E+X /80 | 0 | 0 | 0 |
| Correct Noul abstentions, E+X /60 | 22 | 40 | 47 |
| Assessed Choice E /10 | 7 | 10 | 4 |
| Assessed Choice X /40 | 40 | 40 | 40 |
| Correct Choice abstentions E /40 | 4 | 0 | 14 |
| All E+X behavioral passes /310 | 83 | 99 | 117 |

Aggregate gains mostly reflect abstentions and must not be mistaken for learned
assessed Noul/Score competence. Noul was already failing assessed delivery; it
is not a new regression from successful Noul behavior. Score's correct X status
falls from 40/40 to 7/40 and 12/40, a concrete new regression. The mixed Choice
results also prevent claiming blanket retention. All **620/620 new development
sessions pass transport/binding/cache/accounting checks**, with no runtime failures.

## Validation and artifacts

- **38 CPU tests pass in 15.203 s**; Python compilation and whitespace checks pass.
  No unchanged expensive numeric matrix was rerun.
- Both first-update checkpoints restore model, optimizer and CPU/CUDA/Python RNG
  exactly. Both arms have identical initial weights/moments/RNG and exact initial
  non-attribute wire/source/reference/accounting checks. Each arm's first eight
  training forwards exactly match its no-gradient heads/counts/tensors/ledgers.
- Same-graph separate/combined VJPs over 393 active parameters have maximum
  relative RMS **3.7994000599e-6 /2.5423569742e-6**, minimum cosine
  **0.9999999999928306 /0.9999999999968955**. These pass unchanged FP32 bounds
  1e-4 /0.99999. Quantity-only D, atom and readout gradients are nonzero.
- All **320 W tensors**, physical tying, frozen reference/R/nonreceiver owners,
  and grad=None pass. First/fourth exposures of all 254 records per arm have
  identical frozen reference payloads and changed D-dependent source states,
  confirming the fresh-source policy. Qualified v3 arithmetic remains in use;
  original BF16 transformer arithmetic retains its recorded failures.
- CPU receipt audit **PASS**, including 88 archived source files, every exposure,
  exact source count atoms, allowed optimizer owners, checkpoints and summaries.
  Exactly **3,668 Sessions /395,482 positions**: 2,032 training, 1,016 training-set
  measurements, 620 development; one prefill/lineage each. Literal/int64 positions
  197,751 /197,731, maximum Session length 209. Training components do not claim
  return consumption; all 620 development returns are actually consumed.
- Offline preparation: **11,140 source encodings /275,608 tokens**, **954 frozen
  reference encodings /25,982 cache hits**. Separate from Session accounting.
- Measured run region **7,666.574491 s**, including checkpoint I/O and measurements,
  excluding metadata preparation/final packaging; peak local CUDA allocation
  **3,622,856,704 bytes**. Window regions total 2,866.166678 /2,869.717531 s.
  One GPU run, exit0, no failed GPU attempts. Only original PID2094 /3,852 MiB
  remains. No 3060, agents, services, Operating Table or deployment changes.

Raw artifacts: `data/quantity-learning-20261107/run/`, with immutable source
snapshot, shared initial checkpoint, five checkpoints per arm, all training and
measurement receipts, plans and report. Parent directory contains audit scripts,
audit/test receipts, exit/GPU evidence and log. Report SHA256:
`16be7f0583d56e3a7e9ca4dfb5aea6a6f145e023abbc454af76d2a067851385e`.

**Next unfinished slice:** preserve/recover status and assessed behavior while
retaining value learning, then qualify quantity-range and operator transfer.
The current objective omitted status/retention losses, so it does not establish
that a joint or isolated repair would fail. The missing receiver operator/range
coverage is a separate prerequisite for claiming trained operator generalization.
No further fit, data generation, native training or promotion is authorized by
this result. Reference fidelity, Reason meaning and complete native use remain
unqualified.
