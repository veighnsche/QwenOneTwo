# RD-01 — reviewed coverage and conditional native supervision

RD-01 implements `redesign-coverage-v1` over the existing reviewed training and
development releases. This is a completed metadata audit, with no learning or
qualification claim. RD-02 and later are unstarted. The
[queue protocol](redesign-queue-prompts-2026-11-08.md) and
[progress ledger](redesign-progress.md) govern delivery.

The [portable manifest](evidence/redesign-coverage-v1.json) pins release, source,
panel and raw-inventory hashes. Raw lists remain under
`data/redesign-rd01-20261108/accepted/`: `inventory.jsonl` resolves all 9,000 selected
records to original record paths/bytes, family/split, review artifacts and source
bytes; `panels.json` fixes ordered panels and all 90 receiver intervention pairs.
The 20 X supplied-return reader cases are listed separately. No locked release
was opened. Original training/development families have zero overlap across the
whole selected collections, not just the receiver panels.

## Implemented behavior

`training/architecture_audit.py:redesign_manifest` extends the existing metadata
audit through `training/redesign_coverage.py`. The semantic audit's original
template recognizer is now shared, with its behavior preserved. The new audit
adds a conservative Choice criterion check and leaves unknown or inconsistent
prose unclassified. It never applies keyword routing to a runtime request.
The existing semantic and architecture audit receipts are reused with their
original raw-report hashes checked; no prior GPU measurement was repeated.

Every component view is rebuilt from the existing whitelist and exact source
coordinates. Gold labels, Reason targets, guidance and review columns remain
external to receiver inputs. Native prompt hashes use the existing model-free
`native_plan`; development native annotations remain metadata where the current
producer requires a training category. Original operation/status fields on
reference and language-replay rows are inherited source metadata, explicitly
excluded from assessment coverage. Reference reconstruction and language replay
do not become receiver examples.

Panels use metadata alone, with no candidate predictions: full training receiver,
full E, full applicable X, their operation/status cells, Score endpoint/interior
groups, descriptive familiarity/wording/range cohorts and a 50-case assessed
Choice retention panel. `learning_report.summarize_coverage_learning` retains the
full report alongside subgroups, rejects changed full panels/stale identities,
and uses the original graders. Failures remain scheduled; head success and actual
delivered behavior remain separate. Interval distance retains its measured-value
denominator while failed/missing heads remain visible in scheduled counts.

## Coverage and limits

| Receiver panel | Records | Assessed | Abstentions | Assessed Score endpoint / interior |
|---|---:|---:|---:|---:|
| Training | 1,500 | 668 | 832 | 170 / 84 |
| Development E | 130 | 30 | 100 | 7 / 3 |
| Development X | 180 | 150 | 30 | 0 / 40 |

All 1,500 receiver training rows /288 families are accounted for. Assessed
training contains 256 **unnegated AND** Noul cases (168 false, 88 true), 158
capacity-maximum Choice cases and 254 ordinary completion Score cases. Score
requirements are 100 in 170 records and 20 in 84 records; no assessed receiver
training examples cover negated AND, OR, remaining-fraction, half-completion or
denominator 200. Abstentions are not evidence of learning those assessed tasks.

On unseen development families, familiar operator/range cohorts contain eight E
cases (four Noul/four Choice) and 90 X cases (50 Noul/40 Choice). E has 19 assessed
cases whose operator meaning remains unclassified, rather than 19 invented
operator annotations. Three recognized E Score cases and all 40 X Score cases
have denominator 200. Thus **no assessed Score E/X cohort meets the declared
familiar numeric-range condition**. This limits a future familiar-task Score
improvement claim; it does not leave RD-01 unfinished or authorize new examples.

All 180 X receiver cases change at least one canonical role annotation relative
to training. Sixty also have an unseen operator/range (20 Noul and 40 Score).
These overlapping axes describe concurrent changes in family, wording/binding,
operation and range. They do not isolate one causal explanation for failures.
Canonical annotation novelty is a byte-level fact, not proof of semantic novelty.
Full E/X results remain required and repeatedly inspected development remains
development. No new holdout, tolerance, grading rule or annotation was introduced.

## Native target inventory

Training contains 750 writer records/750 request targets, 2,000 interleaved
records/2,400 request targets, 750 isolated supplied-return reader records,
500 direct records, 1,000 decision replays, 1,000 language replays and 500
reference records. Development inventories 70 writer, 100 R reader plus 20 X
reader, and 400 direct records, alongside the receiver/reference panels.

Each writer/interleaved call records its exact request-role hashes, operation,
scope, result hash, per-call Reason eligibility and every exact fact/rule byte
occurrence. **224 interleaved calls lack a separately bound Reason target**;
the first call's Reason is never copied to later calls. **373 interleaved records
have ambiguous source occurrences** (including repeated rules across scopes).
Those occurrences stay visible; the audit does not manufacture a unique source
pointer. Runtime pointers and semantic agreement remain unqualified by metadata.

The existing `compatible_return` guard requires a unique Choice target for native
continuation binding: 44 writer and 139 interleaved assessed Choice calls have
multiple acceptable selections and therefore do not meet that condition. The
audit records the limitation without weakening the guard. Noul continuation
binding requires its exact reviewed probability, Score its unchanged interval,
and non-assessed returns their exact Reason payload with no attachments. All
ordered calls must remain bound/compatible before a joint continuation is used.

All 750 training reader records have supplied-return/continuation annotations;
these are isolated reading targets, not evidence of consuming actual returns.
Direct/language continuations apply only to their original permitted input before
an intervening decision changes state. Decision replay is component supervision,
not an automatic native direct/no-call target. No free-writer semantics,
correction trajectory or actual-result target was invented.

## Reproduction and validation

```sh
python -m scripts.audit_redesign_coverage --output data/redesign-rd01-20261108/accepted
python -m unittest tests.test_redesign_coverage tests.test_architecture_audit tests.test_latent_semantic_audit tests.test_learning_report
python -m compileall -q qwen_one_two/training/architecture_audit.py qwen_one_two/training/redesign_coverage.py qwen_one_two/training/learning_report.py scripts/audit_latent_semantics.py scripts/audit_redesign_coverage.py tests/test_redesign_coverage.py
git diff --check
```

The output directory must be fresh; use a new local path for reproduction.
Acceptance includes deterministic repeated full construction while model-library
imports are blocked, complete receiver accounting, family-overlap rejection,
ambiguous coverage, stale release/row/review/source rejection, hidden-label
invariance, native Reason/binding gaps and full-denominator reporting. Exact
test results and logs are recorded in the portable evidence.

Early model-free attempts exposed development-R admission and training-subpanel
release-selection errors; both were fixed. Two test expectations also confused
unnegated AND with true labels and reversed denominator counts; original rows
determined the corrected expectations. Earlier logs are retained locally and
superseded by the accepted evidence. No model/GPU attempt occurred.

No RD-01 technical check is deferred to RD-07. No models, tensors, fitting,
optimizer updates, agents, data generation, semantic review calls, service changes
or deployment occurred. Later GPU qualification and learning remain pending
under their own queue prompts; current learned capability is unchanged.
