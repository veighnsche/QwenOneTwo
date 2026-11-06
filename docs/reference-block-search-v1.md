# Bounded reference block-replacement search v1 — 2026-11-06

The user authorizes a lean diagnostic runner and bounded block-replacement search
on the prescribed RTX 3090, existing data only, frozen W, no agents or gradient
training. This protocol is fixed before model measurements. Selection using
labels is explicitly data-driven weight optimization, with zero gradient or
optimizer steps. It is not capability training or a claim of learned recovery.

## Sources and edit space

Search baseline: original failed Target-private final checkpoint 1,000,
`data/reference-encoder-20261106/run/reference-1000`, fingerprint
`6b90e7b706239374a088462777eb072cd166a055e353135b0058c583b9da1dd8`.
Donor: saved six-case successful checkpoint,
`data/reference-encoder-20261106/partition-probe/reference-0064`, fingerprint
`335a4e0aa024fc7eb92b0dec0527105f50f2b04f40f02e882d5a80faee8d4437`.
Verify original portable receipt/export/file hashes. Require identical source
names/shapes/dtypes and byte-identical nonpooler tensors, including all T/codec.
Both sources have common ancestry and identical Target-private architecture.
Do not use the retired common-mode projection exports.

Five binary decisions, in bit order: **queries, memory projection, first
attention block, second attention block, output lift**. Each choice copies the
entire named component (weights and biases) from donor or baseline, in BOTH
physical branches. Exhaustively evaluate all **32 masks**, including unchanged
baseline 0 and full donor 31. Order: 0, 31, then 1 through 30. No separate branch
search, interpolation, strength tuning, low-rank repair, random search, added
candidate, outcome-driven retry or further fitting. Interactions are part of
these 32 combinations; results do not isolate a universal single bad layer.

Before every candidate reset all 58 pooler tensors from immutable GPU source
copies, then verify every destination tensor against its intended source. Record
actual fingerprint, copied components, changed tensors/FP32 entries and untouched
count. No cumulative edits. Deduplicate identical full fingerprints, retaining
an explicit alias/result record. Base W, T, codec and live parameters stay frozen
and byte-identical. End on exact original baseline, including deadline exhaustion.

## Existing data and deterministic split

Load the pinned reviewed training release only for selection. Use its reference
records through the existing provenance/role whitelist builder; preserve answer
blindness and the documented Reason exception. Deduplicate annotation hashes.

Two sets of **20 cases**, exactly two distinct annotations per role across all
ten roles. Search set includes the six existing controls 0000/0010 (Question),
0001/0011 (Target), 0008/0018 (Key). Fill remaining role slots by SHA256 order of
`reference-block-search-v1:17:<annotation-sha256>`, taking the lexically first
record ID for duplicates. Then fill screen set using the same rule, excluding
ALL annotation hashes and source families already used. Account for all family
aliases of each duplicated annotation. Reject insufficient coverage rather than
silently relaxing separation. Record row hashes, field addresses and exact IDs.
Both sets were historical training data: screen means separate from candidate
ranking, not unseen during prior training. No development feedback selects masks;
no locked labels are loaded. No new source records, semantic labels or reviews.

Read the completed broad run's hash-bound exposure log; record actual prior
exposures per selected annotation, plus the donor's six-case exposure counts.
Do not infer forgetting from two runs with different initialization/exposure.

## Measurements and cache reuse

Load W/codec once. Encode each of the 40 selected annotations once on CUDA and
retain only frozen-W states in an in-memory cache, bound to input/token hashes,
base identity, actual chat template and source/numeric profile. Never cache
candidate pooler outputs or a mutable decoder cache across candidates/episodes.

For each same-role pair find the first differing canonical token. Supply only
their IDENTICAL preceding target-token sequence after the actual latent wire
prefix. Compute both alternative token log probabilities using unchanged v3
compensated vocabulary arithmetic. Record own-versus-foil log-probability margin
and correct-packet-versus-swapped-packet effect. This is literal conditional
sensitivity, not a new semantic equivalence label or proof of competence.

Also run the existing exact autoregressive recovery independently for every
case. No gold target is fed before its prediction in that recovery. Record first
mismatch, matched/target counts, role, original annotation identity, elapsed time,
positions and memory. Diagnostic teacher-history scores are kept distinct from
actual recovery. Full donor must reproduce 6/6 controls; baseline must reproduce
2/6. Failure of these controls aborts the experiment as an integrity issue.

## Selection, limits and conditional qualification

Rank candidates by search exact recoveries (descending), mean first-difference
margin (descending), mean swap effect (descending), fewer donor blocks, then
smaller mask. Carry exactly the top **three distinct nonbaseline** candidates to
the second set, and measure baseline there. Preserve all results, including
regressions. Eligibility requires preserving EVERY exact baseline recovery in
both sets and strictly increasing exact recovery on the second set. Rank eligible
finalists by the same second-set ordering; select at most one. No next-best retry.
If none qualifies for promotion, report **NO_FINALIST** and stop.

A fixed **600-second cooperative search deadline** starts before the 40 feature
prefills. Check between model episodes; at most the current bounded episode and
final integrity checks may finish after it. Persist each measurement immediately.
An incomplete search has no winner and is labelled INCOMPLETE, never silently
accepted or extended. Maximum 32 search evaluations + 4 second-set evaluations:
**720 sensitivity prefills and 720 recovery episodes**, plus 40 W prefills.
No gradient computations, optimizer state or model-parameter gradient updates.

Only an eligible finalist receives full qualification. Apply the same chosen
mask/donor to original parents 900 and 1,000; export both before development
feedback. Run the original 100 RF cases and 100 contrast groups for each and
invoke the real qualification loader. Require unchanged >=95/100 recovery and
>=90% contrasts at BOTH endpoints. Full qualification is a separate bounded
workload, at most this one pair of suites. No live initialization or capability
training follows, even on success. Repeatedly inspected development is not an
untouched final test. Original locked-data restrictions remain.

## Acceptance and evidence

Engineering acceptance: deterministic disjoint selection; exact 32-mask coverage;
noncumulative apply/restore; forbidden tensor/shape/mask rejection; unchanged
source and base/live/T/codec identities; correct positive controls; no gradients;
per-case raw ledger, complete/rejected/incomplete candidate accounting, shortlist
and selection replay; exact export/source/report hashes. CPU is permitted for
metadata/file checks only. All model/tensor work uses RTX 3090 UUID
GPU-225509c3-d702-6c27-de03-76b5e99b33b1. No CPU/3060 fallback, service/Operating
Table changes or repeated completed numeric matrices. Keep the current v3 and
cache/latent-runtime contracts unchanged. Publish honest results and next
unfinished slice regardless of whether a candidate helps.
