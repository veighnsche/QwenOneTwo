# Completed reference block-replacement search — 2026-11-06

**All 32 candidates and all four screening evaluations are complete. No finalist
met the unchanged improvement/retention rule.** The arbitrary time cutoff was
removed, and valid prior measurements were reused. The best first-set candidate
(mask 20) recovered **12/20**, versus baseline **11/20**, but fell to **7/20** on
the separate screening set versus baseline **10/20**. This completes the agreed
experiment; it does not fix reference recovery. [Portable receipt](evidence/reference-block-search-completion-v1.json).

## Scope and implementation

The user's explicit instruction supersedes the first run's agent-chosen timeout.
Implementation `0d6b1e9` removes the deadline class, checks and fixed duration;
there is no replacement time/token/cost cutoff. The [resume protocol](reference-block-search-resume-v1.md)
retains the same 32 masks, parent/donor weights, cases, arithmetic, ranking,
top-three screening and eligibility rules. `AGENTS.md` now records the user's
standing instruction: keep local experiments simple, finish the agreed finite
work, reuse valid measurements and do not add unrequested agent-chosen cutoffs.
The original timeout report/protocol and raw evidence remain unchanged historical
records, rather than being rewritten as a successful full run.

Resume binds the previous receipt/report, source artifacts, cases and feature
identities. All **21 complete reports were copied byte-for-byte**, with no repeat
of their 420 sensitivity or recovery evaluations. For partial mask 20, **nine
completed greedy recoveries were reused** after source/candidate/target identity
matching; only its remaining 11 recoveries ran anew. The old partial log omitted
the foil log probabilities needed to complete pairwise swap scores. Its nine
short sensitivity forwards were therefore reconstructed; the missing values
were not fabricated from rounded differences. New logs retain both probabilities.
The prior in-memory feature cache had disappeared when its process exited, so
40 frozen-W encodings were reconstructed once, with identical feature identities.

**Seven metadata tests passed in 0.016 s**, including complete/partial result
reuse, artifact corruption rejection and source/target mismatch rejection.
The unchanged block replacement implementation retains its prior CUDA validation;
it was not redundantly retested. Compilation and whitespace checks pass. Actual
continuation on the RTX 3090 exercised the new resume path, 11 remaining candidate
reports and the complete second-stage selection. No gradient training occurred.

## Complete first-set results

Both sets contain 20 distinct existing annotations, two per role, with disjoint
families and annotation hashes. Both are historical training data; the second
set is separate from candidate ranking, not unseen during prior model training.
No new data, semantic labels or review calls were made. Original answer-blind
request rules and the Reason exception are unchanged.

| Mask | Donor components | Exact recovery | Mean content margin | Measurement origin |
|---|---|---:|---:|---|
| 0 | baseline | 11/20 | 3.468704 | reused |
| 1 | queries | 11/20 | 3.343383 | reused |
| 2 | memory | 11/20 | 2.461719 | reused |
| 3 | queries, memory | 11/20 | 2.192924 | reused |
| 4 | blocks.0 | 11/20 | 2.630526 | reused |
| 5 | queries, blocks.0 | 11/20 | 2.383205 | reused |
| 6 | memory, blocks.0 | 8/20 | 1.603238 | reused |
| 7 | queries, memory, blocks.0 | 8/20 | 1.530822 | reused |
| 8 | blocks.1 | 10/20 | 2.539569 | reused |
| 9 | queries, blocks.1 | 10/20 | 2.548534 | reused |
| 10 | memory, blocks.1 | 8/20 | 1.894138 | reused |
| 11 | queries, memory, blocks.1 | 8/20 | 1.776964 | reused |
| 12 | blocks.0, blocks.1 | 9/20 | 1.885630 | reused |
| 13 | queries, blocks.0, blocks.1 | 9/20 | 1.717682 | reused |
| 14 | memory, blocks.0, blocks.1 | 8/20 | 1.276502 | reused |
| 15 | queries, memory, blocks.0, blocks.1 | 8/20 | 1.217599 | reused |
| 16 | lift | 11/20 | 1.942914 | reused |
| 17 | queries, lift | 11/20 | 1.961975 | reused |
| 18 | memory, lift | 9/20 | 1.483042 | reused |
| 19 | queries, memory, lift | 9/20 | 1.546739 | reused |
| 20 | blocks.0, lift | 12/20 | 1.849228 | completed now |
| 21 | queries, blocks.0, lift | 11/20 | 1.794099 | completed now |
| 22 | memory, blocks.0, lift | 7/20 | 1.360908 | completed now |
| 23 | queries, memory, blocks.0, lift | 8/20 | 1.344953 | completed now |
| 24 | blocks.1, lift | 9/20 | 1.585404 | completed now |
| 25 | queries, blocks.1, lift | 9/20 | 1.613452 | completed now |
| 26 | memory, blocks.1, lift | 9/20 | 1.216775 | completed now |
| 27 | queries, memory, blocks.1, lift | 8/20 | 1.323399 | completed now |
| 28 | blocks.0, blocks.1, lift | 9/20 | 1.319121 | completed now |
| 29 | queries, blocks.0, blocks.1, lift | 10/20 | 1.290742 | completed now |
| 30 | memory, blocks.0, blocks.1, lift | 8/20 | 1.336388 | completed now |
| 31 | queries, memory, blocks.0, blocks.1, lift | 7/20 | 1.281954 | reused |

Mask 20 replaces the first attention block and output lift in both physical
pooler branches. On the first set it gains Question 0010 and Target 0011 but loses
Zero 0025: **12/20**, a net gain of one with a regression. This newly completed
result matters; the previous partial search could not establish the best count.
Mean content margin and mean swap effect remain algebraically equal for balanced
bidirectional pairs apart from rounding, and are not independent aggregate
confirmation. Their per-case values remain available.

## Completed separate screening

The complete first-set ranking selects **20, 1, 4**. No rule or candidate changed
after seeing these results. Baseline was evaluated on the same second set.

| Mask | First-set recovery | Screening recovery | Screening content margin |
|---|---:|---:|---:|
| 0 | 11/20 | 10/20 | 3.356602 |
| 20 | 12/20 | 7/20 | 1.286478 |
| 1 | 11/20 | 10/20 | 3.158587 |
| 4 | 11/20 | 9/20 | 2.148283 |

Mask 20 gains no second-set recovery and loses Zero 0005 and Keys 0028/0048.
Mask 1 (queries) preserves the baseline's exact recovery set but adds none.
Mask 4 (first attention block) loses Key 0048 and adds none. Thus none satisfies
strict second-set improvement plus preservation of every baseline recovery in
both sets. **NO_FINALIST** is the measured outcome, not an interrupted run.
There was no eligible finalist to send to the conditional original 900/1,000
qualification. Development/locked record labels were not loaded, and no new
100-case development score is claimed. Historical 0/100 qualification remains
historical. No candidate was promoted or initialized into live weights.

## Exact work and isolation

| Work | This continuation | Across both invocations |
|---|---:|---:|
| Frozen-W annotation prefills | 40 | 80 |
| Annotation positions | 1514 | 3028 |
| Sensitivity prefills | 300 | 729 |
| Greedy recovery episodes | 291 | 720 |
| Executed diagnostic decoder positions | 7807 | 19346 |

Final logical coverage is **720 cases for each metric**: 32×20 first-set cases
plus 4×20 screening cases. Logical decoder positions are
**19252**. The difference from physical work is the
nine reconstructed sensitivity prefills / **94 positions**;
no greedy recovery was duplicated. Reused records are explicitly labelled in
the ledger. Per-candidate elapsed times retain their measurement origin; new
partial-candidate timing describes only this continuation's work.

Continuation runtime was **511.3701073270022 s**, including setup and final
integrity checks; peak allocated CUDA memory **3,622,856,704 bytes**;
exit **2/NO_FINALIST**, with `search_complete=true`. The run was allowed to finish
its work. Runtime is a measurement, not a stopping condition.

All model/tensor work used RTX 3090 UUID
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1`. Base W passes all **320 frozen tensor
comparisons and tied storage**. All **483 T/codec tensors** remain fixed; decoder
identity is `88e34bbec1442fe798c18b4b67a87ff49df388dbcd5c8c12379602b0ad412c21` before/after. Live identity is
`c56c8f45d11f4eca02ba4bbc1d1024c50d1b74cceeee8e1d3a5c8e7378485450` before/after. All owner gradients are None, original
baseline is restored, parent files and release manifests are unchanged. There
were **zero gradient/optimizer/capability updates, agents, external model calls
or new source records**. Selection using labels is still data-driven weight
optimization. No service or Operating Table artifact changed. The original v3
precision, latent transport/cache and budget-mapping contracts are unchanged.

## Evidence and next unfinished work

Raw completion evidence is `data/reference-block-search-completion-20261106/`;
prior measurements remain in `data/reference-block-search-20261106/`. The receipt
binds all 32 candidate reports, four screening reports, resume provenance, cases,
feature identities, edit recipes/fingerprints, raw ledger, source snapshot, tests
and accounting. The metadata audit verifies 21 byte-identical reused reports,
nine reused recoveries, all new calls/positions, 55 source/protocol/test files,
complete case coverage, ranking and the actual no-finalist decision. Prior
unchanged GPU checks and original source artifacts are reused.

The completed command was:

```sh
CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 .venv/bin/python -m scripts.diagnose_reference --snapshot <pinned-base-snapshot> --output data/reference-block-search-completion-20261106/run --resume data/reference-block-search-20261106/run --resume-receipt docs/evidence/reference-block-search-v1.json
```

No search cases or screening remain unfinished and no job is active. This
specific block-replacement family is exhausted without a qualifying improvement.
It does not prove all weight-space repair impossible. The next unresolved work
is reference recovery/generalization and retention, including the documented
exposure imbalance; an exposure-controlled causal comparison has not been run.
No further search, data generation, gradient fitting or capability training is
queued. The scientific problem is not resolved by a runtime constraint. Semantic
binding and full learned-flow qualification remain separate unfinished work.
