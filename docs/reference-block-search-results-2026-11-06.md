# Bounded block-replacement search results — 2026-11-06

**The runner is implemented; the search is INCOMPLETE at its predeclared time
limit.** It completed 21 of 32 configurations, then nine cases of mask 20 before
the 600-second search deadline. Ten configurations (21–30) were never started.
No completed configuration exceeded the unchanged baseline's **11/20 training
recoveries**. There is **no finalist, second-set evaluation, new development
qualification, export promotion or recovery fix**. This is partial coverage,
not a conclusion that all replacements fail. [Portable receipt](evidence/reference-block-search-v1.json).

## Implemented machinery and fixed experiment

Implementation/protocol `87b01c0` preceded every model measurement. The
[fixed protocol](reference-block-search-v1.md) enumerates all five-component
binary choices: queries, memory, attention block 0, attention block 1 and lift.
Each component comes from the original failed checkpoint 1,000 or the saved
6/6 donor, across both physical pooler branches. W/T/codec remain fixed. The
full donor is a positive control as well as one configuration. Replacements are
independent, never cumulative; immutable source copies and all destination bytes
are checked. The runner also supplies deterministic case selection, cached frozen
features, conditional packet sensitivity, original greedy recovery, candidate
fingerprints, immediate per-case logging, deduplication, ranking, retention gates
and conditional original paired qualification. This is data-driven weight
optimization with zero gradient/optimizer updates, not a gradient-training run.
Shortlisting, second-set evaluation and conditional qualification are implemented
but were not exercised by this timed-out model run; their completion is not claimed.

The first 20 cases contain two distinct annotations per role and the six known
controls. A second fixed 20-case set excludes every first-set annotation hash
and source family. Both sets come from historical training data, not unseen
model-training material. Source hashes/field addresses and the existing
answer-blind/Reason rules are retained. All 40 frozen-W encodings were cached
once; the second set was prepared but never evaluated because search coverage
was incomplete. Development and locked record labels were not loaded.

Five metadata tests passed in **0.002 s**, two CUDA tests in **0.326 s** and 19
related regressions in **0.479 s**: **26 checks total**. CUDA tests cover all 32
masks in reverse order, restoration, source isolation and rejection of forbidden
nonpooler changes, shape mismatches and invalid masks. Metadata tests cover
family/hash separation, deterministic coverage, shared-history position, ranking
and retention. Compile and whitespace checks pass. No completed numerical matrix
was repeated; the v3/graph-v1 forward, cache, latent ABI and budget code are unchanged.

## Actual complete configurations

Content margin is own-versus-foil log probability at the first differing
canonical token, with identical preceding tokens. It is a diagnostic score, not
semantic qualification. Mean packet-swap effect is algebraically equal to mean
content margin for equally weighted bidirectional pairs (apart from rounding);
they are not independent aggregate evidence. Their per-case values can differ.

| Mask | Donor components | Exact recovery | Original controls | Mean content margin |
|---|---|---:|---:|---:|
| 0 | baseline | 11/20 | 2/6 | 3.468704 |
| 1 | queries | 11/20 | 2/6 | 3.343383 |
| 2 | memory | 11/20 | 2/6 | 2.461719 |
| 3 | queries, memory | 11/20 | 2/6 | 2.192924 |
| 4 | blocks.0 | 11/20 | 2/6 | 2.630526 |
| 5 | queries, blocks.0 | 11/20 | 2/6 | 2.383205 |
| 6 | memory, blocks.0 | 8/20 | 2/6 | 1.603238 |
| 7 | queries, memory, blocks.0 | 8/20 | 2/6 | 1.530822 |
| 8 | blocks.1 | 10/20 | 2/6 | 2.539569 |
| 9 | queries, blocks.1 | 10/20 | 2/6 | 2.548534 |
| 10 | memory, blocks.1 | 8/20 | 2/6 | 1.894138 |
| 11 | queries, memory, blocks.1 | 8/20 | 2/6 | 1.776964 |
| 12 | blocks.0, blocks.1 | 9/20 | 2/6 | 1.885630 |
| 13 | queries, blocks.0, blocks.1 | 9/20 | 2/6 | 1.717682 |
| 14 | memory, blocks.0, blocks.1 | 8/20 | 2/6 | 1.276502 |
| 15 | queries, memory, blocks.0, blocks.1 | 8/20 | 2/6 | 1.217599 |
| 16 | lift | 11/20 | 2/6 | 1.942914 |
| 17 | queries, lift | 11/20 | 2/6 | 1.961975 |
| 18 | memory, lift | 9/20 | 3/6 | 1.483042 |
| 19 | queries, memory, lift | 9/20 | 3/6 | 1.546739 |
| 31 | queries, memory, blocks.0, blocks.1, lift | 7/20 | 6/6 | 1.281954 |

Mask **20** (block 0 + lift) completed **9/20 cases**; its individual measurements
are retained in the ledger and are not given a complete-case aggregate or ranked.
Masks **21–30** are untested. All 22 applied configurations have independently
reconstructed byte fingerprints. None was silently substituted or retried.

Full donor reproduces **6/6** original controls but recovers only **7/20** overall,
versus baseline **2/6** controls and **11/20** overall. The full replacement gains
four original controls while losing eight other baseline recoveries. Masks 18/19
(memory + lift, optionally queries) gain Target 0011 while losing Conditions 0024
and Zero 0025/0045, ending at **9/20**. Other completed alternatives produce no
new exact recoveries relative to baseline. This demonstrates specific tradeoffs
in these cases; it does not identify a universally defective component.

## Exposure and packet-use findings

The saved exposure log gives these actual counts for each canonical annotation:

| Existing control | Broad run presentations | Focused donor presentations |
|---|---:|---:|
| train-reference-0000 | 1 | 86 |
| train-reference-0010 | 1 | 86 |
| train-reference-0001 | 2 | 85 |
| train-reference-0011 | 2 | 85 |
| train-reference-0008 | 81 | 85 |
| train-reference-0018 | 81 | 85 |

The broad run restarted from the original parent; it did not continue the 6/6
checkpoint. These counts support exposure imbalance as a concrete hypothesis;
they do not prove the loss mechanism or authorize further fitting. The Key
controls' much greater exposure is consistent with their retained recovery.

On the baseline, the desired token becomes more likely with its own packet in
**15/20** paired checks, and the first differing token is predicted correctly in
**14/20**. Mean margin is **3.46870414018631**. Thus the decoder
shows packet sensitivity on these examples; complete recovery is still only
11/20. Pairwise token preferences neither prove semantic fidelity nor exclude
errors later in the annotation. Original 0/100 development reports remain
historical evidence; this run produced no new development score.

## Limits, accounting and isolation

The search deadline fired at **686.3953795820016 s** after process start. The
600-second cooperative limit starts after metadata/model setup and is checked
between model episodes. Final integrity checks followed. Complete process time
was **691.9866396940015 s**; median completed configuration **27.0861807830006 s**;
peak allocated CUDA memory **3,622,856,704 bytes**. Exit **2/INCOMPLETE**.
The limit was not extended after observing results. Scoring all 20 free-running
recoveries per configuration was too expensive for the complete proposed search
within this cap; that is a measured limitation of this first runner.

Executed **40 frozen-W annotation prefills / 1,514 annotation positions**,
**429 sensitivity prefills**, **429 recovery episodes**, and **11,539 diagnostic
decoder positions**. These counts include the partial configuration and are
independently replayed from the per-case ledger. There were **zero gradient
computations/updates, optimizer steps, new source records, agents, external model
calls or capability updates**. All model/tensor workloads used only RTX 3090 UUID
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1`; metadata/file auditing used CPU.

All **320 base tensors and tied storage** pass. All **483 T/codec tensors** are
byte-identical across source artifacts; decoder identity before/after is
`88e34bbec1442fe798c18b4b67a87ff49df388dbcd5c8c12379602b0ad412c21`. Live identity before/after is
`c56c8f45d11f4eca02ba4bbc1d1024c50d1b74cceeee8e1d3a5c8e7378485450`. All owner gradients are None. The exact original
baseline was restored. Parent files and all three release manifests are unchanged.
No live service or Operating Table artifact changed. Independent file-byte audit
verifies every one of **22 candidate fingerprints**, all **55 archived source/
protocol/test files**, case separation, counters and no-selection state.

## Artifacts and next unfinished slice

Raw sources, tests, cases, provenance, feature identities, per-case ledger,
21 complete reports, partial measurements and all 22 edit recipes/fingerprints
remain under `data/reference-block-search-20261106/`. The original parent/donor
weights are retained; candidate weights are exactly reproducible from those
bytes and masks without storing 22 duplicate full model files. No candidate
export was promoted. The receipt binds all raw evidence and the measured source
commit. The completed command was:

```sh
CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 .venv/bin/python -m scripts.diagnose_reference --snapshot <pinned-base-snapshot> --output data/reference-block-search-20261106/run
```

No job remains active. The next unfinished machinery slice is a cheaper initial
screen with measured cost before committing the candidate schedule: reserve
full free-running recovery for shortlisted candidates, while preserving an
independent validation set and the original qualification gate. This would be
a separately defined experiment, not a continuation under silently changed
rules. The ten unstarted masks, partial mask 20 and second-set checks remain
unmeasured. No further search, gradient fitting or data generation is queued.
The underlying reference recovery/generalization problem remains unresolved;
semantic binding and the full learned flow remain separately unqualified.
