# Reference encoder learning results — 2026-11-06

**The requested recovery fix is not established.** Target-private pooling passes
the focused six-training-case test, but the complete existing-data fit still
fails **0/100 development recovery at both final checkpoints**. The final broad-
data artifact also recovers only 2/6 of those training controls. No reference
qualification, live initialization or capability training occurred. See the
[portable receipt](evidence/reference-encoder-v2.json).

## Causal investigation and accepted implementation slice

All experiments reuse the original six training reference annotations, start
from the exact same parent and keep T, codec, live parameters and base W fixed.
No new source records, semantic reviews, agents or capability training occurred.
Each learning probe uses exactly 64 reference optimizer windows / 512 exposures.
The original 95%/90% development gate and numeric tolerances are unchanged.

| Candidate | Final exact training recoveries | Acceptance |
|---|---:|---|
| Shared pooler, ordinary CE, 2e-4 peak LR | 5/6 | Failed |
| Shared pooler plus dense latent-target MSE | 4/6 | Failed |
| Field-value-only memory | 5/6 | Failed; source retired |
| Shared pooler, 5x peak LR | 5/6 | Failed; source retired |
| Shared pooler, focal token CE | 5/6 | Failed; source retired |
| Shared pooler, only the two Targets fitted | 2/2 Targets; 3/6 overall | Passed isolation only |
| Target-private pooler, original mixed-role CE | **6/6** | **Passed six-case acceptance** |

No full block followed a failed candidate. The MSE probe additionally performed
96 disposable latent-variable steps to obtain its six numerical targets; these
are not model-parameter updates or new annotations. All rejected sources,
patches, logs and artifacts are retained under `data/reference-encoder-20261106/`.

The same Target confused work completion with arrival lateness across the first
five probes. A read-only pair diagnostic finds its payload cosine increased
from 0.9623206257820129 to 0.9933992624282837 in the high-rate probe. Attention
and input gradients are not confined to the pair-blind prefix. Target-only
learning recovers both Targets while regressing other roles, supporting a test
of cross-role interference. This is evidence for these cases, not proof of a
unique global root cause.

[Target-private v2](reference-encoder-target-v2.md) isolates the demonstrated
interference with one private Target branch; the other nine roles remain shared.
Migration copies the existing small memory/block/lift weights, splits the 30
queries into 28 shared and two Target queries, and initially preserves **all ten
role outputs exactly**. Pooler parameters increase from **2,110,720 to 4,213,760**;
W/T/codec, latent dimensions and actual executed position/slot counts stay the
same. No new runtime channels or textual decision requests are introduced.

The partitioned probe verifies both directions of gradient isolation: Question
updates only shared pooling; Target updates only private Target pooling. Final
CEs for the six controls are 0.059830646961927414, 0.06253871321678162,
0.09985162317752838, 0.0750420019030571, 0.0014711753465235233 and
0.002057911129668355. All six autoregressive recoveries are exact. Runtime was
**173.26270085299984 s**. Decoder/live hashes match and all **320 frozen base
tensors/tying** pass. Probe weights are not used to initialize the full block.

**Six relevant GPU regressions passed in 15.560 s**, including ordinary serial/
parallel loss and gradient parity, causality, ablation, stationary ownership,
exact partition migration, private gradients and frozen export/load. **68
model-free regressions passed in 2.811 s.** A separate partition test passed in
3.650 s. Rejected field-value/focal implementations also have retained passing
engineering checks, but their failed learning outcomes prevented acceptance.

## Completed full reference block: failed qualification

The [preregistered v2 protocol](reference-encoder-target-v2.md) completed exactly
**1,000 reference-only updates / 8,000 eligible exposures** over all **3,376**
existing reviewed views, one to 82 presentations per view. Every sampled ID and
role-local active owner matches the exact sampler; all histories are ordinary
teacher forcing and failed exposures are zero. Probe weights were discarded:
the full run reset to the original parent and fresh pooler AdamW moments.

All ten scheduled gates, 100 through 1,000, measured **0/100 exact recovery and
100/100 contrasts**. Both required final reports, 900/1,000, are rejected by the
production qualification loader with `reference qualification did not pass`.
The >=95%/90% final-two gate, cases, denominator and tolerances were unchanged.
No earlier checkpoint substituted, extra epoch ran or development/locked row
entered fitting. Contrast separation already passed before fitting and cannot
establish meaningful reconstruction.

Mean ordinary CE in the first/last 100 windows was **0.5453864361013985 /
0.5070646067904454**. Median eight-exposure window time was
**2.4458106085003237 s**; complete run time **3,913.555469761999 s**; peak CUDA
allocation **4,193,888,256 bytes**. This timing includes preparation, checkpoint/
qualification work and one brief concurrent read-only GPU weight inspection;
it is not a controlled performance comparison. Runner exit was **2/BLOCKED**,
a failed semantic gate rather than a device-access failure.

All live weights remain byte-identical; all **320 frozen base tensors** and
physical tying pass. Independent serialized-byte comparison verifies all **483
T/codec tensors against the original parent at every checkpoint 0000–1000**.
All checkpoint/export file hashes and all **49 source files** match their
archives/current code; the revision contract and all three release hashes are
unchanged. Initial full checkpoint save/restore succeeded before update one.
Capability optimizer updates were **zero**. BF16 frozen storage / FP32 transformer
computation / compensated BF16 vocabulary remain unchanged; the original BF16
transformer arithmetic remains a failed profile. No completed numeric matrix
was rerun and no service or Operating Table artifact was changed.

The final export fingerprint is
`6b90e7b706239374a088462777eb072cd166a055e353135b0058c583b9da1dd8`.
Independent RTX 3090 readback verifies that identity, no owner gradients and a
finite nonzero frozen-decoder input derivative (norm **0.07774336636066437**).
The final artifact recovers only the two Key controls (**2/6 total**), with all
four Question/Target controls unresolved. The focused 6/6 result was therefore
not transferable to the full-data recipe; it cannot certify encoder learning
across the collection or the full learned flow. Exposure/mixture sensitivity,
encoder generalization and decoder generalization remain unresolved; these
results do not establish that new data is necessary or recovery impossible.

The artifact lineage has **3,000 reference updates** (two prior 1,000-update
stages plus this stage). Total physical reference work is **3,464 updates** when
448 discarded controlled-probe updates and the original discarded 16-update
setup are included. Current probes separately used 96 disposable latent steps;
the earlier isolation's 384 latent steps remain separate historical work. Do not
call latent-variable steps model-parameter updates or count probe weights as
part of the final artifact lineage.

## Weight-space programming discussion and inspection

The user asked about direct weight-space programming while the fixed job ran;
this was a question, not an instruction to alter the frozen base or abandon the
scheduled block. The [November 2 primary report](https://rikiyat.github.io/blog/weight-space-programming/)
reports targeted removal of a looping direction in Qwen3.5-0.8B. That is relevant
motivation for a bounded repair experiment; it does not demonstrate our latent
semantic bridge. No external agent/model-service call or weight edit was made.

A read-only GPU weight audit of the parent and checkpoint 700 finds memory/lift
projection condition numbers **3.099–4.245**, effective ranks **197.03–203.93 of
256**, and bias-to-unit-input-output RMS ratios **0.00184–0.00862**. No obvious
dead projection, dominant bias or matrix rank collapse was identified by these
coarse checks. They do not rule out a targeted mechanistic repair. All matrix
analysis ran on the prescribed RTX 3090; it used no forward passes, gradients,
training cases or parameter changes. Raw results are `weight-audit.json`.

## Reproduction and next unfinished slice

Raw evidence is `data/reference-encoder-20261106/`; completed fit is `run/`, final
artifact `run/reference-1000/`. The portable receipt binds probe/helper sources,
source snapshots, protocols, logs, accounting, manifests and actual results.
The completed command was:

```sh
CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 .venv/bin/python -m scripts.fit_reference --snapshot <pinned-base-snapshot> --output data/reference-encoder-20261106/run --profile encoder-target-v2 --initial-reference data/reference-recovery-20261106/run/reference-1000
```

No fit remains active and no further fitting is queued. The next unfinished
slice is full-mixture reference recovery/generalization, including why the
focused training gain does not survive the broader recipe. A direct weight
repair needs a concrete mechanism and the same held-out acceptance, not an
invented success or an arbitrary patch. Keep W frozen and all original scope
restrictions; do not generate data, introduce agents, tune on locked cases,
weaken the gate or start capability training. Semantic binding, observed
correction materialization and full participating-trainer memory remain separate
unfinished work. This slice supplies implemented/verified mechanisms and failed
qualification evidence, not a completed recovery fix.
