# Reference encoder learning results — 2026-11-06

**Training-only encoder acceptance passed; development recovery is not yet
qualified.** The new Target-private pooler recovered all six fixed training
controls at window 64. This is not a replacement for the original 100-case
final-two development gate. The conditional full block remains to be measured.

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

## Pending full qualification

The [preregistered v2 protocol](reference-encoder-target-v2.md) now permits one
fresh 1,000-window / 8,000-exposure run from the original parent over all 3,376
existing reviewed training views. It uses ordinary CE, the original LR/schedule,
exact sampler, frozen decoder, independent Target pooling and no development or
locked fitting. All 100 unchanged development RF records and 100 contrast groups
are evaluated every 100 windows. Both 900 and 1,000 must pass >=95% recovery and
>=90% contrasts. No live initialization or capability training follows from
six-case acceptance. Record the actual final result before claiming a recovery
fix or promoting the reference.
