# Fixed-representation Noul/Score readout diagnostic — v1

The user authorizes this head-only experiment on the explicitly selected RTX
3090, using existing data, frozen representations/W/reference, no agents or new
data. Leave services, production defaults and The Operating Table unchanged.

Use the same 32 training receiver records/families and ordering as the
[completed probe](receiver-training-probe-v1.md). Start from checkpoint-0188;
the joint probe's initial heads are the extraction parity reference, and its
32-update endpoint is a matched-exposure comparison. Saved checkpoint-0188 E/X
receipts are the held-out starting baseline, not the later joint checkpoint.

Only **codec.prototypes (2×1024)** and **codec.score_row (1024)** can train.
Freeze D/R, input encoders, shared Qev q/k projections, status/Choice/feasibility/
Reason heads, budget, W and reference. Noul uses the existing prototypes and
fixed Qev projections; Score uses the existing BF16-effective LM-head row with
FP32 accumulation. No new readout MLP, bias, threshold or wider Score interval.

Restore the full parent under its original strict identities/ownership first.
Project only these two heads' existing AdamW moments into a new optimizer with
their original group settings: Score peak1e−4, prototypes peak2e−4, no decay,
betas(.9,.95), epsilon1e−8. Retain A4 schedule position188 and clip1. New source/
protocol identities declare this ownership fork; no sampler/moment history is
invented. Head checkpoints omit frozen backbone adapters and require the pinned
parent to install them before restoration. They are not standalone/promoted
native-model exports. Save and verify an exact first-update roundtrip.

Extract each training feature once through the real outcome-blind latent request
on the 3090. Capture the EVALUATE hidden state before label access; preserve its
source-view hash, original ledger and frozen-representation fingerprint. Keep
features detached on CUDA and save a hash-bound local cache. This is explicitly
a fixed-feature diagnostic: gradients terminate at the cache and remain connected
to the two head masters. It does not replace connected representation learning.

Only assessed Noul/Score supply losses, using unchanged binary/raw interval
primitives and per-component eligible denominators/coefficient1. Choice and
non-assessed records are passive scheduled visits; no status/Reason/Choice loss
or update. Frozen status means delivered success can remain blocked even when
an independent value head improves; report both independently.

Complete the remaining existing A4 component schedule, windows **189..2000**:
**1812 head updates /453 repeats /14,496 cached visits**. This gives readouts more
opportunity than the joint probe's32 updates without repeating the transformer.
The later endpoint is not an equal-update comparison with that probe. No
elapsed-time cap, adaptive coefficients, restarts or outcome-based extension.
Measure repeats0/1/2/4/8/16/32/64/128/256/453. Repeat8 is the matched32-update point.
These visits execute **zero new causal positions/prefills**, and are never
reported as14,496 latent episodes. The32 extraction ledgers remain separate.

Before updates, require exact initial heads/losses against saved receipts and
exact minimal-head versus actual Codec forward/VJP parity for Noul and Score.
At measurements require exact minimal-head forward agreement with Codec on all
eligible operation inputs. After training, fresh extraction of all32 records
must reproduce hidden states/candidates and cached final readouts exactly.

Evaluate actual decoded/delivered returns on all130 E+180 applicable X cases.
Report independent Noul/Score heads regardless of predicted status, calibration,
statuses and delivered behavior. Verify all90 Choice cases exactly retain parent
actual results/statuses/views/grades; also report training Choice selection.
Use the existing grader/FP32 endpoints/bindings. No development updates, locked
labels, native-reader tests, generated scenarios or new content reviews.

Acceptance is an auditable completed experiment: exact scope/ownership/moment
projection, immutable W/reference/representations/features, unchanged accounting,
head parity/roundtrip/fresh-replay proofs and honest outcomes, including failures.
It does not require a positive result or certify convergence/capacity, native
invocation/answer use, Reason meaning or MVP behavior. No automatic promotion or
further training follows. Local derived tensors/checkpoints stay under data/.
