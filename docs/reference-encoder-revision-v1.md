# Stationary-decoder encoder revision v1 — 2026-11-06

The user authorized fixing encoder/pooling learning with existing data on the
RTX 3090; no new source data, agents or capability training. This revision does
not relax qualification or claim recovery before measurement.

## Causal basis and fixed change

The previous joint reference revision failed 0/100 development recovery despite
broader data reuse. Its independently frozen decoder/wire recovered six existing
training annotations exactly from optimized disposable payloads, but only one
from produced pooler vectors. This establishes local decoder expressivity for
those cases, not held-out decoder competence. Joint T/wire/pooler adaptation and
whole-history ablation did not produce a usable encoder.

Use a **stationary T and reference wire codec**, warm-started from unqualified
reference `d0d4aef44b4415beb19311350eb24533190e1a27406c5d6f59a15a5a57402dec`.
Train only the existing reference pooler; preserve all module shapes, slot
allocations, wire normalization and arithmetic. Restore ordinary causal
teacher-forced reconstruction for all exposures. No zero-history task is used
in this profile. Frozen T/codec still transmit derivatives to the pooler.
There is no annotation-text decoder prefix or dictionary decoding.

The hypothesis is that fixing the reconstruction coordinate system lets pooling
learn a stable mapping instead of co-adapting with a decoder that memorizes
frequent templates. This is a hypothesis, not proof of a unique root cause.

## Training-only implementation acceptance

Before any full block, run the focused GPU ownership/gradient check and one
**64-window, eight-exposure-per-window** pooler-only learning probe on the same
six training reference IDs: 0000/0010 Question, 0001/0011 Target, 0008/0018 Key.
Cycle them deterministically across exposures; use the actual A4 Window,
eligibility averaging, reference pooler LR 2e-4, decay rules, global clipping and
reference schedule. Freeze T/codec and record their fingerprint plus all live
weights before/after. Check reconstruction at windows 0/16/32/64 using an
independently frozen evaluation copy. Require all six final recoveries exact,
finite eligible exposures and unchanged T/codec/live state before proceeding.
Do not tune the probe, increase its windows or continue its learned weights into
the full block. Retain its real update/exposure count separately, whether it
passes or fails. A fitted six-case result never qualifies held-out recovery.

## Conditional full block and qualification

Only if implementation acceptance passes, reset to the exact parent reference
above and fresh pooler AdamW moments. Run **one fixed 1,000-window / 8,000-exposure**
reference-only block using the existing 3,376 reviewed training-view catalogue,
seed 17 and deterministic four-diversity/four-role-balanced sampler. Reuse
original row hashes/field addresses; Reason keeps its documented exception.
Cache only frozen-W annotation states on CUDA. No development/locked record is
fitted. Reconstruction coefficient/normalization, masked absent explicit
training contrasts, finite guards, clipping and LR schedule are unchanged.

`--profile encoder-v1` explicitly selects this development revision; historical
profiles remain available. Full checkpoints must preserve the actual pooler-only
optimizer ownership, frozen decoder tensors, sampler and RNG. Save/restore
checkpoint zero before the first update. Verify frozen decoder identity at each
100-window checkpoint and at completion, alongside unchanged live weights,
320 frozen base tensors/tying and original release hashes.

Evaluate all 100 unchanged development RF records and all 100 contrast groups
every 100 windows. Qualification still requires **>=95/100 exact recovery and
>=90% contrasts at both windows 900 and 1,000**. No earlier checkpoint selection,
new semantic approval, easy-case substitution or threshold/denominator change.
Record actual outcomes, including failure. Neither a passed implementation probe
nor a partial development improvement certifies the learned flow. Do not extend
the block automatically, fit locked cases, generate more data or train live
capabilities after failure. Any subsequent work needs a separate causal revision.
