# One common-mode weight-space repair — 2026-11-06

The user authorizes one bounded repair of reference weights and existing-data
testing on the RTX 3090. Base W stays frozen. No new source data, agents,
external model calls, gradients, optimizers or further fine-tuning.

## Hypothesis and fixed edit

Previously measured Target payloads became more alike (cosine 0.9623 to 0.9934)
while recovering the wrong meaning. The projection matrices themselves have no
obvious rank collapse or dominant bias. Test **activation common-mode removal**,
not a claimed dead-matrix fix: a shared output component may obscure content
variation, while the unchanged wire tags already carry role/slot syntax.

One candidate only. For each existing final checkpoint (900 and 1,000), use its
own fixed calibration outputs to compute one unit direction per physical pooler
branch. Apply exactly this rank-one affine edit, full strength, once:

    L' = L - u (u^T L)
    b' = b - u (u^T b)

Here L/b are that branch's output lift, and u is the Euclidean-unit mean output
direction. Construction uses CUDA FP64 arithmetic and stores FP32 weights;
normal forward computation stays in the existing v3 profile. Canonical N and
wire scaling remain unchanged. Only `pooler.{shared,target}.lift.{weight,bias}`
may change: four tensors per artifact. All other reference tensors, T, codec,
live weights and W must be byte-identical. No architecture, vocabulary, slot,
position, threshold, cache, tolerance or service changes.

## Input-only calibration and bounds

Use the pinned reviewed training release/catalogue only. For each role, select
at most **eight distinct existing views**, ordered by SHA256 of
`reference-common-mode-v1:17:<view-id>`. Use all views when fewer than eight
exist. The current catalogue yields 71 views (maximum 80). Preserve their
original row hashes/field addresses and F7/Reason whitelist. No development or
locked input, decoder output, target loss or success score determines the edit.

For each view, average its canonical pooler output over its slots, then average
views within each role. The private Target direction is its role mean. The
shared direction is the equal-weight average of the other nine role means.
Normalize each direction to Euclidean length one. Reject a nonfinite or zero
direction instead of choosing another repair. Compute frozen-W annotation states
once per calibration view and reuse them for both checkpoints. At most 80 W
prefills and 160 calibration-pooler forwards, plus four post-edit mechanical
checks using the same cached states; no target optimization or gradients.

One deterministic rule is applied to both saved final checkpoints so paired
validation remains visible. These are not two candidate searches. No alternative
strength, direction, parent choice, retry or automatic follow-up repair is allowed
based on outcomes. Repaired files live in a new directory; parents are preserved.

## Checks and acceptance

Before application require GPU synthetic checks for affine projection,
orthogonal-complement preservation, finite values, input nonmutation and
no autograd. Metadata checks pin deterministic training-only calibration IDs.
Snapshot/hash the actual source, protocol, calibration and parent manifests.

Record component alignment before/after, delta magnitude/counts and exact changed
tensor names. Verify frozen W's 320 tensors and physical tying, live identity,
reference nonedited tensors and all original release hashes. No optimizer or
capability updates. Numerical integrity does not establish semantic recovery.

Run the **same 100 development RF cases and all 100 existing contrast groups**
on each repaired artifact, using normal learned pooling and autoregressive
recovery. No test-time latent repair, source-text decoder prefix, dictionary,
new annotation or semantic approval. Reuse the source-bound parent reports
(0/100 recovery, 100/100 contrasts) rather than rerunning completed baselines.

Require >=95/100 exact recovery and >=90% contrasts for **both** repaired final
checkpoints for the paired recovery criterion. The original failed training run
remains historical and unchanged. No live initialization or capability training
follows this experiment, even if the criterion passes. Record unchanged or worse
results honestly. Stop after this candidate; no more fitting, data generation,
locked-label inspection or further weight repair is queued.
