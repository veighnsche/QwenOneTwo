# Slice 4: controlled receiver residual repair

This opt-in experiment follows the slice 2 threshold/status and transfer findings
and retains slice 3's learned-pooling control. The user authorized this repair on
the 3090, using existing data, frozen W/reference, no agents or new data.

The pinned parent is receiver-learning `continued/checkpoint-0188`, as identified
by `training/two_gpu.py`. Reuse its hashed E/X receipts as the unmodified control.
The treatment is one operation-specific residual at the receiver readout:

    x = h + Linear32->1024(SiLU(Linear1024->32(Norm(h))))

Noul and Score have separate branches, FP32 weights and biases. The output
projection starts at zero, making the treatment exactly identity initially.
The residual can express an offset and nonlinear adaptation without changing
the packet basis. Existing status and scalar equations read x; the Noul threshold
stays 0.5 and Score still uses actual BF16-effective LM-head row 248078. Reason
emission and structural heads retain their original h/features. Choice bypasses
the repair entirely. All existing model/codec/reference parameters remain frozen.
There are 133,184 new parameters; no new transformer positions or cache branches.

Train on all 1,003 selected, already-reviewed Noul/Score receiver records, with
the existing permitted inputs and source/semantic provenance. Capture h from
actual textless Packet forwards, before reading labels. Reuse the 24 compatible
saved panel features after hash/binding checks. Frozen feature replay is an
offline readout workload, adding zero prefills/causal positions, not end-to-end
transformer training. No development or locked inputs enter fit. All 254 assessed
Score training targets are increasing-scale points; this known gap remains.
Do not synthesize reversed examples or loosen exact targets.

Run exactly the existing A4 component schedule: 2,000 eight-example windows,
fresh AdamW (.9/.95, eps 1e-8), 100-window warmup and cosine to 0.1 peak, matrix
LR 2e-4/decay .01 and bias LR 2e-4/decay zero, global clip 1. Deterministic
hash order cycles through all 1,003 examples: 16,000 visits, each 15 or 16 times.
Use existing eligible-count normalization and unchanged unit status/Noul/Score
loss weights. Scalar supervision is eligible on reviewed Assessed cases regardless
of predicted status. Reason is frozen and has no repair objective. No optimizer
search, early stopping, development-based selection or automatic continuation.

Acceptance for **experimental completion**: complete exposure and explicit losses;
initial exact identity; GPU FP64 forward/VJP checks for the new residual at the
existing 1e-5 relative RMS tolerance (absolute 1e-11 near zero); exact minimal/full
head forward and gradient agreement; frozen parent/reference and feature hashes;
checkpoint/export/resume with matching weights, moments, counters and RNG; fresh
final training anchors matching cached h/readouts; complete 130 E plus 180 X
actual returned-result evaluations, retaining failures in every denominator;
all 90 Choice cases exactly retained, including full canonical ledgers; one
prefill/cache per newly executed Session, exact accounting, no invocation text.
Record actual results even if the repair fails.

A **useful repair** must improve development Noul assessed truth and delivered
success, reduce Score interval error and improve its exact delivered hits, with
unchanged Choice and valid transport on all 310 cases. Report status, calibration,
assessed values, exact intervals and paired interventions separately. This bar
is for improvement, not MVP qualification. No semantic threshold changes,
reference certification, native invocation/answer competence or deployment claim.
Only the final scheduled checkpoint is evaluated; training measurements at windows
0/500/1000/1500/2000 describe learning, not checkpoint selection. This comparison
tests the whole repair package against the parent, not an isolated causal
attribution to capacity versus broader exposure or fresh optimizer state.

The exported small repair bundle requires the exact parent manifest and source
profile; it is not a self-contained model. No general model loader auto-installs
it. Production arithmetic is unchanged, so completed transformer numeric matrices
are reused and the new path gets targeted GPU checks.
