# Receiver learning report and training-panel probe — v1

The user authorizes a bounded training probe on existing records, with W and
reference frozen, no agents or new data. This is separate from the completed
one-pass receiver experiment and from a full receiver/native training phase.
Use only RTX 3090 UUID `GPU-225509c3-d702-6c27-de03-76b5e99b33b1`; no CPU fallback
or secondary GPU. Leave live services and The Operating Table unchanged.

## Fixed existing-data experiment

`scripts/train_receiver_probe.py` prepares a source-bound plan without model
imports. `--execute-gpu` explicitly executes its declared probe. Outputs stay in
a fresh `data/` directory; original checkpoint-0188 is never overwritten.

Select 32 reviewed training receiver records with 32 distinct families, disjoint
from selected development E/X families. Selection uses labels only as offline
stratification metadata, never as model inputs or development-based selection:

- Noul: eight assessed examples, four true/four false, plus one each of the
  three non-assessed statuses (eleven records).
- Choice: four assessed selections, plus one each of four non-assessed statuses
  (eight records).
- Score: ten assessed examples spread across existing target quantiles and
  available rubric directions, plus three non-assessed statuses (thirteen).

Hash-based ordering and deterministic family selection are fixed before any
model measurement. Preserve original reviewed labels, values and Score intervals.
No synthetic padding, annotations or scenarios. The panel deliberately emphasizes
assessed values; it is not an unbiased full-catalogue sample.

Fork checkpoint-0188: restore weights, exact optimizer ownership/moments, RNG
and A4 counters, retaining schedule position/warmup. Validate its original
identities and file hashes before installation. The new probe has its own source
and protocol identities, records the original sampler/cursor and source changes,
and does not claim seamless continuation of the old catalogue experiment.

Train existing D/receiver owners only. W, R, budget row and reference T/pooler/
codec remain frozen. Use unchanged eligibility normalization, losses, coefficients,
clipping and optimizer. Keep the established training backend and ordinary cache
retention; no new inference-only performance profile is silently enabled.

Complete **eight repetitions**, each visiting all 32 records once: **256 actual
presentations, 32 scheduled eight-example windows**, cumulative schedule positions
189..220. Record actual steps and failed exposures; never replay a completed
window or silently extend the run. No elapsed-time/token/cost cutoff.

## Learning report

`training/learning_report.py` separates:

- All-case status confusion and failed/scheduled denominators.
- Assessed status correctness; independent Noul/Score head correctness and
  joint head/status correctness on the training panel.
- Noul Brier score/ties and Score distance to unchanged FP32 ABI intervals,
  regardless of predicted status when a finite head value exists.
- Training raw readout range and Score clipping, with explicit measured counts.
- Choice exact identity matching and selected-value availability.
- Loss means together with eligible-example counts. Reason eligibility may
  change; its mean is not automatically a matched-example improvement.
- Actual decoded/delivered behavior and transport on held-out evaluations.

Head-only training components terminate before feedback and cannot claim delivery,
native answer use or learned invocation. Correct abstentions never count as
assessed task success. Behavioral reports reuse the existing grader and do not
widen intervals or change Noul's strict probability threshold. Missing/failed
cases stay in denominators; missing head values are explicit, not zero error.

The receiver producer records detached actual head telemetry after the real
latent request forward and before accessing outcome supervision. This neither
changes packets/losses nor supplies gold returns. Existing answer blindness and
the separate approved Reason exception remain. D-dependent source states are
fresh; only immutable frozen reference annotation payloads are cached.

## Measurement, checks and acceptance

Measure the same training panel at repetitions **0, 1, 2, 4, 8** without updates.
Use existing VJP buffers for component weighted norms and D-gradient cosine
against status in the first/last repetition only. Include coefficients and
eligible denominators; diagnostics never mutate gradients or optimizer math.
These window aggregates do not establish a unique causal explanation.

Save a new initial checkpoint, first-update checkpoint and measured-repeat
checkpoints. After the first actual update require exact model/optimizer and
CPU/CUDA/Python RNG roundtrip. Check frozen reference/R/nonreceiver codec and
grad=None at boundaries; check pinned W/physical tying before and after.

After updates, evaluate all **130 E + 180 applicable X development cases** with
actual latent results and the existing paired-intervention checks. Development
never enters optimizer updates. Reuse the saved checkpoint-0188 E/X baseline
with explicit provenance; do not claim it was newly evaluated. Exclude the 20
native-reader cases and keep locked labels unopened.

Acceptance of the experiment means complete exposure/ledger/checkpoint evidence,
honest learning curves and held-out outcomes, unchanged frozen owners and no
unrecorded fitting. A poor fit or generalization regression remains a valid
reported outcome. There is no required exact reconstruction or perfect-panel
gate, and no automatic weight promotion or additional training phase.
