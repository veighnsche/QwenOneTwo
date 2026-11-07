# Matched value-only receiver comparison

The user authorized one comparison on the RTX 3090, reusing the saved 1,003
features with initialization/exposure matched, frozen W/reference, unchanged
Choice, no agents or new data. This follows the unsuccessful
[joint status/value repair](receiver-residual-repair-results-2026-11-07.md).

Run the existing launcher with `--value-only`. Its control is pinned to
`data/receiver-repair-20261107/run/` and portable control evidence. Validate the
control plan/report, archived sources, feature/receipt files, initial checkpoint
and original exposure stream. All model, numeric, loss, optimizer, ownership and
schedule source files must still match; only orchestration gains the comparison
mode. Preserve the prior measurements and reuse their numeric qualification.

Load the original **checkpoint-0000**, including exact repair weights, empty
AdamW state/settings, counters and RNG. Do not reconstruct initialization from
a seed or start from the fitted repair. Keep the original 133,184-parameter
architecture, frozen parent-0188, reference, role/binding provenance, all 1,003
feature tensors and exact order. Verify every initial head, loss and metric
against the saved initial measurement. Nothing is selected using development.

The **only intervention** is removing `status` from the loss dictionary before
the existing eligible-count accumulation. Scalar losses, coefficients, labels,
interval endpoints, FP32 arithmetic, LR schedule, clipping and optimizer settings
are unchanged. Non-assessed records remain scheduled but contribute no objective.
The same 2,000 windows contain the same 16,000 visits: 955 examples visited 16
times and 48 visited 15 times. Scalar eligibility stays Noul 4,087 / Score 4,049.

Ten windows have no scalar-eligible examples: 162, 413, 538, 636, 789, 1165, 1416,
1541, 1639, 1792. Under the existing rules they advance the scheduled window
without stepping AdamW: **1,990 updates**, no replacement/padding exposures or
schedule extension. Inactive branches retain grad=None and do not receive decay
or moment updates. These are consequences of removing the objective, not an
attempt to equalize optimizer counts by changing exposure.

Keep the same forward path: repaired h feeds Noul/Score status and scalar heads;
original h feeds Reason/structure. Status predictions may therefore change even
without status supervision. Do not graft in the fitted control's status outputs.
Choice still bypasses the repair. Preserve its full actual outputs and ledgers.
No threshold calibration, rounding, changed tolerances, reference updates or
native capability training.

Acceptance for experimental completion:

- Exact initial weights, optimizer, RNG, full initial readouts/losses/metrics,
  source bindings, tensor file hash, order, exposures and scalar eligibility.
- Targeted GPU checks: minimal scalar losses/VJPs match the real full codec;
  nonzero value gradients connect to the selected residual branch, other branch
  unused. Reuse unchanged residual FP64/transformer numeric qualification.
- No new bulk extraction. Four fresh final training anchors reproduce cached h
  and repaired heads/ledgers; checkpoint/export/resume and frozen-state checks pass.
- Evaluate only final scheduled window 2000 on all 130 E and 180 X cases. All 90
  Choice cases exactly match the parent/control, and all 310 transports remain
  valid. Preserve failures in denominators and accounting; cached fitting and
  measurement add zero causal positions/prefills. Separate offline token work.
- Independent receipt audit checks the matched comparisons. Report whatever
  happens; no sweep, early stopping, time cap, automatic continuation or promotion.

Compare training and development scalar losses, Noul truth/Brier/ranking, Score
interval distance/exact hits, status and delivered behavior against **both** the
joint control and parent. Report Score direction groups and complete intervention
pairs. Improvement in training alone does not establish transfer. Lower scalar
error would support interference in this setup, not prove a universal architectural
cause; unchanged/poor results would not establish that the task is impossible.
Status regressions matter to delivered behavior even if raw values improve.

The saved reference is still semantically unqualified. Existing exact values,
identities and bindings remain mandatory; all assessed Score training examples
are increasing-scale points. Learned invocation, Reason meaning and native answer
use are outside this readout experiment. The exported repair still requires the
exact parent and explicit experimental installation; no default loader changes.
