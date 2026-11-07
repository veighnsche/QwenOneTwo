# Controlled receiver loss-balance comparison — v1

The user authorizes one controlled comparison on the RTX 3090 using existing
data, frozen W/reference, no agents or new data. This extends the
[training-panel protocol](receiver-training-probe-v1.md) with exactly one
experimental factor: reduce the status/feasibility pair of coefficients tenfold.

| Component | Completed control | Candidate |
|---|---:|---:|
| Status | 1 | 0.1 |
| Feasibility | 0.5 | 0.05 |
| Every other component | Original | Unchanged |

The fixed factor 0.1 is selected before execution, using the control's first-repeat
gradient observations: weighted D norms of roughly 10–18 for status, up to 24 for
feasibility, 0.18–0.92 for Noul and 1.4–2.2 for Score. It tests reduced initial
auxiliary pressure. Later Score norms already exceed status; this is a hypothesis,
not a diagnosed cause or a coefficient search. No adaptive weights or retries.
The pair changes together, so the result cannot isolate either loss individually.

Use `scripts/train_receiver_probe.py --loss-balance-baseline
data/receiver-probe-20261107/run` with the usual snapshot/output/execute arguments.
The optional comparison selects its own `receiver-loss-balance-v1` identity;
the default probe and production coefficient table remain unchanged. Only the
experiment-local Window subclass overrides combination coefficients. Loss
primitives, per-component VJPs, eligible-example division, masks, AdamW state,
learning-rate schedule, clip 1, backend and numeric contracts are unchanged.
Checkpoint sampler metadata records the effective coefficients; source/protocol
identities distinguish the fork. Resume validates the parent under its original
identities, retaining moments/RNG/counters without resetting warmup.

Reuse the completed control. Its plan/report, saved receipts and source archive
must match committed portable evidence. All executable source identities must
match the control except this runner. The panel/order, pinned parent/references/
releases and initial head/loss observations must match exactly before updates.
All **32 records × eight repetitions = 256 presentations**, windows **189..220**,
are repeated; there is no extra run, elapsed-time cutoff or extension based on
outcomes. Fixed seeds/RNG restored from the same parent. New output directory;
never overwrite control data or checkpoints.

Measure training heads at repeats 0/1/2/4/8; capture existing first/last-repeat
VJP buffers as before. After the final step evaluate all **130 E +180 applicable X**
cases, actual decoded/delivered returns and paired interventions. Never update
on development. Retain every failure in denominators. No locked/native-reader
evaluation, data generation, additional model review or native capability training.

Primary comparison: assessed Noul truth accuracy/Brier and Score interval hits/
distance on the fixed training panel. Report held-out values independently from
status and delivered behavior. Report Choice and all statuses for retention,
including regressions; correct abstentions do not count as assessed success.
Reason eligibility may change with predicted status, so its downstream eligible
counts/gradients need not match the control. Preserve the existing Reason exception.

Accept the **experiment** when the complete ledger, initial parity, unchanged
owners, frozen reference/W checks, exact first-update checkpoint roundtrip and
auditable comparison exist. A negative result is a completed experiment. No
automatic weight promotion, coefficient-default change or further training.
No component result certifies learned invocation, native result use or MVP behavior.
