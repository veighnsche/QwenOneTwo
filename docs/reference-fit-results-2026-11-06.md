# Bounded reference fit and qualification — 2026-11-06

The authorized reference-only fit completed on the RTX 3090. **It did not
qualify.** Both final checkpoints (900 and 1,000) recovered **0/100 exact canonical
annotations**, leaving all 100 unresolved, and separated **100/100 contrast
groups**. The >=95% recovery / >=90% contrast gate and final-two-checkpoint rule
remain unchanged. A trained artifact now exists; the missing-artifact blocker in
the earlier [binding report](reference-binding-results-2026-11-06.md) is historical.
The current blocker is failed held-out recovery. See the
[portable receipt](evidence/reference-fit-v1.json).

## Slice 1: causal teacher forcing

Acceptance: preserve the serial reconstruction objective, connected derivatives,
answer blindness and qualified v3 arithmetic while avoiding one transformer call
per target token during reference fitting.

`Reference.reconstruct_teacher_forced` runs the latent role/slot prefix followed
by preceding targets in one causal sequence. Each position predicts its next
target. The source annotation is never a decoder prefix. The existing serial
method remains available. Frozen storage is BF16, transformer computation is
FP32, and the vocabulary projection uses two-part compensated BF16 arithmetic.
This does not rehabilitate the original failed BF16 transformer arithmetic.

Actual RTX 3090 validation: **2/2 checks passed in 12.574 s**. For 34 targets
crossing the 32-position recurrence boundary, serial loss was
6.006882667541504 and parallel loss 6.0068769454956055. Input/T_B gradient
normalized RMS errors were 9.231887907816762e-6 / 9.425605138275557e-6, with
cosines 0.9999999999574449 / 0.9999999999561513, inside the existing 0.05/0.995
tolerances. Changing future targets left the first prediction's hidden state
exactly equal. These focused checks do not certify the entire learned flow.

## Slice 2: fixed reference-only fit and gate

Acceptance: execute the prescribed 1,000 windows of eight existing reference
records with reference-only ownership; retain checkpoint/export identities and
all scheduled exposures; evaluate every 100 windows; require both final gates;
leave live parameters and frozen W unchanged.

`scripts/fit_reference.py` selects only the 500 existing training reference rows.
They contain **50 distinct canonical meanings**, five per role, rather than 500
independent meanings. Frozen-W annotation states are cached on CUDA for those 50
meanings; trainable pooling, wire construction and T are recomputed for every
exposure. No pooled targets or training graphs are cached. Explicit reviewed
training-reference contrast relations are absent, so that training loss is
masked; arbitrary batch negatives are not invented.

The existing eligibility-based accumulation, AdamW ownership, schedule, clipping,
finite guards and rollback remain in force. No capability optimizer is created.
Evaluation uses an independent frozen reference copy and preserves training RNG.
Full checkpoints retain optimizer/sampler/RNG state. Standalone immutable
`qot-frozen-reference-v1` exports contain reference weights only and can be read
by the qualification CLI. The final export was reloaded on CUDA and its full
reference fingerprint verified. Exporting a frozen artifact grants no semantic
qualification and does not initialize the live codec.

| Measurement | Result |
|---|---:|
| Completed artifact-lineage windows / optimizer updates | 1,000 / 1,000 |
| Scheduled and eligible exposures | 8,000 / 8,000 |
| Exposure count for every one of 500 IDs | 16 |
| Failed exposures | 0 |
| Mean reconstruction loss, first / last 100 windows | 2.658715932392515 / 0.046998847847844444 |
| Median update-window duration | 9.316831011499744 s |
| Run elapsed, including preparation and periodic evaluation | 10,568.742410472012 s |
| Peak CUDA allocation | 3,627,365,888 bytes |
| Live parameter hash before / after | identical |
| Frozen-base check | 320 tensors equal; physical embedding/head tying retained |
| Capability optimizer updates | 0 |

All ten prescribed evaluations (100 through 1,000) measured 0/100 exact recovery
and 100/100 contrast success. Development contains 50 distinct canonical RF
meanings across its 100 rows; none shares a canonical annotation hash with the
50 training meanings. Contrast groups reuse the earlier 40 reviewed intervention
pairs across 39 families, with identical canonical positives. Their success
alone is weak evidence and cannot override failed recovery. Nonexact outputs
remain unresolved; no paraphrase judgments or approvals were fabricated.
The actual final reports both reject through `load_reference_evidence` with
`reference qualification did not pass`. Runner exit was **2 / BLOCKED**.

Retained restart evidence: the first attempt completed **16 reference updates**
before a checkpoint owner-name mismatch was caught. Those transient weights were
discarded; no development evaluation had run. The corrected run restarted from
the original seed/weights, prefixed owners with `reference.`, and successfully
saved/restored checkpoint 0000 before its first update. Physical work therefore
includes 16 discarded updates plus the 1,000 final-lineage updates. This was not
an extra epoch selected from development results. No additional epochs followed
the failed gate. The final populated checkpoint is retained; an interrupted
mid-run continuation was not exercised here. A resumed run into a fresh output
directory does not import prior qualification reports and conservatively cannot
promote without both recorded final checkpoints.

Model-free regression checks: **63/63 passed in 2.799 s**:

```sh
PYTHONPATH=tests .venv/bin/python -m unittest test_reference_qualification test_episode_producer test_training_inputs test_training_driver test_training_release test_training_schedule test_data_production -q
```

Reproduction command (use a new output directory; it performs the entire fixed
reference fit, so do not rerun merely to inspect evidence):

```sh
CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 \
.venv/bin/python -m scripts.fit_reference \
  --snapshot /home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
  --output data/reference-fit-20261106/run
```

## Slice 3: CPU dispatch diagnosis without more fitting

Acceptance: measure the suspected bottleneck on the actual GPU-only workload,
then capture a bounded derivative-only trace without changing training arithmetic
or adding optimizer updates.

An 8.310682439012453 s sample of the running fit measured **99.3901531025354% of
one CPU core**, while eight RTX 3090 driver samples averaged **16% GPU use**
(range 15–17%). These are process CPU and device-wide GPU samples, not exclusive
attribution of every GPU operation. All model weights/features/gradients stayed
on CUDA. This indicates host dispatch pressure, not model execution on CPU.

After fitting exited, the final export supplied one existing training episode
(`train-reference-0000`, question, 40 targets). Two warm derivative passes and
one traced pass performed **zero optimizer updates**. Reference weights remained
identical, gradients were finite CUDA tensors, and no parameter `.grad` was
accumulated. Warm unprofiled forward/VJP time was **1.596983863011701 s**.
The trace counted **52,497 CUDA kernels** and **52,243 cudaLaunchKernel calls**;
kernel interval union was **172.77820751953124 ms** in a **3,071.649952 ms**
instrumented episode. Profiler overhead materially affects host duration; this
is not a speedup or a clean utilization benchmark. The derivative-only trace
excludes accumulation checks and optimizer execution.

The trace's many multiply/reduction/backward operations, together with the
per-position eager loop in `codec/backbone.py::recurrence`, support recurrent
dispatch/backward overhead as the first optimization target. No per-function
stack trace was captured, so it does not assign every gap to that loop or isolate
all synchronization costs. A separately validated implementation reducing small
kernel launches is the next performance slice; no arithmetic/kernel change was
made during this fixed fit. CPU dispatch costs and failed held-out recovery are
separate findings.

## Artifacts, boundaries and next unfinished slice

Local raw evidence is retained under `data/reference-fit-20261106/`: both attempt
logs, accounting/gate audits, `run/source-snapshot/`, checkpoint 0000 and every
100-window checkpoint, all ten frozen exports/qualification reports, utilization
samples, profiler helper, 171 MiB Chrome trace and summary tables. The final
reference is `run/reference-1000/`, fingerprint
`fb38a31359a4ff367780bb8729695894e663810efd124c016bc2d5ba3fdbf7de`.
Binary weights, datasets and traces remain outside git; the portable receipt
binds their actual bytes.

Existing training/development/locked release hashes remain unchanged. No new
generation, model-service calls, agents, capability training, deployment or live
service changes occurred. The Operating Table supplied only the read-only base
snapshot. Completed 324-case and 108-case numeric matrices were not repeated.

**Next unfinished semantic slice:** investigate the failed reference generalization
using this retained run and existing data, document a development revision before
any new fitting, and obtain a genuine passing final-two-checkpoint gate. Low
training loss does not certify recoverable held-out meaning. No automatic extra
epochs, threshold relaxation, locked-test tuning or data generation is queued.
Only after qualification may supported live bindings and later capability phases
proceed; full participating-trainer memory, actual correction binding and learned
semantic/promotion evidence remain outstanding.
