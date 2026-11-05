# Numeric agreement and gradient verification

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

Owner: exact invariants, floating-point tolerances, control parity, gradient
correctness and export checks. Follow [scalar rows](scalar-row-handling.md),
[losses](loss-configuration-and-training-schedule.md),
[cache/gradients](cache-and-gradient-strategy.md),
[data/readiness](training-data-and-readiness.md) and the
[implementation specification](architecture-and-training-plan.md).

The recorded [A3 numeric profile v1](numeric-profile-v1.md) fixes near-zero
denominator floors, canonical budget arithmetic, GPU kernels/flags and concrete
fixture coverage. It is a design, not passing numeric evidence.

## 1. Controlled comparisons

Use identical weights, inputs, packet tensors and explicit R/D schedules in
the functional differentiable path, that path with checkpointing, and mutable-
cache serving/optimized kernels. Each execution has its own one-prefill causal
history. Separate verification runs are not extra production prefills.

Begin with dropout off, greedy decoding, seeds 17/29/43 and recorded dtype/kernel
settings. Disable TF32 in FP32 reference checks. Floating-point operations can
differ by backend/reduction order; do not assume mathematical equivalence means
bitwise identity. Stochastic training/checkpoint tests must preserve RNG state.

## 2. Exact invariants

Require exact token/reference identities, versions, bank ordering, operation,
status, value-presence, counters, reservations, granted budgets, token masks,
frozen weights, installed BF16 scalar-row bytes, lifecycle/consumption state,
mode schedule and prefill count. Wrong binding, extra positions or changed
frozen weights never pass through a floating-point tolerance.

Across separate runs, compare fixture-relative identities/meanings; run IDs and
cache storage addresses are legitimately different. Within each run, enforce
its own exact bindings and uninterrupted state identity. Never accept foreign
references just to make a paired fixture appear identical. Frozen bytes are
compared before/after within a dtype profile; cross-dtype reference checks use
numeric comparisons rather than asserting FP32/BF16 byte identity.

## 3. Forward tolerance defaults

Pointwise agreement uses abs(x-y) <= atol + rtol*abs(reference). Also enforce
the normalized RMS limit where specified. Check each layer/state family and
boundary separately, not one pooled average.

| Comparison | atol | rtol | Additional limit |
|---|---:|---:|---|
| FP32 functional/reference forward | 1e-6 | 1e-5 | Normalized RMS <= 1e-5 |
| BF16 train/serve hidden states | 2e-2 | 2e-2 | Normalized RMS <= 1e-2 |
| BF16 train/serve attention, convolution and recurrent state | 2e-2 | 2e-2 | Normalized RMS <= 1e-2 |
| Finite allowed vocabulary logits | 5e-2 | 2e-2 | Normalized RMS <= 1e-2 |
| Scalar-row/export calculation, identical hidden input | 1e-6 | 1e-5 | Same effective row values |
| Whole-path Noul probability/task Score | 2e-3 | Not used | Absolute difference <= 0.002 |
| Whole-path startup budget score | 1e-5 | Not used | Integer allocation also exactly equal |

Normalized RMS is RMS difference divided by max(reference RMS, 1e-6), with
verification metrics accumulated in GPU FP64. The A3 profile fixes the implied
near-zero absolute RMS limits: 1e-11 FP32 and 1e-8 BF16/logits.
Compare live entries in common canonical state layout, including convolution
order and recurrent indices. Do not compare uninitialized padding or omit a
state family. Record maximum/per-layer errors rather than only a global mean.

Unexpected NaN/Inf fails immediately. Intended negative infinity in disallowed
logits is valid only at the exact expected mask. Finite inputs do not excuse
nonfinite intermediates, gradients, optimizer moments or outputs.

These defaults require pilot validation. Any change is justified/versioned on
development evidence before locked testing, never widened afterward to pass.

## 4. Discrete consequences

The retained budget curve has maximum slope 196608 tokens per score unit;
1e-5 drift can move allocation by about two tokens near the middle. Floating-
point agreement alone does not establish agreement after rounding/argmax.

- Use the exact ordered FP64 operations and half-up rounding in the A3 profile
  in both paths, starting from the exact raw FP32 readout.
- Require identical predicted/granted integers on paired certification fixtures.
- Include zero/one, clamping endpoints and immediate rounding-boundary neighbors.
- Require matching operation/writer/status/Choice decisions on paired free runs.
- Require matching greedy token traces for generation-parity fixtures.

A changed trace fails that parity claim even if tensor errors are within limits.
Report winner margin and numerical discrepancy on near ties. Use a common
arithmetic path or mark the backend pairing incompatible; do not add an
undocumented margin-based routing heuristic. A reviewed subjective Score interval
is a label criterion, not an enlarged backend-agreement tolerance.

## 5. Gradient connectivity

Use targeted vector-Jacobian checks for intended paths, not merely any nonzero
gradient somewhere in the model.

| Path | Expected evidence |
|---|---|
| Semantic alignment -> writer -> R | Finite effective gradients |
| Frozen semantic decoder -> predicted slots | Input gradients; decoder parameters stay frozen |
| Native continuation -> result encoder -> task Score | Connected gradient on interior assessed-value fixtures |
| Score -> D -> latent question/rubric | Connected gradient |
| Noul probability -> receiver/writer | Connected gradient on nonsaturated fixtures |
| Candidate/reference selection | Explicit supervised-head gradients |
| Effort loss -> budget row | Connected gradient |
| Native answer -> budget control-feedback scalar | Intentionally blocked |
| Loss -> frozen base/reference parameters | No parameter gradient/update |

For a continuous return, compute continuation gradient with respect to the
returned scalar, then propagate that contribution into receiver/writer. This
isolates the return path from gradients through other shared-history routes.

Not every element must be nonzero: some LoRA factors have legitimate zero
initial gradients. Test active paths after bootstrap or with appropriate
diagnostic parameters. Test inactive branches independently and preserve the
distinction between None, exact zero and a small finite gradient.

## 6. Small GPU FP64 derivative checks

Use finite differences on smooth reference components: scalar dot products,
slot/value encoders, draft-feedback projections, short functional recurrent
updates and short call/return compositions with discrete choices held fixed.

    epsilon = 1e-6
    atol = 1e-5
    rtol = 1e-3

Do not finite-difference BF16 rounding, argmax, integer lookup or clamp boundaries.
Check the smooth reference and test discrete behavior separately. FP64 checks
still execute on the selected GPU; they are not CPU smoke tests.

## 7. Gradient implementation agreement

Compare unscaled/unclipped gradients of the same losses and fixtures, per
meaningful parameter group/intermediate path:

| Comparison | RMS-relative error with A3 floor | Cosine similarity |
|---|---:|---:|
| FP32 reference vs functional implementation | <= 1e-4 | >= 0.99999 |
| FP32 functional checkpoint off vs on | <= 1e-4 | >= 0.99999 |
| BF16 production training vs FP32 reference | <= 0.05 | >= 0.995 |

Use gradient RMS error/max(reference RMS, 1e-10). The implied near-zero absolute
limits are 1e-14 FP32 and 5e-12 BF16. Evaluate cosine only when BOTH RMS values
exceed 1e-10. Near-zero agreement cannot prove connectivity; separate nonsaturated
path fixtures remain mandatory. See the A3 profile for exact definitions.
Checkpoint recomputation preserves modes/RNG and repeats no accounting, registry
writes or consumption side effects. No in-place saved-state corruption or hidden
call-boundary detach is acceptable even if a different branch receives gradients.

## 8. Optimizer and export checks

From reset checkpoints, isolate updates: budget-only changes eligible budget/
upstream parameters but not Score/base; Score-only leaves budget unchanged;
language-only without task return leaves both rows unchanged. Frozen table/base/
reference artifacts stay unchanged. Inactive rows have no gradient/update so
momentum cannot move them.

Distinguish FP32 master updates from BF16 effective values; a small master
change may not cross a BF16 rounding boundary immediately. After save/load
verify installed bytes, tying, masks, raw scalars and integer budgets again.

## 9. Scope and pass rule

The [G1-core protocol](small-gpu-proof-v1.md) fixes the initial bounded subset:
two prescribed continuous-value traces, targeted VJPs/negative controls, numeric
profiles, native continuation and mechanical checks. Its partial scope cannot
waive the full numeric gate below. All executions and results remain unperformed.

Start with short local derivative fixtures and complete episodes around
128-512 positions covering direct answers, each operation, successive calls,
answer-phase calls and non-assessed returns. Expand only after these pass and
never silently truncate a packet to meet a target fixture length.

The numeric gate passes only when exact invariants, float tolerances, discrete
control agreement, gradients and export checks all pass. Report tested lengths,
hardware, hashes, kernels, dtype, seeds, failures and forcing settings. Numeric
agreement is not semantic understanding, autonomous invocation or maximum-context
reliability; those remain separate readiness gates.

References: [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html)
and [gradcheck](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.gradcheck.gradcheck.html).
These support numerical methodology, not measured adequacy of our proposed limits.
