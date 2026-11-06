# Frozen-reference receiver learning — v1

Authorized scope: verify receiver training paths, then learn receiver-side
parameters using existing data on RTX 3090 UUID
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1`. Keep W, R, the budget row and the
entire final pass-12 reference frozen. No agents, new scenarios/reviews, native
writer/gate/reader training, deployment or Operating Table changes.

## Fixed experiment

Use all 1,500 selected receiver records from training release
`c70d4f44135a34c00e7fa0348833b1ec56613def15f5753231589492ed35478f`.
Sort by SHA256(`receiver-learning-v1:17:` + record ID). Complete **one full pass**:
1,500 unique records plus the first four actual records repeated to fill eight-
example windows, **1,504 presentations / 188 scheduled windows**. Record every
exposure and failure. This is an initial learning experiment, not the full
2,000-window component phase or a convergence claim. No elapsed-time cutoff or
automatic extension. The first 188 windows use the existing A4 component schedule,
including its 100-window warmup, rates, coefficients, clip and FP32 CUDA AdamW.

Train D adapters, Qev pointer/readouts, status/feasibility heads, Score row,
receiver input protocol modules and the shared Reason emitter/count head. Freeze
all other codec parameters. Reuse exact parameter ownership and eligible-example
VJP accumulation. No CPU model workload or fallback.

## Complete isolated objective

The existing component producer decodes an actual return before constructing
receiver losses. A malformed untrained Reason therefore discards valid status/
value supervision. The new isolated objective completes at the receiver heads:
construct a reviewed clean latent request, evaluate actual D, bind reviewed
labels to the actual bank/snapshot, build the losses, and end the declared
component. Release unused return holds without supplying or claiming a result.
There is no native/return objective or gold-return substitution. This is distinct
from truncating a failed model forward: any actual construction, numeric,
resource or forward failure remains an ineligible failed exposure.

Use the same neutral prompt as behavioral evaluation. Request inputs contain
only whitelisted request semantics and source-bound facts/rules. Access outcome
labels only after evaluation. Preserve Noul/Choice/Score losses on reference-
assessed examples even when actual status is wrong. Feasibility uses only
reviewed Boolean entry labels; generic unknown is not relabeled as a particular
four-way status. Reason semantic alignment is eligible only for matching actual/
reviewed non-assessed carriers. Its separate annotation may contain status and
support, following the existing Reason exception. The canonical reviewed Reason
packet has semantic slots without attachments, matching the clean-reader target;
its attachment-count CE is eligible with that Reason target. No head override or
runtime repair is used during evaluation.

Cache only frozen W/reference annotation payloads, keyed by the exact role-
whitelisted annotation hash. Verify cached, uncached and returned-copy payloads
are bit-identical on CUDA. **Always recompute D-dependent clean source states**
after parameter changes. Each actual Session retains one prefill, continuous
cache and exact accounting. Offline clean encodings are separate contexts.

## Verification and evaluation

Before stepping, probe actual existing receiver records covering every available
operation/status stratum. Require finite connected per-component gradients into
D, no reference/R gradients, complete accounting and zero optimizer updates.
Verify exact checkpoint/optimizer roundtrip after the first actual update.
Check frozen reference/R/other-codec hashes at checkpoint boundaries, W/tying
before and after, and preserve full optimizer/RNG/order/cursor state. Save at
0, 1, every 25 windows and the final boundary. No base-weight duplication.

Reuse the recorded 180-case X baseline only if initial weights, core model
sources, every permitted-view hash and cached/uncached reference payloads agree.
Measure the previously unmeasured **130 E development receiver cases** before
training. These include all five statuses. After training run all **130 E + 180
applicable X cases**, without gradients, updates, head overrides or gold returns.
Keep the 20 category-9 native-reader cases outside this receiver experiment and
leave locked labels unopened.

Report actual per-operation outcomes/statuses, rejected decodes, paired
interventions, and Score interval-distance/Noul calibration diagnostics without
widening approved intervals. Exact typed values/bindings remain mandatory.
Record all outputs, eligible denominators, loss/gradient evidence, before/after
identities and frozen checks. A completed learning run is not a passing receiver
qualification or proof of autonomous invocation/native answer consumption.

## Existing annotation limitation

Inspection of `train-receiver-0002` found a Reason text discrepancy: it requests
missing capacities for A and B, while the supplied facts give B's capacity and
omit A and C. The pinned release is retained unchanged for this experiment;
this observation is not a new independent review or corrected annotation.
Reason alignment to existing targets cannot establish correct support meaning.
Record such failures separately from status/value/selection learning and revisit
annotation quality before claiming semantic Reason qualification.

## Bookkeeping-only continuation

The original outer eligibility Counter accidentally summed live loss tensors;
Window's optimizer eligibility counts were unaffected. The fix counts component
keys and reconstructs totals from saved episode reports. Five regression tests
pass, including a comparison of the original and corrected update-loop AST after
excluding only accounting. At checkpoint 75 the process was intentionally
interrupted (exit -2) and exact weights, optimizer, RNG, sample order and cursor
were restored. All learning-module sources and non-source identities match;
only the launcher/accounting source migrates with explicit old/new provenance.
The continuation begins at 76; no completed optimizer window is replayed.
Unlogged in-flight preparation is not claimed as zero physical repeated work.

## Subsequent user-authorized audit

On 2026-11-07 the user explicitly requested one Astra subagent to investigate
CPU/performance bottlenecks. This overrides the earlier no-agent boundary only
for that read-only audit. It changed no training source, job, service or setting
and launched no additional model workload. See the
[performance audit](receiver-performance-audit-2026-11-07.md). The receiver
experiment's data, optimization method, frozen owners and endpoint are unchanged.
