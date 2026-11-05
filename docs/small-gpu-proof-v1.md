# Small GPU proof: G1-core v1

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

Owner: bounded G1-core scope, fixtures, trace/accounting, path-specific gradient
evidence, comparison matrix and exit bundle. Follow the
[specification](architecture-and-training-plan.md), [A2](model-configuration.md),
[F6 modules](module-and-optimizer-contract-v1.md), [A3](numeric-profile-v1.md),
[numeric verification](numeric-verification.md), [controls](control-flow-v1.md)
and [admission](position-limits-and-budget-accounting.md). Passing this subset
does not complete all A3, G1 or [promotion](promotion-matrix-v1.md) obligations.

## 1. Required demonstration and limits

One real backbone executes D prefill -> R request formation -> D evaluation ->
R result consumption/normal generation. R/D remain separately registered,
never stacked or merged. Independently demonstrate causal-state continuity,
native-loss connectivity THROUGH returned value/Qev/writer, and exact accounting.
Prescribed switches do not demonstrate learned invocation or semantic meaning.

Use Qwen/Qwen3.5-0.8B revision2fc06364715b967f1860aea9cf38778875588b17 and
twainsk/qev-0.8b revision1d2a6d7f891daab7483a49c3abe507c0c9d7ea62.
Select RTX3090 UUID GPU-225509c3-d702-6c27-de03-76b5e99b33b1 explicitly;
verify selected device/environment access before diagnosing unavailability.
No CPU model execution, parameter offload, second backbone or simplified
one-vector substitute. Ordinary static bookkeeping is allowed on CPU.

Use recorded A2/F6 modules and A3 CUDA equations/kernels. W stays frozen while
operations propagate gradients through adapters/inputs. Mode switching cannot
change requires_grad inside an episode. No service/checkpoint-pointer changes
or modifications to The Operating Table are part of this proof.

## 2. One causal state lineage

Track attention K/V, recurrent memory, convolution buffers and position counters
at every switch/packet boundary. Each step consumes predecessor state and emits
its successor; neither adapter gets independent history. No reset, replay,
rollback or separate decision prefill.

Training retains earlier state versions for backward; shared Python object ID
alone is not proof of continuity. Serving may mutate a container with the same
equations, without corrupting autograd-saved training state. Compare every
relevant layer/state family and monotonic positions, not just one final tensor.
Checkpoint recomputation is separately logged physical work; logical forward
still has one prefill and no duplicated ledger/publication/consumption events.

## 3. Two prescribed full traces

| Setting | Value |
|---|---:|
| Episode proof limit | 256 positions |
| P including chat template | <=80 positions |
| Evidence span m | 1..16 positions |
| Declared diagnostic B override | 128 |
| A | 16 |
| H reserve | 64 |
| Ordinary thinking in prescribed trace | 4 |
| Ordinary answer in prescribed trace | 8 |

C remains262144;256 is a proof limit, not a model-context change. Compute/log
actual initial budget score, keep predicted/granted/override separate. Override
does not rewrite predicted budget feedback. Define source/token IDs and exact
P/m/native token sequences in reviewed pinned-tokenizer fixtures before running;
never crop a packet or silently alter thresholds to fit.

Noul-during-thinking uses a short access-rule scenario. Score-during-answering
uses a short preparation scenario with explicit rubric. Prescribe discrete
operation, writer actions, Assessed carrier and native token sequence for numeric
comparison. All continuous Q/T/U/result tensors come from ACTUAL forward; no
gold scalar/reference latent packet substitutes. Evidence comes from original
prefill states, not a separately encoded prompt. These are prescribed whole-flow
infrastructure fixtures, not clean component/semantic/autonomy passes.

Score delivers two ordinary answer positions before call and six afterward;
latent work spends original B without reopening thinking.

## 4. Exact ledger

| Work | Noul thinking | Score answering |
|---|---:|---:|
| Ordinary thinking | 4 | 4 |
| Writer positions | 3 | 4 |
| Qev request | 14+m | 25+m |
| Assessed return | 14 | 25 |
| B spent | 35+m | 58+m |
| Answer A spent | 8 | 8 |
| H spent | 6 | 6 |
| Total with prompt | P+49+m | P+72+m |

Noul writer=start,ADD_EVIDENCE,FINISH. Score adds SET_RUBRIC. Q/T initialize
at start, without seven-action event. H=3startup+2closure+1termination; consumption
ticks=0. Upper bounds are145/168 positions. Heads/projections add no uncharged
backbone positions. Native prescribed token/termination origin is logged.

Admission holds request+MAXIMUM return: Noul64+m<=80;Score86+m<=102. Before
dispatch B remaining is128-7/128-8 respectively, sufficient for both holds.
Assessed return releases36 unused position holds in either case, never counts
released hold as spent or replenishes already spent B.

Without revisions Noul allocates Q/T/snapshot/Result=4 new records; Score adds
U=5. Selected source already exists; span is inline. Reserve Result+Reason until
outcome known, even though this Assessed carrier uses Result only. Validate
canonical handles and their independent holds using actual frozen K; no future
Result/Reason selectable as supporting evidence. Commit every position once.

## 5. Bounded bootstrap, not semantic training

New Score row initially zero can block native-loss path into Qev. Permit <=2
disposable bootstrap windows of8 episodes. Use declared numeric calibration to
move Score to an interior operating point, plus neutral native-continuation
objective. Calibration target is engineering operating point, not claimed
correct task assessment for unvalidated generated semantics.

Freeze diagnostic snapshot before comparisons. Require nonsaturated Noul and
interior Score in[.05,.95] on positive path fixtures. If bounded bootstrap fails,
report failure; no silent longer training or formula/temperature substitution.
Legitimate initial zero LoRA-factor gradients need post-bootstrap checks.

Subsequent explicit user authorization added a separate bounded
[gate-readout repair](g1-gate-repair-v1.md) after diagnosing the failed released
continuation. It preserves the original parent, freezes all but the two existing
gate tensors, records an explicit 19-row/256-step component fit and freezes a
new diagnostic child before repeating G1. This is an amendment for the proof,
not a silent extension of bootstrap or permission for capability training.
The original numeric limits remain unchanged; the rerun additionally requires
correct raw call/Continue logits and unforced executed-action telemetry.

## 6. Targeted gradient evidence

For native continuation loss L/returned scalar v, measure dL/dv; contribution
through v into receiver/readout; through v into actual request slots; and that
slot-mediated contribution into pre-call R/writer parameters. Also measure R's
native result-interpretation gradient. Isolate these with vector-Jacobian products
from history/echoed-field bypasses. Check active Qev adapter operations DURING
evaluation, not merely D's initial prefill parameter gradient.

Require finite measurable nonzero derivatives on positive fixtures under A3
near-zero/tolerance rules. Never count a matching pair of zero gradients as
connectivity. Discrete argmax has no ordinary derivative: explicit supervised
gate/writer/status/pointer checks remain distinct.

Negative controls: disconnect only scalar feedback, then separately request
payloads before receiver. Targeted checks must detect the intended break even
if shared-history gradients remain. These are labelled diagnostic executions,
never successful connected episodes or production graph modifications.

Verify native loss cannot reach detached budget feedback; budget row receives
its own scalar objective gradient; W remains unchanged; inactive scalar rows
stay grad=None and cannot drift through existing Adam momentum. Preserve proper
FP32 master/BF16 effective row distinctions and text-vocabulary exclusion.

## 7. Numeric comparison matrix

From identical diagnostic snapshots/prescribed inputs compare:

1. BF16 functional training, checkpoint off.
2. BF16 functional training, checkpoint on.
3. BF16 serving using same equations.
4. FP32 reference.

Two fixtures*four paths*seeds17/29/43*three repeats=72 core executions. Compare
switch/boundary states, scalar readouts, logits, gradients, bindings, exact
predicted/granted budgets and ledger events. A3 tolerances apply; wrong discrete
trace/accounting fails even when floats match. Run profiles sequentially, no
requirement for two resident backbones. One serving prefill per execution;
checkpoint backward recomputation remains separately reported physical work.

## 8. Actual generated continuation

Prescribe first call, then release subsequent gate/native-token decisions.
Compare functional/serving for both fixtures,three seeds,three repeats=36
executions. Continue from actual return into native answer tokens, without
another generation request/prefill; record text and stop outcome. First call
remains prescribed/B overridden: generation continuity, NOT autonomous evidence.

Healthy bound P80+B128+A16+actualH22=246<=256, using bounded notices/control
policy. No packet truncation needed. Empty/no post-return answer continuation
fails this requirement, not a semantic success invented from a cache trace.

## 9. Mechanical controls

Seed17,three repeats each=15 executions:

1. B=0: normal answer,no optional call.
2. Exact-fit Noul reservation: admitted.
3. One-position-short reservation: refuse before Qev evaluation.
4. Duplicate return: no position/second consumption.
5. Cancel during evaluation: no later delivery/closure; charge started work.

Noul exact-fit override B=71+m:seven pre-dispatch B positions leave64+m.
One less refuses. Expected mechanical cancellation/denial carries no invented
semantic label. Do not run forbidden CPU work to test its prohibition.

## 10. Evidence, count and pass rule

Record immutable hashes,hardware/kernels,forcing flags,complete event trace,
bindings,per-position debits,holds/releases,targeted gradients and generated
continuations. Measure peak GPU allocated/reserved memory separately for
forward/backward/checkpoint/optimizer buffers/reference. Actual fit is unknown.

After gradient checks one additional eight-episode verification window checks
F6 accumulation/eligible updates. No optimizer update inside unfinished episode.
Maximum complete executions=16bootstrap+72core+36continuation+15mechanical+
8verification=147. Local derivative/optimizer-state tests are additional small
GPU checks. All failed attempts and physical recomputation are reported.

Pass only when all required continuity,gradient,accounting,numeric and native
continuation checks pass. OOM,unsupported backward or corrupted state is a
blocker, not permission for CPU, second prefill or detached graph.

Passing permits broader GPU/codec qualification. Choice,non-assessed returns,
successive calls,full codec bounds and512/1024-position qualification remain
before full joint training. No claim of learned invocation,promotion readiness,
maximum context or optimal effort follows. The [post-proof expansion plan](post-proof-expansion-plan.md)
owns conditional sequencing after a measured pass.
