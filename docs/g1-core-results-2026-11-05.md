# G1-core GPU results — 2026-11-05

**Overall result: FAILED. Do not expand the codec, curriculum, training or UI.**

The user authorized implementation/execution of G1-core only. The isolated
runner executed on RTX 3090 UUID GPU-225509c3-d702-6c27-de03-76b5e99b33b1.
All model forward/backward, fitting, tensor comparisons and derivative checks
ran on CUDA. Static ledger/bit tests and file/tokenizer processing were model-free.
The running service, current capability checkpoint, frontend and Operating Table
artifacts were not changed. Existing local model artifacts were read only.

## Implementation and evidence

The runner is `python -m qwen_one_two.g1_proof`, with isolated
[functional backbone](../qwen_one_two/g1_backbone.py),
[exercised protocol](../qwen_one_two/g1_protocol.py),
[accounting](../qwen_one_two/g1_accounting.py),
[numeric checks](../qwen_one_two/g1_checks.py) and
[supplementary evidence](../qwen_one_two/g1_evidence.py).
It uses the pinned real Qwen text backbone/Qev adapter and F6-shaped writer/
protocol modules, not the legacy one-vector bridge. R/D contribution is explicit;
switching does not toggle trainability. Both share one functional hybrid state.

Artifacts are local ignored files under `data/g1-proof/run-20261105/`:
`report.json`, `manifest.json`, `fixtures.json`, diagnostic/verification
safetensors and row export. Logs `data/g1-proof-*.log` and timestamped prior
reports preserve failed setup/measurement attempts. No model/data artifact was
committed, published or deployed.

Final report SHA256:

    dfb7ff98a0bd91c8e5c1992c6fb247816ce94f14734246a7f929a69e747cb35c

The final supplementary run source hashes matched the implemented G1 files.
Manifest records software/hardware and artifact hashes. Earlier diagnostic
attempts used intermediate implementation revisions and are not silently pooled
as certification of the final source. Initial setup fixed missing nonpersistent
RoPE buffers and explicitly selected cuBLASLt, required by Torch's no-split-K
flags. No numeric threshold was widened.

## Cache and accounting: passed measured fixtures

| Fixture | P | Evidence m | B spent | A spent | H spent | Total | Logical prefills | Unused hold released |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Noul during thinking | 33 | 9 | 44 | 8 | 6 | 91 | 1 | 36 |
| Score during answering | 52 | 9 | 67 | 8 | 6 | 133 | 1 | 36 |

Both follow D -> R -> D -> R, one lineage, monotonic positions and original
source states. Each past-state tensor's version is checked across forward
transitions:1008 checks for Noul/1056 for Score. Attention K/V lengths match
total positions; convolution/recurrent state stays on the same lineage.
Return is consumed once; no extra consumption ticks; no remaining position hold.
Core new-record counts are4/5. Answer-phase Score spends original B.

BF16 training versus no-grad serving matched all sampled boundary states and
scalars exactly. Checkpoint off/on matched measured gradient vectors exactly
(normalized error0,cosine1); logical accounting did not change. Physical layer
calls increased from504 to1800 Noul and528 to2112 Score under targeted checkpoint
VJPs, explicitly distinguished from logical prefills/positions.

All15 GPU mechanical controls passed: three repetitions each of zero B,
exact-fit reservation, one-short refusal, duplicate delivery and cancellation
after evaluation work/before return. These are bounded engineering controls,
not exhaustive concurrent-service race testing. A sealed snapshot allocation
is retained even on budget denial; no fake semantic result is created.

## Connected gradients: passed measured paths and negative controls

BF16 native-continuation derivatives:

| Measurement | Noul | Score |
|---|---:|---:|
| dL/d(returned scalar) | 0.13539115 | 0.12075034 |
| Request-packet VJP RMS | 4.2246e-4 | 1.00294e-2 |
| Semantic-slot VJP RMS, through packet | 3.66994e-6 | 1.38940e-4 |
| Pre-call R gradient RMS, through slots/packet/scalar | 6.19284e-6 | 3.13882e-4 |
| D gradient RMS, through returned scalar | 1.41865e-5 | 2.72582e-4 |

All are finite/nonzero. Adapter operations captured DURING Qev evaluation and
native return reading have nonzero gradients too. Slot VJPs explicitly pass
through request_packet, excluding writer-feedback/shared-history bypasses.

All four negative controls passed: scalar feedback detachment removes the scalar
path; request-payload detachment removes the packet-mediated slot path, for each
operation. Native loss leaves budget row grad=None. Its own diagnostic scalar
objective has a connected gradient. Noul continuation leaves Score grad=None;
Score continuation reaches Score. These are connectivity results, not semantic
understanding or learned invocation.

The eight-example optimizer verification matched an independent CUDA FP64
aggregation of component VJPs: normalized error5.1172e-8,cosine approximately1.
Eligible denominators were native8/discrete8/calibration4. R,D,Score changed in
the diagnostic update; budget stayed unchanged. Masters were restored afterward.
An earlier attempted BF16 combined-loss-backward oracle differed1.8308%; it
is retained as a diagnostic, not mislabeled an FP32 arithmetic reference.

All320 frozen text-backbone parameter tensors matched original checkpoint data
on GPU. Shared effective scalar-row provider/base tying, inactive rows with
existing Adam momentum, private-row exclusion and row export checks passed.
Identical-hidden-input exported row readouts had zero difference. This does not
certify an actual production installation or full optimizer checkpoint recovery.

## Cross-precision A3 agreement: FAILED

FP32 is the reference in these normalized comparisons:

| Error | Noul | Score | Allowed |
|---|---:|---:|---:|
| Worst hidden/cache normalized RMS | 0.0366066 | 0.0867262 | <=0.01 |
| Semantic-slot gradient normalized RMS | 0.177832 | 0.464958 | <=0.05 |
| Writer-R gradient normalized RMS | 0.160356 | 0.635150 | <=0.05 |
| D gradient normalized RMS | 0.119125 | 0.448349 | <=0.05 |
| Native-R gradient normalized RMS | 0.107472 | 0.343601 | <=0.05 |
| Returned-value absolute difference | 0.00007737 | 0.01068449 | <=0.002 |

Pointwise state/logit failures and gradient-cosine failures are recorded too.
Similar Noul output does not excuse the large state/gradient differences.
GPU FP64 local1/2/8/16-step smooth derivative checks passed; they do not override
the full-path cross-precision failures.

Peak allocated memory in the final measured cases: Noul/Score BF16 gradients
9.81/10.65 GiB; checkpointed6.93/6.13 GiB; no-grad serving2.94/3.00 GiB;
FP32 reference15.46/15.44 GiB. This is measured process allocation, not a
full-context capacity claim or total GPU use including the existing service.

## Released continuation: incomplete/blocked

Noul's released gate selected Choice in both functional/serving runs. The G1
subset does not implement Choice, so it explicitly failed instead of silently
forcing Continue, masking the decision or expanding the codec.

Score produced identical non-empty native continuation in functional/serving:

    The user is asking for a fraction

It was preceded by newline tokens and hit the16-position answer limit. Totals:
B113,A16,H21,P52,total202;16 attempts, one actual dispatched/consumed return.
The other attempts were denied with bounded notices. This is generation
continuity, not a complete or semantically acceptable answer. An earlier
whitespace-only result exposed an insufficient acceptance check; it was corrected
and rerun, with proper BudgetExhausted notice metadata. Earlier output is retained.

## Gate decision and limitations

The full72-case numeric matrix stopped after the first failing reference
comparison; supplementary runs measured both operations and independent controls.
The full36-case released-continuation matrix is not certified. Partial repetitions
are not treated as complete coverage. The two-window bootstrap and extra update
were disposable diagnostic exercises, not A4 capability phases or a reviewed
training dataset. Operation/status/carrier controls and budget override were
prescribed in core comparisons; no autonomy, calibration or optimal-effort claim.

Resolve the numerical disagreement and released-branch/proof-scope failure,
then rerun the required G1 matrix. Preserve the current thresholds until a
separately justified/versioned design change; never manufacture a pass by
loosening them, using CPU, another prefill or a detached graph.
The [post-proof expansion gate](post-proof-expansion-plan.md) remains closed.
