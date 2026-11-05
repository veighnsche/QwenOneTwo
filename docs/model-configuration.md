# Model configuration: initial GPU pilot

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

Owner: dimensions, pooling/projections, normalization, adapter settings, heads,
precision, initialization and frozen/live component boundaries. Follow the
[specification](architecture-and-training-plan.md), [typed ABI](typed-value-abi.md),
[semantics](semantic-supervision.md), [losses](loss-configuration-and-training-schedule.md),
[scalar rows](scalar-row-handling.md) and [numeric checks](numeric-verification.md).

The recorded [exact module/optimizer contract](module-and-optimizer-contract-v1.md)
specializes this profile with safe normalization, block equations, shared trunks,
draft packing, complete heads/pointers, initialization, parameter owners and
per-component accumulation. It records F6's design response, not a passing
implementation or GPU memory measurement.

The user-authorized G1 numerical fix has a scoped
[compensated precision profile](g1-compensated-precision-v2.md): BF16 dense
products with FP32 accumulation/residual/state/backward arithmetic. It supersedes
the generic precision rows below for that proof candidate only, with unchanged
tolerances/weights. Broader capability-training/serving adoption is not performed
by the proof, and the legacy profile remains reproducible.

For the full-codec candidate, [feedback precision v3](codec-feedback-precision-v3.md)
supersedes both the generic precision table and the G1 dense-product zones:
frozen BF16 storage, transformer base/adapter FP32 arithmetic, FP32 incoming
residual/latent vectors and compensated BF16 native vocabulary projection.
Widths, normalization, scalar-row isolation and adapter settings do not change.
This is an arithmetic revision, not a claim that BF16 transformer products passed.

## 1. Preserve the backbone

Use the pinned Qwen3.5-0.8B, hidden width 1024, without changing supplied layers,
attention heads, recurrent dimensions, tokenizer or context configuration. The
protocol attention settings below do not change Qwen's own attention structure.
One base W is resident; no quantization in this version.

| Component | Precision |
|---|---|
| Backbone forward | BF16 |
| Adapter master parameters/optimizer state | FP32 |
| Adapter forward matrix operations | BF16 |
| Protocol projections/attention/losses | FP32 |
| Input vectors entering backbone | BF16 |
| Scalar-row accumulation | FP32 |

## 2. Adapter settings

| Adapter | Rank | Alpha | Dropout | Purpose |
|---|---:|---:|---:|---|
| Native R | 8 | 16 | 0 | Request writing, result reading, reasoning/answering |
| Decision D | 16 | 32 | 0 | Aligned Qev execution |
| Reconstruction T | 8 | 16 | 0 | Training-only semantic reconstruction |

Target matching text-backbone linear modules:

    q_proj, k_proj, v_proj, o_proj
    in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
    gate_proj, up_proj, down_proj

Exclude embeddings, LM-head rows and normalization parameters. Use no LoRA bias,
DoRA or rsLoRA. Initialize R/T with initialized A and zero B. Load D from pinned
Qev weights. Qev's original adapter dropout is 0.05; zero dropout is an explicit
aligned-profile change, with original artifact retained unchanged.

Runtime uses W+R or W+D. T is offline only, never stacked in deployed execution.
Use a new versioned checkpoint/profile; legacy rank-4 R/bridge artifacts do not
silently become compatible with this architecture.

## 3. Protocol normalization and input scale

Compute in FP32, per learned semantic vector:

    N(x) = x / max(sqrt(mean(x*x)), 1e-6)

No learned affine gain/bias in canonical N. Apply the same N to semantic targets,
predicted slots, reconstruction inputs and receiver payloads. Store canonical
slots; expose no extra unchecked raw-magnitude channel.
Implement the equivalent safe form x/sqrt(max(mean(x*x),1e-12)), selecting the
constant-floor branch at equality, as specified by the exact module owner.

Define one constant from the frozen base's valid tokenizer rows:

    sigma_E = sqrt(mean(E_valid_token_rows * E_valid_token_rows))

Compute on the selected GPU, record it in the manifest and do not recompute from
changing protocol weights. A neural wire position is:

    x = sigma_E * N(N(payload) + N(projected_tags))

Tags identify role, slot and applicable owner handle. Missing contributions are
zero. Cast the finished vector to BF16 for the backbone. This applies to protocol
inputs, not Qwen's native token embeddings or its internal normalization layers.
Do NOT normalize budget/Score row dot products or add a head-specific scalar bias.

## 4. Answer-blind reference pooling and reconstruction

Encode reviewed canonical annotations through frozen W with adapters disabled.
Pool [length,1024] states using learned role/slot queries and padding masks:

| Setting | Value |
|---|---|
| Memory projection | 1024 -> 256 |
| Pooling blocks | 2 cross-attention blocks |
| Heads / head width | 8 / 32 |
| Feed-forward | 256 -> 1024 -> 256 |
| Activation / dropout | SiLU / 0 |
| Output lift | 256 -> 1024, then N |

Preserve slot order. Outputs: question 8; alternative 4; each of five rubric
fields 2; target 2; attribute key 2; reason 4. Question/rubric inputs are answer-
blind. Reason annotations may include justified status/evidence. Canonical
annotations are limited to 256 tokens per record; revise/review oversized text,
never silently truncate it.

The [training-input rules](training-input-rules-v1.md) specialize this content
boundary for every role and training branch; Reason is post-decision and may
contain its justified non-assessed status. No whole annotated scenario row is
an encoder input. Reference preparation uses semantic, not decision-outcome losses.

Reconstruction uses W+T and Qwen's existing decoder/vocabulary head, not another
full decoder. Its input is the allowed latent record and roles, not the source
annotation. Apply the scalar/private-vocabulary exclusion to reconstruction too.

## 5. Runtime writer and memory

Protocol working width is 256. The writer trunk receives normalized native
hidden state plus eight normalized counter/phase features:

    [1024 hidden + 8 counters] -> 256 -> SiLU

Features cover remaining B, A, H, context, writer steps, attempts, object slots
and thinking/answer phase. [B1](neural-input-encoding-v1.md) fixes denominators
131072/262144/64/262144/32/16/32 and phase 0/1, plus exact sampling timing.
Validate ranges; do not guess different mappings in training/serving. No
extra human/tool text is needed to supply them.

Draft reader: one cross-attention block, eight heads of width 32, FFN
256->1024->256, SiLU, zero dropout. Read full bounded draft tensors; pooled
address summaries never replace recoverable contents.

For parallel semantic outputs, combine the 256-wide writer feature with role,
slot and attribute query embeddings, then use residual FFN 256->512->256,
output projection 256->1024 and canonical N. Updated draft feedback is lifted
256->1024, added to encoded B1 control metadata in FP32, and encoded as the next
WRITER_TICK: one backbone position. WRITER_START uses encoded control metadata.

Source/object pointers use 256-wide queries/keys over source states or typed
catalogue entries. No keyword-selected subset. Source-boundary pointers range
over legal byte boundaries of selected tokens, with structural masks. Original
source/history and draft tensors retain required training gradient connections.

## 6. Tables and input projections

All query/tag tables have working width 256:

| Table | Entries |
|---|---:|
| Protocol roles | 64 |
| Slot/byte positions | 128 |
| Local owner handles | 256 |
| Attribute indices | 8 |
| Writer events | 7 |

Project applicable tags 256->1024. Only defined roles are legal; unused rows are
not new actions. Opaque identities are not semantic labels.

| Payload | Projection |
|---|---|
| Canonical semantic slots | Identity, already 1024-wide |
| Cached source states | 1024 -> 256 -> 1024 |
| Discrete atoms | 67 -> 256 -> 1024 |
| Fixed metadata features | 128 -> 256 -> 1024 |
| Continuous task value | 3 -> 128 -> 1024 |
| Budget control value | Separate 3 -> 128 -> 1024 |

MLPs use SiLU and biases. Atom features are 64 bits plus deterministic sign,
bounded log-magnitude and decimal-exponent features; non-numeric atoms use zero
numeric features. Derive these from the canonical value, not free writer floats.
Exactness comes from the bit payload/record, not approximate magnitude features.

Continuous encoders take [1,value,value*value]. Constant input and biases prevent
normalization from merely erasing positive scalar magnitude. Task returns retain
gradients; budget feedback has its deliberate boundary. Metadata input mapping
must be deterministic, reserved fields zero, with no free-form outcome labels.
The recorded [B1 profile](neural-input-encoding-v1.md) fixes role IDs, all 128-bit
metadata layouts, ordered tag addition, numeric helper formulas, counter timing
and prototype initialization IDs. Metadata becomes 128 FP32 binary features.
Matching reference vectors/codec interoperability still require implementation
and GPU validation; do not confuse the documented map with passing results.

## 7. Heads

| Head | Shape |
|---|---|
| Native operation gate | 1032 -> 256 -> 4 |
| Writer action | 256 -> 7 |
| Attribute count | 256 -> 9 |
| Attribute type | 256 -> 6 |
| Presence | 256 -> 1 |
| Source mode | 256 -> 3 |
| Record Construct/Reuse | 256 -> 2 |
| Evidence SourceSpan/HistoricalResult | 256 -> 2 |
| Rubric direction | 256 -> 3 |
| Reason attachment kind | 256 -> 3 |
| UTF-8 length | 256 -> 129 |
| Source-span length | 256 -> 33, illegal lengths masked |
| Decimal exponent | 256 -> 37 |
| Shared byte classifier | 256 -> 256 -> 256 classes |
| Result status | 1024 -> 256 -> 5 |

Byte outputs are index-conditioned/parallel, with ABI caps rather than a hidden
autoregressive generator. Structural masks do not supply semantic decisions.

Retain Qev's 1024->256 query/key architecture, bias-free, with this profile's
dot-product scale 1/sqrt(256). Align/train the versioned pointer; claim no
inherited calibration. Noul has two trainable false/true prototypes initialized
from corresponding frozen vocabulary embeddings; Choice uses candidate-boundary
states. B1 fixes false/true initialization rows 3721/1802, independent FP32
prototypes with no pre-projection normalization, and p_true=sigmoid(logit_true-
logit_false). Candidate feasibility concatenates projected query, key, product and
difference (1024 features), then 1024->256->4 for feasible, infeasible,
insufficient and conflicting evidence. No feasible choice is masked for non-Choice
global status outputs. Budget/Score remain two actual scalar rows, no extra MLP.

## 8. Frozen reference versus live codec

After reference preparation, freeze/version reference pooling/projections, T and
the reference codec used for reconstruction. Live writer/runtime codec remain
trainable; initialize compatible small tables/projections from the snapshot,
without sharing mutable parameter storage. R/D share ONE live runtime codec.

The frozen reference uses the same W, not another full backbone. It can be
omitted from serving. A frozen decoder still propagates gradients to predicted
latent inputs. Updating live codec weights must not mutate reference meaning
targets or the frozen reconstruction path.

## 9. Initialization and validation

New linears use Xavier uniform initialization, gain 1 and zero biases; embeddings use standard
deviation 0.02. Scalar FP32 master rows start independently at zero and bootstrap
before autonomous reliance; zero is not an uncertainty label. All protocol dropout
is zero. Preserve frozen pretrained weights, static adapter registration and the
recorded row/gradient ownership rules.

These are concrete pilot settings, not measured fit/performance. Recorded
[A3](numeric-profile-v1.md) and [A4](training-run-profile-v1.md) fix initial numeric
and training defaults; [B1](neural-input-encoding-v1.md) fixes exact neural input
encoding. Actual manifests/reference vectors/GPU validation remain unperformed. No
model was instantiated, trained or executed by recording this configuration.
