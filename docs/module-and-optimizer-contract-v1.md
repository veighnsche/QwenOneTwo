# Exact modules, optimizer ownership and accumulation v1

The opt-in [prepared receiver component v1 contract](prepared-receiver-component-v1.md)
versions RD-05 experimental admission, exact snapshot binding, joint objectives,
Reason masks and receiver ownership. It claims no reference qualification or
training authorization; legacy/native rules below remain in force outside it.

Rules and starting defaults. See the [current handoff](next-agent-handoff.md)
for implementation status, measured evidence and remaining work.

Owner: exact new protocol blocks, shared-module boundaries, missing heads/pointers,
initialization, optimizer ownership and eligibility-based accumulation. Specializes
[A2](model-configuration.md), [A4](training-run-profile-v1.md) and
[losses](loss-configuration-and-training-schedule.md); preserves
[B1](neural-input-encoding-v1.md), [control](control-flow-v1.md),
[resources](position-limits-and-budget-accounting.md) and
[specification](architecture-and-training-plan.md). F3/F7/F8 data/test decisions
are separate. This does not modify the legacy prototype.

## 1. Normalization and cross-attention equations

Frozen Qwen architecture is unchanged. New protocol operations are FP32. For
width d, m=mean(x*x), Norm(x)=x/sqrt(max(m,1e-12)), without learned gain/bias.
Use a safe constant-floor branch at zero, not sqrt(0) followed by a clamp whose
backward can be undefined. At the floor equality choose the constant branch;
smooth finite-difference checks exclude this nondifferentiable boundary.
This specializes A2's mathematically equivalent RMS floor 1e-6. Train/serve
must use the same declared operation order and A3 agreement checks.

For query q and available memory M, width256/eight heads/head width32:

    Q = Norm(q) W_Q + b_Q
    K = Norm(M) W_K + b_K
    V = Norm(M) W_V + b_V
    a = Concat(softmax(Q K^T / sqrt(32) + mask) V) W_O + b_O
    u = q + a
    Block(q,M) = u + W_2 SiLU(W_1 Norm(u)+b_1) + b_2

Head splitting/concatenation uses eight contiguous 32-wide heads. FFN is
256->1024->256. All projections have biases; dropout zero. Exclude padding;
no extra causal mask because memory contains already available records. With
empty memory set attention output exactly zero, bypassing output bias, while
still applying residual FFN. Never softmax an all-masked empty memory.

One block has 788736 trainable parameters. Protocol attention scale 1/sqrt(32)
is distinct from the 256-wide Qev pointer scale 1/16. No backbone norm changes.

## 2. Reference encoder and reconstruction

One shared reference pooler for all semantic roles:

    frozen W annotation states
    -> Linear 1024->256
    -> two cross-attention blocks
    -> Norm -> Linear 256->1024 -> canonical Norm

Its learned query table is 30x256:

| Rows | Meaning |
|---|---|
| 0-7 | Question |
| 8-9 | Target |
| 10-13 | Alternative |
| 14-15 | Rubric dimension |
| 16-17 | Rubric conditions |
| 18-19 | Zero anchor |
| 20-21 | One anchor |
| 22-23 | Midpoint |
| 24-25 | Attribute key |
| 26-29 | Reason |

Absent midpoint payloads are canonical zero and excluded from corresponding
semantic losses. Pooler parameter count 2110720 excludes codec/T. Reference
annotation contexts remain bounded/role-whitelisted; F7 owns permitted content.

Reconstruction uses the same frozen W with T. Prefix is latent record plus
role/slot/presence information, excluding source annotation and case-specific
identifiers. Predict canonical annotation using ordinary next-token teacher
forcing and end ID248046. After reference freeze use frozen T/reference codec,
retaining gradients to predicted inputs. This is a labelled offline auxiliary
context, not another prefill inside production inference or a third full model.

## 3. Shared native control trunk, draft reader and semantic emission

    f(h,c) = SiLU(W_ctl [Norm(h);c] + b_ctl)   # 1032->256

Share this first layer between gate and writer. Gate is Linear256->4 directly
from f. Writer feature is one Block(f,draft_memory); writer-action Linear256->7
reads that feature. Control flow determines when each head runs; WRITER_START
initializes records without a seven-action event.

Draft memory contains every committed current record in canonical order,
including semantic/exact/source contents. Project its FP32 encoded rows with
Linear1024->256. No fake request header/EVALUATE or missing-record placeholders.
Use an INTERNAL draft reference palette for ownership grouping, distinct from
sealed packet-local handles. Omit unpublished bindings such as final bank.
Rebuild the view deterministically after updates; it is small-reader memory,
not a separately executable wire packet. Record packing preserves full contents;
address summaries alone never replace them.

Next-tick feedback reads updated draft, previous native hidden state and
next-tick counters through the shared draft reader, then Linear256->1024, plus
encoded control metadata, then existing wire normalization. Append ONE tick.
Do not detach draft/feedback/history. Head computation is not another backbone
position; keep its expansion bounds visible.

For role/slot/attribute-conditioned 256-wide field query u:

    v = u + W_b SiLU(W_a Norm(u)+b_a) + b_b
    payload = Norm(W_out Norm(v)+b_out)

Use residual FFN256->512->256 and output256->1024, shared across roles/reasons.
Emitter parameter count 526080. Existing tag tables condition applicable fields;
absent contributions are omitted as in B1.

## 4. Complete bounded heads

Classification heads are one linear projection from the conditioned 256-wide
feature, unless listed as an MLP. B1 fixes class order; new subtypes below have
the explicitly listed order. They do not generate vocabulary tool text.

| Head | Output classes / dimensions |
|---|---|
| Record mode | 2: Construct, Reuse |
| Evidence kind | 2: SourceSpan, HistoricalResult |
| Child count | 9: 0-8 attributes/attachments |
| Attribute type | Existing 6 B1 types |
| Presence | 1 BCE logit, also midpoint presence |
| Attribute source mode | Existing 3 B1 modes |
| UTF8 length | 129: 0-128 |
| Source-span length | 33, present length0 masked |
| Decimal exponent | 37: -18..18 |
| Rubric direction | 3: None, Increasing, Decreasing; None masked for required rubric |
| Reason attachment kind | 3: Attribute, SourceEvidence, HistoricalResult |
| Shared byte classifier | MLP256->256->256 classes, SiLU between linears |

Record reuse is not attribute-value copying: a constructed parent with copied
attributes still allocates its parent Ref. Decode mode/count/kind, then type/
presence/source mode, required pointers/lengths, semantic/typed payloads, then
full structural validation. Apply masks once their inputs are known. Aggregate
size/UTF8 failure takes the mechanical path; never crop or change semantic
status to fit. All shape/byte work remains bounded, not another unbounded LLM.
New classification subtypes join existing structural loss weight0.5, without
another top-level coefficient. Byte/type/pointer loss ownership follows A4.

## 5. Source, object and clipping pointers

Nine separate bias-free Linear256->256 query projections for RecordReuse,
EvidenceSource, HistoricalResult, CopyTypedValue, UnitReference, ReferenceValue,
SpanStart, ClippingStart and ClippingEnd. Share one new bias-free key projection
1024->256 for source tokens and object/field summaries. Score dot product/16.

| Key source | Construction |
|---|---|
| Source identity | Mean of its available cached source states |
| Source token | That token's cached state |
| Object | Mean of its bounded typed presentation, including semantic/exact contents |
| CopyTyped entry | Mean of owner's complete attribute presentation; bind owner+exact field path |

Summaries locate entries; selected contents remain recoverable and are delivered
in full under codec bounds. No keyword subset. Hard selections need supervision;
no ordinary derivative through argmax is claimed. Preserve gradients through
selected live tensors and continuous values.

Span procedure: source pointer -> start-token pointer -> length1..32 -> derived
final token -> legal first/last byte-boundary pointers. No redundant learned
end-token pointer. Boundary key is MLP272->256->256 from 256 token-key features
plus 16 exact offset bits. Thus exclusive endpoint128 does not index beyond the
128-row slot table. Structural masks require verified original-source mapping.

Optional pointers have a fixed zero-key None entry; required pointers mask it.
This catalogue selector entry is not a Ref or wire handle0. Reuse/type/span
catalogues obey exact bound identities and the sealed K restrictions for reasons.
Contextual source states remain production data; F3 governs clean isolation.

## 6. Qev value, status, reason and feasibility wiring

Inherited Qev Q/K are separate from new native pointers. Use raw final EVALUATE
hidden state for inherited bias-free1024->256 pointer products/16. Noul uses its
two initialized prototypes; Choice keys are candidate-final hidden states.
Score/Budget stay actual separate LM-head rows with FP32 accumulation, without
extra scalar MLP or normalization before their dot product.

Shared evaluator trunk: Norm(h_eval) -> Linear1024->256 -> SiLU. Status is
Linear256->5. Shared semantic emitter produces Reason slots from this trunk;
shared typed heads/pointers produce attachments restricted to K. Feasibility:

    [q,k,q*k,q-k] (1024) -> Linear1024->256 -> SiLU -> Linear256->4

Use structural candidate masks, never gold winner/feasibility inputs. Result
status determines value presence; a largest pointer score proves no feasibility.
Valid return construction obeys all record/handle/position holds.

## 7. Exact optimizer ownership and initialization

| Group | Peak LR | Decay |
|---|---:|---:|
| Native R LoRA | 1e-4 | 0 |
| Decision D LoRA | 2e-5 | 0 |
| Inherited Qev Q/K matrices | 2e-5 | 0 |
| New projection/attention/FFN/head matrices | 2e-4 | 0.01 |
| New biases/embeddings/queries/Noul prototypes | 2e-4 | 0 |
| T LoRA, reference preparation only | 1e-4 | 0 |
| Budget master row, separate leaf | 1e-4 | 0 |
| Score master row, separate leaf | 1e-4 | 0 |

Keep A4 AdamW/betas/epsilon/schedule. Deduplicate by parameter identity and save
named ownership manifest. Frozen W has no optimizer membership. After reference
preparation freeze pooler/T/reference codec; compatible live codec copies have
independent storage/fresh optimizer state. Carry live moments through subsequent
component/joint phases. No trainability toggles within unrolls/adapter switches.

New linears: Xavier UNIFORM gain1, bias0. Tables/queries Normal(std0.02), reserved
role rows fixed0. R/T LoRA A Uniform[-1/sqrt(fan_in),+1/sqrt(fan_in)], B zero.
D/inherited pointer load pinned weights; Noul prototypes copy agreed frozen
word rows; scalar masters independently zero then bootstrap before autonomous
use. Every new query/prototype/bias has positive LR despite decay exclusion.

## 8. Exact eligibility-based accumulation

For example e/component c, let l_ec be its within-example normalized loss and
I_ec indicate valid supervision. The window objective is:

    L = sum_c lambda_c * sum_e(I_ec*l_ec) / sum_e(I_ec)

Omit components with zero eligible examples. Within examples average tokens/
bytes/fields as recorded; average semantic fields within role then eligible
roles. Sixteen alternatives cannot automatically outweigh one question. Reference
validity masks apply to the actual supported packet, not predicted abstention.

Use GPU per-component gradient buffers, not fixed division by eight:

1. Execute a complete main trajectory once; confirm whole-episode guards.
2. For each eligible component obtain its gradient separately.
3. Add to its FP32 CUDA buffer G_cp; increment eligible-example count n_c ONCE.
4. Retain that episode graph until required component derivatives complete.
5. Free it before the next example. After eight scheduled examples combine:

    grad_p = sum_{c:n_c>0} lambda_c * G_cp / n_c

Assign combined gradient once, check it, clip COMBINED global norm to1, then
one optimizer step. Example gradients2/6 over two eligible examples average4,
not (2+6)/8=1. Phase-specific lambda remains the loss owner's coefficient.

Use unused-gradient results None, without materializing zero: untouched params
keep grad=None and inactive scalar rows receive no momentum/decay update. A
genuinely active zero gradient is different. No count-first second rollout or
gold substitution is required. Pure checkpoint recomputation repeats no
registry/ledger side effects. Whole-episode invalid/overlength guards precede
buffer collection: no truncated-prefix gradient; failed exposures remain reported.
Numeric/device failure aborts window without partial optimizer update. A zero-
eligibility window skips parameter updates but advances scheduled-window counter;
do not silently replace examples. Auxiliary reconstruction may run field by
field so one auxiliary graph coexists with main graph, rather than all at once.

Buffer storage is 4*sum_c(P_c) bytes for participating FP32 component/parameter
pairs plus active graph/other state; allocate only participating buffers. Actual
GPU fit at1024 positions is UNMEASURED and remains G1/length qualification work.
No CPU model fallback or detach to work around memory failure is permitted.

## 9. Required future verification

Verify one owner per trainable tensor, frozen hashes, empty-memory/zero norm,
source/clipping bounds, combined-buffer gradient agreement, inactive grad=None
with existing Adam momentum, complete R-D-R gradients and checkpoint ownership/
optimizer restoration. Static dimension counts above were recomputed; no model
was instantiated and no GPU/gradient/optimizer test was run.

[PyTorch autograd.grad](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.grad.html)
supports separate returned gradients and preserving unused inputs as None. It
does not establish correctness/performance of this unimplemented trainer.
