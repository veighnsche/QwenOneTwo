# Cache and gradient strategy

Status: recorded design strategy, with GPU feasibility still unverified. This
document authorizes no implementation, training, service change or deployment.

Owner: state/gradient strategy. Follow the
[specification](architecture-and-training-plan.md) for the common lifecycle and
[handoff G0/G1](implementation-handoff.md) for codec freeze and feasibility exit
criteria before full data production or training.

The [numeric verification plan](numeric-verification.md) now owns explicit
forward/gradient tolerance defaults, exact control parity, local FP64 checks
and the requirement to verify intended return-path gradients separately.

## One causal history with two explicit modes

Use one shared frozen Qwen backbone W, a trainable native adapter R, and a
decision adapter D initialized from the pinned Qev checkpoint. D may be updated
as a versioned adaptation for the latent protocol; preserve the original Qev
artifact. Native mode is W+R and decision mode is W+D, not W+R+D.

The mode sequence is:

1. W+D: one prompt prefill and the fixed initial continuous budget readout.
2. W+R: native reasoning; the operation-intent gate starts a bounded latent
   draft, then learned completion triggers runtime seal/admission.
3. W+D: process the sealed latent request and produce its typed result.
4. W+R: consume the result and continue reasoning or the normal answer.
5. Repeat steps 2–4 when the learned policy invokes another operation, within
   the recorded budget and attempt limits.

The state includes attention keys/values, convolution buffers, recurrent memory
and position counters. Switching changes transformations for subsequent inputs;
it does not rewrite or remove state produced under a previous mode. No serving
reset, cache fork, rollback, replayed prefix or second prompt prefill is assumed.

This is a purpose-trained alternating-mode model. A shared backbone or matching
cache identity does not establish compatibility with stock Qev or equivalence
to a history computed entirely in either mode. Both modes must learn from the
actual mixed history. Ordinary adapter switching is not a compatibility proof.

## Serving versus training state

Serving may use one mutable working cache under inference/no-grad execution.
Training instead needs differentiable updates that preserve earlier state
values required by backward. Attention appends and convolution/recurrent updates
must not overwrite tensors saved for autograd.

The continuity rule applies to trainable task/history paths. Frozen semantic
reference targets and startup budget control-feedback values are explicitly
detached by the [loss configuration](loss-configuration-and-training-schedule.md).
Do not mistake these intentional boundaries for permission to detach task
Noul/Score returns or the native/decision causal history.

There is still one causal state lineage. Saved activations or checkpointed
states are training memory, not a separately prefetched decision-model cache.
Do not require a full deep copy of the whole prefix on every step. The exact
functional state layout and GPU checkpoint boundaries remain implementation
details that need validation.

GPU activation checkpointing may recompute forward segments during backward.
This must reproduce the original explicit mode. It is not a second serving
prefill and must be reported distinctly. Recomputed functions must be free of
side effects: no duplicate memory writes, attempt charges or result consumption.

Immutable request snapshots preserve gradient connections. Copy tensor storage
when immutability requires it, but do not detach questions, alternatives,
rubrics or returned values at the call boundary.

## Adapter selection and trainability

Keep R, the adapted D and protocol heads/writers/readers registered as trainable
throughout a joint unroll. Use explicit, mutually exclusive mode selection in
the forward computation. Do not rely on a global adapter switch that changes
requires_grad during the trajectory. Do not merge/unmerge or overwrite weights
inside an active training graph.

W remains frozen, but its computations must propagate gradients from trainable
inputs and adapters. Blanket no-grad execution through W would cut those paths.
Budget/Score scalar rows are explicit learned extensions to the frozen
pretrained weights. The [scalar-row design](scalar-row-handling.md) records
tying, masks and precision/export; verification remains a G0/G1 deliverable,
not permission to unfreeze the whole vocabulary matrix.
Take optimizer steps after backward through the complete training episode;
gradient accumulation may combine complete episodes.

## Continuous gradient paths

The intended answer-loss path for a continuous result is:

    native continuation / answer
      -> native result reader and result encoding
      -> decision numeric readout and D computation
      -> latent request writer
      -> earlier native computation

There are also gradient paths through the causal state history. Update native
R on result interpretation, rather than fitting only a bridge while the entire
native path stays frozen. Keep semantic request/outcome/binding supervision;
answer loss alone can reward ignoring the return or encoding a desired verdict
instead of a meaningful question.

## Discrete boundaries

| Boundary | Training requirement |
|---|---|
| Continue/Noul/Choice/Score action | Explicit action/continuation supervision; no ordinary derivative through argmax |
| Choice winner or reference index | Selection, feasibility and binding supervision; no gradient through integer selection itself |
| Semantic status and value presence | Status supervision; numeric loss only on assessed values |
| Continuous probability or Score | Numeric and downstream interpretation losses where supported |
| Budget score, rounding, admission and stopping | Own score supervision; stopping/admission are not ordinarily differentiable |

Gradients through a selected continuous payload do not imply a gradient through
the selection decision. Do not train on soft mixtures and serve hard selection
without declaring and resolving that mismatch. Teacher-forced calls can teach
skills but cannot demonstrate autonomous invocation.

End-to-end training means complete trajectories, connected continuous paths and
explicit supervision at discrete boundaries. It does not mean answer loss alone
teaches every switch. Any policy-gradient stage requires a separately agreed
objective and plan; it is not silently part of this strategy.

## GPU feasibility gate

The recorded [small GPU proof protocol](small-gpu-proof-v1.md) specializes the
first bounded G1-core demonstration and isolates native-loss contributions
through the returned scalar from shared-history bypasses. It remains unrun;
Choice/all-status/multi-call/512/1024 qualification and semantic promotion are
separate subsequent requirements in the [expansion plan](post-proof-expansion-plan.md).

The installed FLA fused recurrent inference path raises NotImplementedError for
backward; the installed recurrent cache update also uses in-place copy_. A
differentiable GPU path with training-safe state updates is therefore required.
Earlier bounded prototype fixes do not establish this whole architecture.

Start with short complete call/return episodes on the explicitly selected RTX
3090 only, microbatch one as needed. Expand to multiple calls and longer histories
after the complete short path works. Do not detach at the call boundary or
truncate away result use while claiming whole-flow training. Short unrolls do
not establish operation near the maximum context or reasoning budget.

Before claiming feasibility, verify:

- Exactly one serving prefill and uninterrupted, monotonic state progression
  across decision/native/decision/native transitions.
- Training and serving forward behavior match under the same mode/input
  sequence, within numerical tolerances declared before examining outcomes.
- Finite effective gradients reach request formation, receiver alignment and
  native result interpretation; no accidental freezes or detach boundaries.
- Checkpoint recomputation uses the correct mode without duplicated side effects.
- Held-out result interventions show semantic use, alongside free learned calls.

If the gate fails, report the failure and revisit the architecture. Do not add
CPU workloads, a second prefill or a second full model as an unannounced workaround.

Reference: [PEFT cache-sharing documentation](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora#using-and-reusing-cache-for-generation)
warns that adapted history cannot simply be treated as ordinary base history.
It does not validate this proposed purpose-trained alternating system.
