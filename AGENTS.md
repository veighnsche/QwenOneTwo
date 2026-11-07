# QwenOneTwo

Read `docs/next-agent-handoff.md` for the current measured status, artifact paths,
remaining work and scope boundaries before continuing implementation.

As of 2026-11-08, the user has closed both GPUs again. Do not run GPU workloads,
device probes or GPU status tools until the user explicitly lifts this hold.
Continue only ordinary CPU editing, metadata analysis and static checks; CPU
model execution remains prohibited. Leave existing services unchanged. This
current hold supersedes the historical GPU authorizations recorded below.

All model workloads must run on an explicitly selected CUDA GPU. Use the RTX
3090 UUID GPU-225509c3-d702-6c27-de03-76b5e99b33b1. No CPU fallback for inference,
training, extraction, fitting or real-model tests. Ordinary static tests may use CPU.

The experimental two-GPU diagnostic runner may also target RTX 3060 UUID
GPU-744c1cf5-9578-6aad-95c1-7a6f1c9d11db in its explicit read-only worker scope.
The user lifted the noise-related GPU hold on 2026-11-07 for throughput work.
Read-only throughput diagnostics and device probes may run on these cards;
the user subsequently authorized the bounded existing-data receiver training
probe (32 training records, eight repeats) and its controlled loss-balance
comparison, and subsequently the frozen-representation Noul/Score head-only
experiment with Choice retention checks. Keep W/reference frozen, no agents
or new data.
General/full training remains unauthorized. Ordinary model/training
paths stay 3090-only; never use the 3060 as fallback. Diagnostic results cannot
grant accelerator or learned-flow qualification. Leave existing services alone.

The user subsequently authorized architecture-review slices 1–2: a CPU metadata
baseline and read-only information-loss diagnostics on the 3090, reusing existing
data with W/reference frozen, no agents or training. See
docs/architecture-baseline-2026-11-07.md and docs/information-loss-v1.md.
The user then authorized slice 3: one simpler latent-input comparison on the
3090 using existing data, frozen W/reference, no agents or training. The explicit
read-only profile/protocol is docs/simple-latent-comparison-v1.md.
The user subsequently authorized slice 4: one controlled Noul/Score receiver
repair on the 3090, preserving Choice, with existing data, W/reference frozen,
no agents or new data. See docs/receiver-residual-repair-v1.md for its finite
training scope; this does not authorize general training or automatic extensions.
The user then authorized the matched value-only comparison, using the saved
1,003 features and exact original initialization/exposure, with W/reference
frozen and Choice unchanged. Scope: docs/receiver-value-only-comparison-v1.md.
The user subsequently authorized the simple direct-readout diagnostic on saved
latent features, with W/reference frozen, training/development separate, no agents
or new data. Scope: docs/direct-latent-readout-v1.md; no production promotion.
The user then authorized a read-only training-to-X/latent-meaning audit using
existing data and saved features, with W/reference frozen and no agents, new
data or training. Scope: docs/training-x-semantic-audit-v1.md. Reuse recorded GPU
measurements; this audit does not authorize a further probe fit or model run.
The user subsequently authorized fixing rubric-direction hashing and one matched
numeric-access comparison on the 3090. See docs/numeric-access-comparison-v1.md:
existing X Score data, all W/reference/receiver weights frozen, no agents, new
data or training. This is a read-only diagnostic, not a production input change.
The user subsequently authorized the matched quantity/operator receiver-learning
comparison in docs/quantity-receiver-learning-v1.md: existing training data,
3090 only, W/reference frozen, development held out, no agents or new data.
This authorizes its two finite receiver fits, not general/native training.

One user submission, one prompt prefill, one hybrid cache. Qev decision LoRA is
on for the initial continuous budget readout, off for native generation, and
on temporarily when Qwen's trained internal gate requests Noul, Choice or Score.
Internal requests and feedback use hidden vectors, not generated JSON/tool text.
The runtime executes the learned gate; do not substitute keyword or regex routing.
The four gate actions are control operations, not thinking budget categories.

Thinking score is scalar regression in one actual LM-head row, clamped to [0,1].
Map it with floor((3*s*s - 2*s*s*s) * 131072 + 0.5), using the exact ordered
FP64 operations in docs/numeric-profile-v1.md. Total context is 262144.
Keep answer and thinking limits separate, cancellation and normal answering.
Mixed adapter cache reuse is experimental; preserve cache identity and report
that limitation honestly. Never claim stock Qev already learned latent invocation.
Training data and capabilities must have honest, reproducible provenance.
For this experiment, no human reviewers, paid review staff or human approval
gates are required. Use mechanical checks and one separate outcome-blind
model-assisted content review per record (batched in one call where practical).
Record model review as model review. Release approval may be automated after
checks pass; retain rejected/ambiguous records without automatic retry loops.

Independent of The Operating Table: never change its service, data or artifacts.
No model caches, local configuration or generated artifacts in git.

When explicitly asked to use Muse, default to one headless, tool-free call for
the whole requested batch: `muse exec --preset empty --max-model-steps 1`, low
reasoning, Contributor on. Disable reminders, web/shell/write tools, foreign
personal context and approval judging; use a fresh empty untrusted workspace.
The active `muse-implementation` skill contains the exact invocation. No
coordinator/worker framework, tmux orchestration, observers, timers or automatic
retry/repair calls. Codex validates and repairs locally. Additional model calls
or greater effort require explicit user direction. Saved run launchers under
`data/` are historical evidence, not default entrypoints; preserve both batches.

For user-authorized work in this repository, commit directly to `main` and push
when the work is complete. Do not create feature branches, pull requests or drafts.
This is standing authorization to commit and push that work; do not ask again.
Deployment still requires explicit user authorization.

Keep local experiment machinery simple. Finish the agreed finite work and reuse
valid completed measurements. Do not add agent-chosen elapsed-time, token or cost
cutoffs unless the user explicitly requests them; an efficiency preference is not
authorization to stop an agreed experiment early. Avoid repeating equivalent
validation when the relevant inputs and implementation have not changed.
