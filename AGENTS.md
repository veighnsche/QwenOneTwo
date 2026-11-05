# QwenOneTwo

Read `docs/next-agent-handoff.md` for the current measured status, artifact paths,
remaining work and scope boundaries before continuing implementation.

All model workloads must run on an explicitly selected CUDA GPU. Use the RTX
3090 UUID GPU-225509c3-d702-6c27-de03-76b5e99b33b1. No CPU fallback for inference,
training, extraction, fitting or real-model tests. Ordinary static tests may use CPU.

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
