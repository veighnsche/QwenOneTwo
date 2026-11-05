# QwenOneTwo

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
No model caches, local configuration or generated artifacts in git. No commits,
publication or deployment without user authorization.
