# Reference recovery isolation probe v1 — 2026-11-06

This bounded diagnostic separates the existing frozen reconstruction decoder's
local recoverability from the trained pooler's output. It does not extend either
reference fitting block, train capability weights or grant qualification. The
recovery revision and read-only persistent-conditioning candidate both failed.

Use only six existing **training** reference annotations: 0000/0010 (Question),
0001/0011 (Target), 0008/0018 (Key). Load/freeze the final recovery-v1 reference
`d0d4aef44b4415beb19311350eb24533190e1a27406c5d6f59a15a5a57402dec` on the pinned
RTX 3090. Leave W, pooler, codec, T and all live weights fixed. Each disposable
payload begins at its actual pooled value; optimize that payload alone for
**64 Adam steps, LR 0.05, betas 0.9/0.999, eps 1e-8**, ordinary causal
teacher-forced reconstruction CE. Wire normalization/role/slot metadata and
2/8 slot allocations stay unchanged. Process six cases serially, no batches,
no learning-rate search, random restarts or automatically increased steps.

At steps 0/16/32/64, record CE and the existing conservative autoregressive
exact-recovery result. Decode inputs are exclusively the current latent record
and preceding predicted/teacher-forced tokens as applicable; no source-annotation
text prefix, dictionary or target lookahead is added. Training labels may affect
the optimized diagnostic vector, so **any successful recovery is a fitted
positive control, never held-out performance or a deployable encoder**. No
optimized vector is installed in the model, data release or runtime. Do not fit
any development or locked record. Verify reference/live tensor fingerprints
before and after and that all model gradients remain None.

There are at most **384 latent-variable optimization steps, zero model-parameter
optimizer updates**. Record these separately from the completed 2,000-update
reference lineage / 2,016 physical reference updates including old setup work.

Interpretation: substantial/exact recovery improvement would show the fixed
T/wire path can express those training annotations and that pooling/training
needs investigation. Failure within this bound would not establish mathematical
impossibility, require new data, or justify relaxing the gate. Neither outcome
authorizes another automatic full fitting run. Record the actual result before
choosing a further causal implementation revision.
