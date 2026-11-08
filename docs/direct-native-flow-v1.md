# Direct native flow v1 — experimental one-cache path and masks

RD-06 adds `EpisodeProducer(..., packet_profile='direct-role-states-v1')`.
The [direct layout](direct-role-states-v1.md) remains opt-in, with unchanged
32-state fields, 256 request /64 return positions and existing record/event/
budget/context/cancellation rules. Default legacy controls remain available.
Implementation delivery is separate from GPU qualification and useful learning.

## Execution and targets

The live sequence is original prompt prefill -> native R writer -> actual D
evaluation -> bound latent return -> native continuation in the same cache.
Source states are `h[0, selected]` from that actual prefill, with original token/
byte mapping. No source re-encoding, annotation injection, gold return or second
live prefill is permitted. The ordinary unforced writer uses existing Decoder
emissions; guided direct emission also uses its predicted length heads and raw
live vectors. There is no gradient through argmax lengths/actions/pointers.

Optional training-only direct targets are preflighted and encoded before Session
creation. `request_targets` uses the frozen-W clean source helper for complete
role-whitelisted annotation content states, excluding template positions. Every
field, full request and worst-case return is checked before any target forward.
Unsupported targets remain explicit exclusions. Only actual prefill source Refs/
hashes are attached later, without encoding inside the call. The old pooled
reference target policy remains a named control; no reference qualification is
fabricated or transferred to the direct profile.

Direct formation uses applicable shape-matched alignment and supported discrete
length/structure targets, never the unsupported variable-layout reference
reconstruction objective. Target lengths may supervise logits; actual emitted
lengths remain predicted. Rubric subfields use explicit offsets and absent
midpoint omission. Targets neither define the meaning of arbitrary writer outputs
nor admit outcome/continuation losses through an exact-reconstruction gate.

## Conservative supervision

Native direct decision labels and post-call approved continuations remain masked:
the current data supplies no verified semantic bridge for arbitrary native
writer vectors. Similarity, copied target IDs, expected values and alignment
success do not waive this condition. New-profile Reason targets remain unsupported.
Incorrect/unbound returns receive no incompatible approved continuation or
invented correction. A guided prefix stops at its first incompatible return.

`NativeEpisode` provides explicit actual-prefill provenance for accumulation. It
allows applicable control/formation losses only, rejects decision/Reason/post-call
native losses, and rejects development accumulation. The original diagnostic-
reference guard remains first. Approved direct/language-replay no-call native
targets are separate; they do not establish call/return understanding. Prepared
receiver components and supplied-return readers retain separate admission paths;
the native direct producer cannot silently substitute either one.

All forcing is recorded: budget override, guided control prefix, construct/no-
attribute choices, operation/candidate count/rubric direction and full-source
spans. Semantic vectors and role lengths remain writer outputs. Native free
tokens carry phase/control tags; approved teacher-forced tokens are separate.
Actual returned status/value/selected Ref is recorded with snapshot/result binding.
No invocation text is introduced. The reference stays immutable and W stays frozen.

## Eligibility, evaluation and qualification

The [RD-06 report](redesign-rd06-results-2026-11-08.md) inventories all 5,590
existing relevant records. All 3,220 writer/interleaved calls have zero complete
direct formation eligibility under current limits; every worst-case return
exceeds 64. Native semantic decision/post-call continuation eligibility is zero.
Direct supplied-reader support is also zero. These are explicit later-fit
blockers, not permission to crop, increase bounds or invent annotations.

Held-out native W/F evaluation uses explicit `evaluation=True`, development only,
`free_joint`, original prompt/source and no gold target, effort target, guided
control or override. Development remains excluded from training accumulation.
Evaluation manifests keep native W/F, prepared-receiver X pairs and supplied-
reader R/X9 cases separate. Existing prose continuations provide no typed final-
answer extractor. `native_evaluation` can grade separately supplied typed native
observations using original values/identities/intervals, but cannot establish their
extraction provenance or infer them from receiver heads or consumed returns.
Missing native observations remain unsupported. No new parser/judge is invented.

Two implemented checks remain pending RD-07:
`RD06-GPU-NATIVE-RETURN` and `RD06-GPU-CONNECTED-GRADIENT`, in
`tests/test_native_redesign_gpu.py`. Require submitted RD-07, explicit
`QOT_RD06_GPU_QUALIFY=1`, the pinned snapshot and exact RTX3090 UUID before model
imports. The return fixture covers Noul/Choice/Score with disclosed guided
controls, Assessed status and one existing continuation token. The separate
Noul VJP fixture proves the return-specific native-loss -> receiver -> writer/R
path through frozen receiver weights. No optimizer step occurs. These are
engineering fixtures, not valid corpus continuations or learned behavior.
All eight earlier deferred checks also remain pending. RD-06 runs no model work.
