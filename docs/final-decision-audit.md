# Final design consolidation and pre-implementation audit

Status: documentation audit, not an implementation/GPU readiness report. No
code, model workload, service, checkpoint, commit or deployment is authorized.
This file owns the closure assessment and residual list; the
[specification](architecture-and-training-plan.md) owns the target contracts.

## Recorded decisions

All seven recent topics have an owner and explicit initial design:

| Topic | Owner | Recorded choice |
|---|---|---|
| Packets | [Codec](latent-packet-codec.md) | Typed immutable records, role vectors, question/rubric echo and continuous values |
| Writer | [Schedule](latent-writer-schedule.md) | Parallel record heads per native step, learned pointers, draft feedback and FINISH |
| Semantics | [Supervision](semantic-supervision.md) | Answer-blind reference encoding, frozen targets/decoder and behavioral interventions |
| Losses | [Configuration](loss-configuration-and-training-schedule.md) | Normalized coefficients/masks, valid-set Choice, shrinking reviewed prefix and actual joint returns |
| Scalar rows | [Handling](scalar-row-handling.md) | Private IDs 248077/248078, frozen shared table, independent FP32 masters, BF16 effective rows and FP32 readout |
| Numerics | [Verification](numeric-verification.md) | Exact invariants/control parity plus forward/gradient tolerances and export checks |
| Data operations | [Procedure](data-operations.md) | Independent semantic/outcome review, provenance, immutable sources and release rules |

Requirements, starting defaults, unreviewed labels and unverified feasibility
are distinct. No coefficient, capacity or score becomes empirically optimal
by appearing in the spec. The three full walkthroughs remain preserved.

## Contradictions and ambiguities reconciled

| Issue | Consolidated interpretation |
|---|---|
| Writer still calls semantics/losses unspecified | Link recorded owners; retain genuine residual implementation details |
| No detach versus detached budget feedback | Preserve task-result/history gradients; budget feedback and frozen reference targets are explicit intentional boundaries |
| Base frozen versus scalar-row replacement | Frozen pretrained storage plus two row parameters in training; only private rows installed before serving |
| FINISH revision versus sealed immutability | Structurally invalid drafts remain unsealed/revisable; sealed snapshots never edited; resource non-admission ends that attempt |
| Attempts versus calls versus writer ticks | Count attempt on entry, D call on dispatch and each native writer position separately |
| New records versus inline fields | Count allocated immutable Ref objects/versions, not every inline field; packet/compute limits still apply |
| Serving mask versus scalar gradients | Native projection excludes private rows; typed scalar reads stay unmasked; masking gold labels alone is insufficient |
| Numeric paired IDs/cache identity | Match fixture-relative meaning, not random run IDs/cross-run addresses; validate each run's own binding and state continuity |
| FP32 reference versus BF16 bytes | Frozen-content equality is within a profile; cross-dtype comparisons are numeric |
| Annotation disagreement versus Unresolved | Adjudication/quarantine is separate from a justified semantic result status |
| Pilot versus blind test | Pilot/worked examples are development material; fresh locked-test families are separate |
| Loss normalization versus raw semantic amplitude | Decoder/receiver use the same canonical semantic normalization as alignment, with no unpenalized amplitude bypass |
| A3 short numeric fixtures versus A4 joint length | Begin at 128/256/512; qualify A3 at 1024 before joint training; no full-context claim |
| Zero decision forcing versus free generation | Free-control training may teacher-force approved native tokens; promotion/free tests force neither text nor decisions |
| 25% replay versus reference preparation | The mixture applies to native/decision windows; reference preparation is a separate role-whitelisted semantic task, including permitted non-assessed Reason status/support |
| Scalar LR 1e-4 versus phase scheduling | 1e-4 is the nominal/peak rate; apply the declared phase multiplier with unchanged row isolation |
| capturable optimizer versus disabled CUDA graphs | Keep CUDA optimizer tensor state/arithmetic; capturable does not enable graph capture |
| Update/exposure counts versus dataset size | 7000 windows/56000 presentations include repetitions and reference records; training remains 6000 core+2000 replay |
| Full-prefix guidance versus actual divergent returns | Only guide the compatible reviewed prefix; never replace actual joint returns or restore incompatible gold actions |
| Resource/cancellation correction versus semantic status | Mechanical labels/termination have no invented semantic verdict or post-cancel answer |

This reconciles the documented contract; it does not prove that code obeys it
or that latent semantics are formally free of hidden answer channels.

## Recorded A1-A4 and B1

A1 is now resolved as a design by the [typed-value ABI](typed-value-abi.md):
eight-byte atoms, exact per-type costs, eight attributes, bounded parallel value
heads and an all-candidate/all-status return envelope. Its implementation and
correct interpretation remain unverified, not an outstanding typed-body/count
choice. The separately recorded B1 fixes the exact neural feature map.

A2 is now recorded by the [model configuration](model-configuration.md):
1024-wide backbone, 256-wide protocol, explicit pooler/head shapes, canonical
normalization, R/D/T ranks 8/16/8, zero-dropout aligned profile and independent
frozen reference/live codec storage. Instantiation/GPU behavior remains unverified.

A3 is now recorded by [numeric profile v1](numeric-profile-v1.md): continuous
near-zero error rules, ordered FP64 budget evaluation, checked integer accounting,
explicit common GPU kernels/determinism flags and boundary/gradient/export fixtures.
This closes the initial design choice, not calibration or passing GPU evidence.

A4 is now recorded by [training run profile v1](training-run-profile-v1.md):
1000/2000/1000/2000/1000 update blocks, eight scheduled examples/window, explicit
AdamW groups/schedule, exact-value loss subtypes, 400 existing correction cases,
1000-case development allocation and final-two-checkpoint promotion. These are
initial defaults, not an instantiated run, reviewed data or training results.

B1 is now recorded by [neural input encoding v1](neural-input-encoding-v1.md):
roles/classes, complete 128-bit header/descriptor/control layouts, presence flags,
ordered tags, local handles, exact decimal exponent storage, numeric helper
formulas, eight fixed counter denominators and Noul prototypes initialized from
false=3721/true=1802. Static metadata/tokenizer inspection is not GPU validation.

The [independent Astra review](review-astra-2026-11-05.md) found that the initial
closure statement was too strong: bit coverage and recorded defaults did not
settle every interface or promotion choice. Its historical report/source hashes
remain unchanged. The current post-review status is:

| Finding | Current status |
|---|---|
| F1: worst-case new Ref allocation holds | Recorded resolution in resource owner: dedicated sealing holds, two completion Refs, frozen K and independent position/handle bounds; no implementation, independent re-review or GPU validation |
| F2: per-role field/tag/owner/length/absence schema | Recorded resolution in B1 section 11; matching fixtures, independent re-review and GPU validation remain unperformed |
| F3: clean isolated-test tensor provenance | Recorded in clean-test construction v1: permitted-view lineage, fresh excerpt encoding/state, clean local refs, hidden-fact twins and clean/transfer/whole-flow distinction; implementation, independent re-review and GPU validation remain unperformed |
| F4: target-attribute grammar and return omission | Recorded: attributes follow request target slots; returns present zero target attributes and preserve target binding; implementation/fixtures unperformed |
| F5: optional consumption/control trace | Recorded in control-flow v1: zero extra ticks/hold, exact token masks/closure, bounded notice code12, terminal/cancel rules and H<=22; bounded G1 controls passed, while full implementation/re-review/GPU qualification remains incomplete |
| F6: exact module/optimizer ownership and dynamic eligible accumulation | Recorded in exact module/optimizer contract: safe block equations, shared trunks, complete heads/pointers, initialization/groups and per-component eligible-example gradient buffers; bounded G1 modules/accumulation passed, while complete implementation, independent re-review and broader GPU qualification remain incomplete |
| F7: role-specific answer-blind/reference-reason whitelist | Recorded in training-input rules v1: request targets exclude annotator-created current outcomes; justified non-assessed Reason status/support allowed; branch exposure, mismatch masks, teacher forcing and checks fixed as design; implementation, independent re-review and GPU validation remain unperformed |
| F8: evaluation slices, coverage/denominators and promotion details | Recorded in promotion matrix v1: fixed 1000/2000 allocations, 13-cell/status/family minima, intervention groups, all-attempt denominators, effort gates and phase/regression rules; approved artifacts, independent re-review, implementation and GPU validation remain absent |

Recorded resolutions are not a new independent review or an implemented codec.
Do not mark all G0 complete, invent approved data/fixtures or collapse remaining
contract decisions into empirical checks. Keep each finding's evidence separate.

Here T denotes the small training-only reconstruction adapter; it is not a
third deployed mode. Runtime still uses only W+R and W+D. Model-profile choices
must preserve the pinned Qev initialization rather than silently swapping models.

## Operational preparation and empirical gates

- Assign actual reviewers and review effort/subjective labels and the pilot.
  No reviewer or approved label is fabricated by documenting a procedure.
- Initial pilot needs no model generator; later model use requires a pinned
  selection, CUDA execution and provenance before generation.
- Run short GPU state/gradient/scalar-export checks only after implementation
  authorization. Installed kernel/state-mutation issues are feasibility risks,
  not missing semantic design choices.
- Instantiate actual model/data/split/config/kernel/fixture hashes and matched
  replay baselines before runs; a prose profile is not an executed manifest.
- Data, semantic, autonomous-invocation and length gates remain separate from
  numerics. No full-context, optimal-budget, answer-improvement or formal
  absence-of-hidden-channels claim is established.

## Coding-readiness conclusion

Latest execution evidence: [G1 gate repair/rerun](g1-gate-results-2026-11-05.md).
The original run and precision-only rerun remain historical failure evidence.
With the compensated numerical profile and explicit diagnostic gate-head fit,
G1-core PASSED: 72 numeric cases, 36 released continuations, 144 raw gate labels,
connected gradients, accounting, optimizer and frozen-base checks. No numeric
tolerance was relaxed. All 612 released actions followed raw argmax without a
resource mask. This does not establish semantic/general invocation readiness.
The subsequent [full codec/runtime work](full-codec-implementation.md) was
separately authorized without training or frontend expansion. Its
[qualification](codec-qualification-results-2026-11-05.md) is distinct: component
and FP32-reference successes do not close the failing expanded mixed-precision
gate. No service/frontend change or production checkpoint promotion occurred.

The [small GPU proof protocol](small-gpu-proof-v1.md) records the concrete
G1-core profile, originally requested as protocol-first. Its measured status is
now given above. The [post-proof expansion plan](post-proof-expansion-plan.md)
records conditional codec/data/training/frontend work after a measured pass.
Its gates remain unperformed; diagnostic bootstrap artifacts cannot silently
become capability checkpoints. G1-core does not close full A3/G1/semantic gates.

Full-profile validation of the initial profiles and F1-F8 responses remains
incomplete beyond the measured G1 subset.
The scoped G1 feasibility implementation and its versioned proof are now complete.
General interoperability
must implement/review the recorded admission/control contracts and produce
matching schema/neural vectors. Full training
additionally needs independent re-review, approved data/run artifacts and GPU
feasibility, with new findings reconciled if they arise. Documentation is not passing
evidence. Earlier documentation-only resolutions did not authorize execution;
the scoped proof and gate repair were separately authorized and measured.

Do not call all G0 complete or claim that no future implementation choices can
arise. Reconcile new defaults
in their owners and this audit; do not silently fill gaps with another prompt,
separate model, CPU workload or semantic runtime rules.
