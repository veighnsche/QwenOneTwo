# Architecture/data baseline — slices 1–2

Slice 1 is a read-only implementation audit. The current product requirements
remain Noul/Choice/Score, learned invocation and native result use, textless
transport, one prefill/cache, continuous budget, exact values/bindings and v3
numerics. W stays frozen. Pooling, alignment, readouts and the curriculum are
design hypotheses, not capabilities established by a mechanical qualification.

## Actual learning and execution paths

| Boundary | Current implementation / gradient ownership | Measured status |
|---|---|---|
| Permitted input | `component_view` / `CleanView` whitelist byte-bound request semantics and source evidence; outcomes excluded | Provenance/answer-blindness checks exist; content review is not a learnability certificate |
| Reference input | `Reference.annotation_states`: frozen W, zero native adapter, no-grad, fresh offline state | Independent of live D/R; not a second prefill inside a Session |
| Reference pooling | Frozen `TargetPooler`, separate target/shared queries and blocks | Current reference is versioned but semantically unqualified; earlier readback failures concern an earlier checkpoint |
| Evidence extraction | `CleanEncoder._states`: current W+D, fresh state, **no-grad** | D-dependent inputs change when D updates, but source extraction receives no receiver gradients; intentional stop, causal effect unproven |
| Wire representation | `CleanPackets` → `Codec.encode`: semantic vectors, source projections, role/tags, metadata/atoms | Live receiver training updates selected encoders; exact metadata alone does not establish useful semantic features |
| Receiver | Same Session appends latent request in D mode | D and live receiver heads receive component gradients; source extraction and frozen reference do not |
| Noul / Choice | Shared inherited q/k; Noul prototype difference gives a linear, bias-free logit over q(h); Choice compares candidate boundary states | Head-only prototypes cannot add nonlinear reasoning to frozen h/q. Choice retention is measured; full status coverage is not |
| Score | BF16-effective learned LM-head row, FP32 dot, clamp | Linear readout; small-panel approximation works, development transfer fails; precision residual cause unresolved |
| Status / Reason | Separate status head and emitted Reason slots; approved Reason targets only after forward | Valid structure is not correct support meaning. Receiver objectives stop before native feedback use |
| Native flow | W+R learned gate/writer, runtime seal, W+D evaluation, runtime binding, W+R consumption | Machinery exists; current component exercises force calls/prepared packets and do not certify this complete learned flow |
| Optimization | Per-component VJPs, eligible-example means, clip 1, AdamW, isolated owners and checkpoint state | Tested implementation; dispatch expense and learning effectiveness are separate questions |

During the new diagnostic **all** model/codec/reference parameters are frozen,
including D/R and heads. No backward, optimizer, fitted classifier, fitted
threshold or regression probe is permitted. Descriptive CUDA linear algebra is
allowed; it does not modify the model or become a deployed scorer.

## Current data evidence

`scripts/audit_architecture.py` loads only the pinned training/development releases.
Raw output: `data/information-loss-20261107/baseline.json`; portable copy:
[`evidence/architecture-baseline-v1.json`](evidence/architecture-baseline-v1.json).

- Training: **8,000 records /288 families**; receiver: **1,500 /288**. Development:
  **1,000 /120**; E **130 /91**, applicable X **180 /60** (90 correlated pairs).
- Receiver train Noul assessed: **168 false /88 true**. E: **5 /5**; X: **20 /50**.
  This is a distribution difference, not proof that reweighting will repair Noul.
- Receiver train Score assessed: **254**, all **higher/increasing**, all exact
  point intervals; **85 zero +85 one**, **23 half**, **61 other**. Development X:
  **30 higher +10 lower**, all 40 exact points; E: nine point intervals and one
  nonzero interval. Reversed direction has no assessed receiver-training exposure.
- Reference records: **500 training /100 development**, but only **five unique
  annotations per role** in each split (ten roles). Repetitions are not independent
  meanings or broad semantic coverage.
  This counts the release's original reference category, **not** the actual
  current reference's expanded 3,376-view fitting catalogue. The
  [subsequent audit](training-x-semantic-audit-2026-11-07.md) measures that actual
  exposure and its overlap with receiver inputs; use it for current interpretation.
- Receiver train/development family and permitted-input overlap: **zero**. No
  contradictory reviewed targets among identical permitted inputs in each panel.
  E has 129 unique views; X has 145. Expected invariance/repeated views remain
  in their original denominators; these are not automatically leakage.
- Locked data remains unopened. No labels, annotations, reviews or releases changed.

Ten model-free metadata/report/ownership tests passed in **14.330 s**. The full
metadata audit executed successfully. Its first development attempt exposed a
tuple-unpacking bug in the new audit, fixed before this report; no model ran.

## Interpretation and next work

The audit identifies narrow semantic-reference exposure, Score direction/range
coverage gaps and Noul class shift. It does not identify the causal source of
failed decisions. Keep original reviewed values and tolerances; do not relabel
failed historical runs or infer architecture impossibility.

Slice 2 follows [`information-loss-v1.md`](information-loss-v1.md): observe existing
representations and saved readouts without fitting or changing their inputs.
Choose a subsequent architecture comparison from those observations. A simpler
latent-input profile is slice 3 and is **not implemented or run by slices 1–2**.
