# Next-agent handoff — 2026-11-05

Read this first for current execution status. The [implementation specification](architecture-and-training-plan.md)
and topic owners retain their design contracts; the older
[implementation handoff](implementation-handoff.md) retains the design/migration
checklist. Historical statements there that implementation or GPU evidence is
absent are superseded by the measured results linked here.

## Current checkpoint

Repository: [veighnsche/QwenOneTwo](https://github.com/veighnsche/QwenOneTwo).
Workspace: `/home/vince/QwenOneTwo`. The completed implementation and numeric
qualification are committed and pushed to `main` at
[`e6cbd0b`](https://github.com/veighnsche/QwenOneTwo/commit/e6cbd0b).
This handoff is a subsequent documentation change.

**The full codec/runtime is implemented and its prescribed numeric matrix passes
through 512 positions. The complete learned decision capability is not ready.**
No capability training, reviewed-data expansion, frontend integration or
deployment occurred in the full-codec/numeric-fix rounds. The last user instruction
was to write a handoff. This document does not start those later stages; follow
the scope of the next user task.

## User requirements to preserve

- One user prompt, one prompt prefill, one continuous hybrid-cache lineage and
  a normal answer from that same inference job. No separate decision request,
  second prefill, decision-first confirmation or second generation button.
- One matched Qwen3.5-0.8B backbone and Qev LoRA. D is on for the initial budget
  readout, R for native work, and D temporarily for internally requested Noul,
  Choice or Score. Runtime executes learned control; it must not supply domain
  rules or keyword routing in place of learning.
- Initial effort is one continuous score, not budget categories. Clamp to
  `[0,1]`, then use the exact ordered CUDA FP64 S-curve from
  [A3](numeric-profile-v1.md): `floor((3*s*s - 2*s*s*s) * 131072 + 0.5)`.
  Maximum thinking budget is half the 262144-position context. Task Score is a
  separate scalar row and meaning.
- Requests/results are textless typed latent packets, with immutable question,
  target, alternatives, rubric and evidence bindings. Qwen must learn both to
  construct them and to interpret the returned result. Stock Qev compatibility
  does not establish either skill.
- All model inference, training, extraction, fitting and real-model tests are
  **GPU only**, explicitly on RTX 3090 UUID
  `GPU-225509c3-d702-6c27-de03-76b5e99b33b1`. Never use CPU fallback or the RTX
  3060. If access fails, diagnose device selection and execution-environment
  access before reporting a blocker. Ordinary editing/static tests can use CPU.
- Keep The Operating Table's service/data/artifacts unchanged. Its pinned model
  snapshot is used read-only. Do not replace the live capability checkpoint or
  assume the deployed legacy UI serves the new codec.
- Follow [AGENTS.md](../AGENTS.md): commit authorized completed work directly to
  `main` and push; do not create a branch/PR or ask again for that authorization.
  Deployment still requires explicit user authorization.

## What is done, with evidence

| Area | Implemented/measured status | Evidence |
|---|---|---|
| G1-core | Passed bounded cache/gradient/accounting and released-continuation gates; gate repair changed only an isolated diagnostic child | [G1 results](g1-gate-results-2026-11-05.md) |
| Full codec | Six typed bodies, all 13 operation/status cells, semantic roles, writer/head interfaces, immutable/history rendering, resource reservations, lifecycle and one-cache session | [Implementation/module map](full-codec-implementation.md), `qwen_one_two/codec/` |
| Components | 6 model-free exact tests and 18 GPU regressions pass; the GPU count is 15 codec tests plus 3 precision tests | [v3 results](codec-feedback-results-2026-11-05.md) |
| Expanded numerics | 324/324 traces pass: 128/256/512 × Noul/Choice/Score × functional/checkpoint/serving/FP32 × seeds 17/29/43 × 3 repetitions | [Machine-readable summary](evidence/codec-feedback-v3-summary.json) |
| Agreement | Worst state RMS 0; logit RMS 5.825623065e-06; gradient RMS 8.316676447e-06; returned scalar/startup raw differences 0 | [Measured report](codec-feedback-results-2026-11-05.md) |
| Cache/accounting | One prefill/lineage, exact P+B+A+H, one consumption per dispatch, zero remaining position holds; calls across both phases | Same report and raw bundle |
| Parameter/memory integrity | All live parameters unchanged; 320 frozen tensors and tied storage checked per group; peak process allocation 9.2546 GiB, including resident state-buffer envelopes | Same report; no optimizer constructed or stepped |

The 324 cases use fixed inputs/weights: seeds/repeats check reproducibility, not
semantic diversity. Fixtures prescribe operations/statuses, use a thinking-budget
override of 2048, and prescribe native continuation. Connected gradients are
engineering evidence, not trained behavior. Source states are contextual
production-tensor transfers, not clean withheld-fact semantic fixtures. New
codec heads remain initialized/untrained; the parent is a diagnostic checkpoint,
not a production capability checkpoint.

## Precision fix: do not accidentally undo or mislabel it

Original two-part BF16 transformer arithmetic failed at 512 positions: worst
state RMS 0.234006513 and gradient errors around 0.71. Projection rounding was
amplified by serial Reason feedback. Extra BF16 split parts and tested narrower
FP32 islands did not fix it.

The qualified [v3 profile](codec-feedback-precision-v3.md) is explicitly:

- Original frozen matrix **storage in BF16**.
- Transformer base and R/D LoRA **computation in FP32**, with FP32 causal state.
- Native vocabulary projection still uses compensated BF16 products.
- Genuine FP32 reference, original weights, fixtures and tolerances unchanged.
- Frozen projection backward retains original BF16 weight storage and promotes
  transiently; no per-call saved FP32 weight copy or second backbone.

`codec/backbone.py` selects v3 by default. `codec/precision.py` implements its
frozen projection. Legacy compensated transformer arithmetic remains selectable
for diagnosis, and still fails. Do not call this a BF16-transformer arithmetic
pass. Preserve the old [failure report](codec-qualification-results-2026-11-05.md),
G1 sources/checkpoints and historical Astra review. Requalify material changes
to the profile, kernels, weights or execution path at unchanged tolerances.

## Remaining work, in dependency order

The [post-proof expansion plan](post-proof-expansion-plan.md) owns the detailed
order and exits. These are outstanding tasks, not new authorization to execute.

1. **Review the implemented codec/v3 against the contracts.** The historical
   Astra document reviewed the design, not this later implementation. Complete
   independent implementation/interoperability review, reconcile findings in
   the owners, and pin the actual protocol/profile/artifact manifests. Retain
   existing component vectors; extend matching train/serve/export evidence when
   those new paths exist. Do not restart the already completed BF16 diagnosis.
2. **Implement the reviewed-data pipeline and clean fixtures.** Build canonical
   annotations, role-specific field builders, provenance/coordinate validation,
   [F7 answer-blind input whitelists](training-input-rules-v1.md) and
   [F3 clean tensor construction](clean-test-construction-v1.md). Questions,
   candidates and rubrics cannot carry the current outcome; permitted
   outcome-bearing Reason supervision follows its separate rules. Re-encode
   genuinely permitted views for omission/twin tests; contextual source tensors
   from the current engineering proof cannot stand in for those clean tests.
3. **Author/review the pilot before bulk data.** Follow [data operations](data-operations.md).
   The recorded starting pilot is
   120 fictional scenarios, 20 per each of six domains, with actual independent
   semantic/outcome reviews and adjudication. Do not invent reviewers, approvals,
   gold scores or elapsed model runs. Natural prompts must not name/force an
   operation. Keep original discussion prompts/walkthroughs out of training,
   tuning and fresh blind claims. Version family-based splits before variants.
   Later release defaults are 6000 core + 2000 replay, 1000 development and
   2000 locked examples; they have not been produced/reviewed by this proof.
4. **Implement reference preparation and the real trainer.** The reference
   pooler exists and is shape-tested; reference/T preparation, reviewed semantic
   targets and capability-training phases do not. Implement the A4 losses,
   subtype masks, forcing/correction schedules, unique parameter ownership,
   per-component eligible-example accumulation, inactive `grad=None`, clipping,
   optimizer state and versioned export/resume. Use the
   [A4 run profile](training-run-profile-v1.md), [loss owner](loss-configuration-and-training-schedule.md)
   and [module/optimizer contract](module-and-optimizer-contract-v1.md).
   `codec/ownership.py` assigns
   owners; it is not a complete trainer. The memory envelopes are not an A4
   optimizer/eligibility qualification. Keep W frozen and scalar rows isolated.
5. **Qualify 1024 positions before joint training.** Extend the applicable
   numeric, state, gradient, accounting and memory checks under a versioned
   protocol on RTX 3090. This may be a separately authorized numeric-only task;
   it does not require starting capability training. The current matrix runner
   hardcodes the 324-case 128/256/512 matrix; it does not certify 1024. No
   full-context or maximum-budget claim follows from either short-length gate.
6. **Train and promote in A4 order once its prerequisites are met.** Recorded
   windows are reference 1000, components 2000, guided joint 1000, transition
   2000, free-control joint 1000. Train native R to form latent requests and use
   actual returned values, alongside the invocation/writer/evaluator skills;
   bridge-only fitting is insufficient. Keep continuous effort distinct from
   task Score. Freeze/version reference artifacts after promotion. Require the
   specified final-two-checkpoint development exits and [F8](promotion-matrix-v1.md) coverage, denominators,
   semantic interventions, effort and replay gates. Do not force calls to pad
   coverage, substitute gold returns in joint runs or backpropagate through
   argmax. These training phases have not run.
7. **Integrate trained inference into backend/frontend.** The new codec is
   isolated; legacy `server.py`, `thinking.py` and the browser are not migrated.
   Wire one Send to one real generation job, expose authentic progress/events
   and the normal incremental answer, and implement targeted Stop, reconnect
   and terminal/error handling without resubmitting the prompt. Existing polling
   may be retained. Telemetry must not invoke a decision endpoint. Mock UI traces
   can verify UI mechanics only; integration exit needs real, learned, unforced
   same-cache behavior. Do not use diagnostic weights as the live checkpoint.
8. **Locked evaluation and release.** Run the fixed locked protocol only after
   development promotion. Prepare compatible checkpoint/profile manifests,
   provenance, numerical and integration reports, and reviewable deployment
   changes. Deployment and longer-context expansion remain separate tasks.

For the next bounded implementation task, prefer a scoped implementation review
or a defined 1024-position proof protocol, then the task the user authorizes.
Do not silently start a 7000-window training run or redeploy after reading this.

## Code entry points and traps

| Location | Use |
|---|---|
| `codec/schema.py`, `records.py`, `render.py` | Exact ABI, immutable records and shared packet rendering |
| `codec/neural.py`, `writer.py` | Encoders/readouts, semantic emission, references and writer interfaces; interfaces do not imply trained semantics |
| `codec/runtime.py`, `session.py` | Admission/holds/lifecycle, one-prefill switched inference, native continuation |
| `codec/backbone.py`, `precision.py` | Qualified v3 dense arithmetic and recurrent rematerialization |
| `codec/qualify.py`, `qualify_matrix.py`, `diagnose_precision.py` | Prescribed qualification, strict aggregate coverage and controlled ablations |
| Legacy `train.py`, `train_feedback.py`, `curriculum.py` | Prototype material; do not reuse as proof of complete connected training or reviewed data |
| Legacy `server.py`, `thinking.py`, `static/app.js` | Migration targets; do not assume they expose the new codec or trained flow |

## Artifacts and reproducibility

Weights/raw artifacts are intentionally ignored by git. A fresh clone needs
the pinned local artifacts; missing files are a blocker to model execution,
not a reason to download a different model, refit the diagnostic child or use CPU.

| Artifact | Location/identity |
|---|---|
| Base | Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17` |
| Local read-only base snapshot | `/home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17` |
| Base weights SHA256 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| Qev | `twainsk/qev-0.8b`, revision `1d2a6d7f891daab7483a49c3abe507c0c9d7ea62`; local `data/qev/` |
| Frozen diagnostic parent | `data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors` |
| Diagnostic SHA256 | `1dd2e8527e14caa9002d672b4187253cdc174fb3c204ab56e80f257f23943640` |
| v3 raw bundle | `data/codec-qualification/feedback-v3-20261105/` — aggregate/child reports, source/test snapshots, logs, inputs and manifest |
| Aggregate report SHA256 | `157b1dcb196da3fb7f97295b59e29ca9f19a644b7623cf7795336e4c359307e4` |
| Portable evidence | [Measured results](codec-feedback-results-2026-11-05.md) and [summary with source/input/child hashes](evidence/codec-feedback-v3-summary.json) |

To reproduce after an affected change, use a **new** output directory:

```bash
cd /home/vince/QwenOneTwo
export CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export QOT_QUALIFY_SNAPSHOT=/home/vince/the-operating-table/data/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
.venv/bin/python -m unittest discover -s tests -p test_codec_exact.py -v
.venv/bin/python -m unittest discover -s tests -p 'test_codec*gpu.py' -v
.venv/bin/python -m qwen_one_two.codec.qualify_matrix \
  --snapshot "$QOT_QUALIFY_SNAPSHOT" \
  --checkpoint data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors \
  --output data/codec-qualification/REPLACE-WITH-NEW-RUN --workers 2
```

The measured matrix took about 55 minutes with two workers on the selected
3090. Do not rerun it for documentation-only edits. In restricted execution
environments, driver access previously required running the same CUDA command
outside the sandbox; model execution on CPU was never the fallback. Inspect
current GPU allocation before a new run; do not kill unrelated services.
