# Budget and Score scalar rows

Status: recorded design choice, not implemented or GPU-validated. Row indices
below were checked from pinned local configuration/tokenizer JSON without
loading or executing a model. Training, tying, precision and export still need
GPU verification. This document authorizes no coding, model workloads or deployment.

Owner: private row indices, shared-table parametrization, row optimizer/gradient
ownership, vocabulary exclusion and precision/export equivalence. Follow the
[specification](architecture-and-training-plan.md),
[loss configuration](loss-configuration-and-training-schedule.md),
[codec](latent-packet-codec.md) and [handoff](implementation-handoff.md).

## 1. Pinned private indices

The pinned Qwen/Qwen3.5-0.8B revision
2fc06364715b967f1860aea9cf38778875588b17 has:

| Item | Value |
|---|---:|
| Shared embedding/output rows | 248320 |
| Valid tokenizer IDs | 0 through 248076 |
| Budget row | 248077 |
| Task Score row | 248078 |
| Hidden/row width | 1024 |
| Unused tail rows | 243 |

Both top-level and text config enable input/output tying. Put indices, tokenizer
and base hashes in the manifest and revalidate on load. Do not assume that
len(tokenizer) always finds a free row. Add no tokenizer entries or printable
names for the scalar IDs; reject incompatible artifacts rather than remapping
their semantics silently.

## 2. Frozen base and two row replacements

Let E be the frozen tied table. The effective table M equals E except:

    M[248077] = budget_row
    M[248078] = score_row

The two rows are separate trainable 1024-value parameters. One effective-table
provider is shared by input lookup and output projection. Do not train two
independent input/output copies. The effective table is a functional row
parametrization during training; never overwrite E inside an active graph.

These parameters represent rows of Qwen's own effective LM head, not a second
decision network. Pretrained E remains requires_grad=False and is excluded
from optimizer membership/state. Other pretrained parameters remain frozen.
Preserve input/gradient propagation through the frozen computations.

## 3. Row optimizer groups

Pilot defaults: AdamW, nominal/peak learning rate 0.0001, weight decay 0, FP32 master rows and
FP32 moments. This learning rate is a development starting point, not measured
optimal. Separate leaf parameters prevent an inactive row receiving updates
from another row's optimizer history. No eligible gradient means grad=None and
skip that parameter, rather than a zero-gradient momentum update.

The recorded [A4 run profile](training-run-profile-v1.md) fixes betas/epsilon,
phase multipliers, accumulation and clipping without changing these ownership
rules. Nominal LR is not a separate unscheduled optimizer for the scalar rows.

Do not unfreeze the whole tied matrix and rely only on zeroed gradients.
Decoupled decay may change parameters independently of current gradients.
Register/deduplicate the two row parameters once and step after complete
episode backward, as for the rest of joint training.

## 4. Direct row readout

    raw_budget = dot(final_prefill_hidden, M[248077])
    raw_score = dot(evaluation_hidden, M[248078])

Use Qwen's final hidden state without an additional MLP, bias, categorical
average or head-specific normalization. Read budget at the fixed initial W+D
port and task Score at its completed packet's EVALUATE position under D.
Selecting a row before multiplication avoids computing a full vocabulary
projection solely for one scalar, while using the same effective LM-head weight.

## 5. Gradient ownership and validity

Budget: reviewed effort regression only for its row; deliberately stop-gradient
on its control-feedback value. No derivative through rounding/allocation/stop.
Shared upstream adapters may change under other objectives, so predictions
still need effort supervision and verification.

Score: reviewed raw point/interval loss and continuation gradients through a
valid assessed task-Score return. Non-assessed cases do not receive invented
numeric targets. Follow reference-validity masks and packet-label applicability
from the loss configuration; log invalid results rather than substituting gold.

Delivered values are clamped to [0,1]; direct regression uses raw values so
out-of-range predictions retain corrective gradients. Continuous task-result
paths remain attached; the budget feedback boundary is intentional and separate.

## 6. Vocabulary exclusion

- Native text loss projects over allowed text IDs, excluding private/unused rows.
  Merely ensuring labels never equal these IDs is insufficient: softmax
  denominator membership would still introduce unwanted language gradients.
- Mask disallowed logits to negative infinity before text sampling/search.
  Exclude every unused tail ID, not just the two scalar IDs.
- Reject private scalar IDs as input tokens; leave them unassigned in tokenizer.
- Access the relevant unmasked scalar row directly through its typed port.

Apply this to thinking text, final answers, reconstruction exercises and all
supported generation paths, retaining valid existing termination tokens.
A language-only fixture with no task-result return must have no gradient path
to either row. Native continuation after a valid Score call may train Score
through that return, as intended.

## 7. Precision and export

Keep FP32 masters but expose BF16-rounded row values during forward, matching
resident table storage. Scalar ports use FP32 accumulation from those effective
values and the selected hidden state. Use the same arithmetic when serving;
do not compare a trained FP32 row with an untested BF16-only export path.

Save FP32 masters for resumption, BF16 serving rows, indices, precision policy,
hashes and optimizer state for resumable checkpoints. At inference load, install
only the two serving rows into the resident tied table before a job begins.
Input embeddings and LM head must still share storage. Never patch rows during
live cache execution or edit the downloaded base files.

## 8. Bootstrap and compatibility

Initialize rows independently, then bootstrap with their own reviewed scalar
examples before joint reliance. Zero initialization is a reproducible starting
option, not a meaningful assessment or uncertainty marker. The recorded training
budget override prevents an untrained budget row blocking early call exercises.

Do not copy budget into Score as an undocumented fallback. Missing/untrained/
unsupported artifacts must fail validation, not serve apparently assessed values.

## 9. Required GPU evidence

| Check | Required result |
|---|---|
| Table preservation | Every row except the two reserved replacements stays unchanged |
| Budget-only update | Budget row changes; Score/base table do not |
| Score-only update | Score row changes; budget/base table do not |
| Language-only update | Neither scalar row changes |
| Tying | Shared effective values during training; shared storage after installation |
| Vocabulary exclusion | No scalar-ID text sampling/targets/input tokens |
| Export round trip | Scalar behavior agrees under the declared arithmetic |
| Budget mapping | Compare derived integer budgets, including rounding boundaries |

For fixed hidden states, scalar-row changes must leave allowed vocabulary
logits unchanged. R/D updates can change hidden states and hence answers;
frozen vocabulary rows do not guarantee unchanged language behavior. The
[numeric verification plan](numeric-verification.md) records starting tolerances,
discrete agreement and gradient/update checks; GPU evidence remains unverified.

References: [PyTorch parametrizations](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.parametrize.register_parametrization.html)
and [AdamW](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html).
These document supporting mechanisms, not automatic correct tying or proof
that this proposed implementation already passes.
