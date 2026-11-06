# Reference recovery revision results — 2026-11-06

**Recovery is still failed; no fix or qualification is established.** The bounded
reference-only revision completed on the prescribed RTX 3090. Both final gates
remain **0/100 exact recoveries and 100/100 contrasts**. No live initialization or
capability training occurred. See the [portable receipt](evidence/reference-recovery-v1.json).
The earlier handoff's running-job notice is superseded by these final results.

## Implemented slice and acceptance

Implementation `908d8b5` resolves a concrete reuse gap: the old fitter used 500
reference IDs containing only 50 distinct meanings, ignoring permitted fields
already present in other selected, reviewed training records. The new immutable
catalogue contains 3,376 canonical views with original row hashes and field
addresses. It excludes current outcomes, continuations, effort and nontraining
splits; explicit non-assessed Reason retains its documented exception. The
sampler covers all views and restores exactly from its saved cursor.

The separately documented [development revision](reference-recovery-revision-v1.md)
also alternates ordinary teacher forcing with whole preceding-target vector
ablation. Targets remain labels; the ablated path never embeds them. Serial and
parallel objectives retain connected prefix/T derivatives. These are exposure
and conditioning changes, not a claim that the learned codec works.

Acceptance for implementation: whitelist/provenance/split checks, exact sampling
and resume, serial/parallel GPU loss and gradient agreement, future-target
causality, finite nonzero prefix derivatives and reference/live isolation.
**68 model-free checks passed in 2.658 s; four GPU objective checks passed in
13.270 s.** The ordinary serial/parallel CE was 7.073400974273682 /
7.073393821716309; input/T gradient RMS errors were 2.774758421781705e-5 /
2.182902051490784e-5, within the unchanged limits. Three focused catalogue tests
also passed in 0.044 s. The initial schema-fixture failure and corrected tests
remain in raw evidence; no model-fit attempt was discarded in this revision.

## Fixed fitting block and actual gate

The revision warm-started only the previous unqualified reference
`fb38a31359a4ff367780bb8729695894e663810efd124c016bc2d5ba3fdbf7de`, with fresh
reference AdamW moments. It completed exactly **1,000 updates / 8,000 eligible
exposures**; zero failed exposures, all 3,376 views used, one to 82 presentations
per view, and 4,000 exposures in each target-history mode. Frozen W feature
states were cached on CUDA; pooler, wire and T were recomputed with gradients.
No development or locked annotation entered fitting. Sparse roles remain sparse:
Key and Midpoint each have only five distinct training meanings.

Every scheduled gate, 100 through 1,000, measured the same counts:

| Checkpoints | Exact recovery | Contrasts | Qualification |
|---|---:|---:|---|
| 100–800, every 100 | 0/100 each | 100/100 each | Failed |
| 900 | 0/100 | 100/100 | Failed |
| 1,000 | 0/100 | 100/100 | Failed |

The unchanged final-two requirement is >=95/100 recovery and >=90% contrasts at
both 900 and 1,000. The production qualification loader rejects **both actual
final reports**, specifically `reference qualification did not pass`. No earlier
checkpoint was substituted. Contrast positives are reused identical reviewed
annotations; their separation already passed before fitting and cannot certify
recovery or semantic competence.

Mean loss in the first/last 100 windows was **4.7712684374183665 /
1.1359519423067104**. Median eight-exposure window time was
**3.4190447954897536 s**; complete runtime, including feature preparation and
periodic evaluations, was **4,916.259019798017 s**. Peak CUDA allocation was
**4,276,769,280 bytes**. Runner exit was **2 / BLOCKED**, not a GPU access failure.

All live parameters stayed byte-identical; all **320 frozen base tensors** and
physical embedding/head tying passed. Capability optimizer updates were **zero**.
The final reference fingerprint is
`d0d4aef44b4415beb19311350eb24533190e1a27406c5d6f59a15a5a57402dec`; independent
GPU diagnostic loads verified it before and after use. Full checkpoints 0000 and
every 100, optimizer/sampler/RNG state, frozen exports, all qualification reports
and the 49-file source snapshot are retained. Metadata audit verifies every
checkpoint/export file hash, unchanged measured sources/revision contract,
actual exposure accounting and all three original release hashes.

The final artifact lineage contains **2,000 reference updates**, including its
parent's 1,000. Total physical reference work is **2,016 updates** when the
parent run's discarded 16-update setup attempt is included. These are separate
counts; this revision itself has exactly 1,000 updates.

## Diagnosis and rejected explanations

Before revision, a read-only deduplicated probe recovered **31/50 training and
0/50 development** meanings. Several roles barely changed loss when their
same-role latent prefix was swapped. This supports partial template/conditional
overfitting and failed transfer; it does not establish a disconnected graph or
a single implementation root cause.

After revision, six bounded autoregressive reconstructions of existing RF
annotations demonstrate **incorrect meanings, not harmless JSON differences**.
Two distinct questions both decode a presentation-polish/functional-coverage
criterion about Sunfield seed swap stall reservation. Two distinct targets
(refund-share and access-readiness assessments) both decode that same unrelated
reservation. An unused-nights Key decodes completed-work-items/items; another
access-prerequisites Key decodes failed-checks/checks. Three tags-only controls
and full predicted tokens are retained locally. These diagnostic outputs are
not new scenario records, reviewed labels or fitting inputs. No gold token is
fed before its prediction.

Attention probes on 20 existing training annotations do not support a simple
chat-header-only explanation. Mean attention mass on causally content-blind
prefix states in the two pooler blocks changes from **0.18472/0.17589** in the
parent to **0.14948/0.06940** in the revision. Nonannotation template positions
also include context-dependent footer states and must not all be called blind.
Reference fitting and binding use the same full frozen-W annotation encoding;
no body-selection mismatch was found.

A final **read-only, zero-update** causal intervention tested adding the latent
prefix mean to every subsequent target-input vector, with the existing frozen
weights. Across six selected training and six development records, ordinary CE
means were **0.37031788875659305 / 2.6451161901156106**; the candidate worsened
them to **9.94296153386434 / 9.778547366460165**. Teacher-forced exact counts were
1/6 and 0/6 before, 0/6 and 0/6 after. These scores are diagnostic, not the
100-case autoregressive gate. This intervention supplies no evidence for a fix
with the current weights; it was **not installed** and no further fitting block
was launched. It does not prove every separately trained conditioning revision
would fail.

A separately [preregistered training-only isolation probe](reference-recovery-isolation-v1.md)
then optimized **only six disposable latent payloads**, 64 steps each, with W,
pooler, codec, T and all live weights fixed. Original pooled vectors recovered
**1/6**; by step 16 and at step 64 the fitted controls recovered **6/6 exactly**.
Wire RMS and slot counts stayed unchanged. Reference/live fingerprints stayed
unchanged, all model gradients remained None and model optimizer updates were
zero. The **384 latent-variable steps** are separately recorded; these fitted
vectors were discarded, never exported as an encoder or used on development.

This demonstrates that the fixed decoder/wire path can express all six selected
training annotations. It isolates a weakness in the produced representations
for those cases, rather than proving the decoder cannot represent them. It does
not demonstrate held-out decoder competence, a globally unique root cause, a
fixed pooling method or passing recovery. The raw positive-control outcomes
must never replace the actual 0/100 gate.

## Reproduction and remaining blocker

Raw evidence: `data/reference-recovery-20261106/`; run: `run/`; final export:
`run/reference-1000/`. The portable receipt binds reports, logs, diagnostic
helpers/results, audits, source snapshot and manifests. No model caches or
weight files are committed.

The actual completed commands were:

```sh
PYTHONPATH=tests .venv/bin/python -m unittest test_reference_data test_reference_qualification test_episode_producer test_training_inputs test_training_driver test_training_release test_training_schedule test_data_production test_qualification_protocol -q
CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 QOT_QUALIFY_SNAPSHOT=<pinned-base-snapshot> PYTHONPATH=tests .venv/bin/python -m unittest test_reference_fit_gpu -v
CUDA_VISIBLE_DEVICES=GPU-225509c3-d702-6c27-de03-76b5e99b33b1 .venv/bin/python -m scripts.fit_reference --snapshot <pinned-base-snapshot> --output data/reference-recovery-20261106/run --initial-reference data/reference-fit-20261106/run/reference-1000
.venv/bin/python -m data.reference-recovery-20261106.audit
```

There is no running fit to resume. The requested recovery fix remains unfinished.
The bounded isolation probe now establishes local decoder recoverability on
six training views. The next unfinished slice is a causal encoder/pooling
learning revision, tested first on existing training views before committing to
another expensive fitting block. Sparse role coverage is a measured limitation, not
proof that more data is necessary or that recovery is impossible. Do not extend
this failed block automatically, fit development/locked cases, weaken the gate,
replace latent inputs with text/lookup answers, or initialize live weights from
these artifacts. Actual learned invocation, semantic binding and full-trainer
memory remain separate unfinished work. No source-data generation, agents,
capability training, numeric matrix rerun or live-service change occurred.
