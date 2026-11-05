# Reference qualification and semantic binding

The qualification runner and conservative binding integration are implemented.
**The reference is not qualified:** no compatible trained artifact exists locally,
and the explicitly untrained diagnostic achieved **0/100 exact canonical
recoveries**, leaving all 100 unresolved. It separated **100/100 existing contrast
groups**; that alone cannot satisfy the joint gate. No fitting was performed.
See the [portable receipt](evidence/reference-binding-v1.json).

## Slice 1: read-only qualification and binding primitives

Acceptance: use the entire existing development RF allocation, preserve role and
Reason whitelists, measure recovery without an annotation prefix, apply the
recorded >=95% recovery / >=90% applicable-contrast thresholds, and grant no
qualification from a frozen flag, tensor hash or diagnostic fixture.

`training/reference_qualification.py` schedules all **100 RF records**, ten per
role, from the pinned development release. Exact autoregressive recovery of the
reviewed canonical annotation is a conservative sufficient recovery criterion.
The decoder receives latent slots/roles only. It stops at the first mismatch;
nonexact outputs remain unresolved rather than receiving invented semantic
approval. No teacher-forced likelihood or vector similarity substitutes for
reviewed recovery. Exact matching is stricter than accepting reviewed paraphrases.

The existing release has no standalone `reference_contrasts` asset. Instead,
the implementation reuses its **40 explicit reviewed intervention pairs** across
39 families, yielding **100 role groups**: question 40, dimension 10, conditions
20, zero 10 and one 20. Categories 1/2 specify negation and all/any changes;
7 specifies remaining-fraction scale changes; 8 specifies calibrated scale
changes. Category 8's zero-anchor paraphrase is excluded from meaning-changing
roles. Each positive is an identical canonical field in another selected,
reviewed record from the same family. Each negative is the existing paired
changed field. Missing sides, positives or declared changes reject. These are
derived relationships from the existing reviewed transformations, not new
annotations or claims of a separate pair-review call. No incidental batch
negatives, generated scenarios or locked cases are used.

`ReferenceEvidence` loads only complete, passing, hash-bound development evidence
for the actual frozen tensor fingerprint and current code/profile. Qualification
uses the already qualified v3 FP32 transformer / two-part compensated BF16
vocabulary arithmetic. `Reference.initialize_live` now requires that evidence.
Freezing random reference weights cannot initialize a qualified live codec.

The metadata inventory inspected **107 local safetensors headers** and found no
reference bundle with the contract's `reference.`, `pooler.` or `T.` keys. The
existing base, Qev, gate checkpoints and scalar heads do not contain that learned
reference pooler/T artifact. They were not repurposed or modified.

## Slice 2: formation losses and actual-content binding

Acceptance: offline annotations supply formation targets without entering a live
request, while outcome and continuation targets require the actual admitted
snapshot's content and compatible actual return. Diagnostic episodes must not
reach optimizer accumulation.

`training/binding.py` constructs per-role frozen targets from immutable selected
records. It supplies normalized alignment and reconstruction through frozen
T/reference codec, retaining derivatives into predicted slots. It supplies no
contrast term where reviewed relations for that formation example are absent.
Only permitted role fields enter the offline encoder; outcomes stay external,
and justified non-assessed Reason is separately encoded. Reference-stage helpers
now accept development RF records and preserve the Choice-only Reason operation.

Binding verifies exact canonical semantic contents, rubric metadata, complete
source byte coverage and the original prefill source identity, token coordinates
and state hash. Candidate order can change when a unique exact mapping binds
reviewed candidate identities to stable actual Refs. Edited targets, nearby
vectors, same IDs with different contents, extra unreviewed attributes and
unreviewed history/subset evidence cannot enable outcome labels. A measured
qualified reference and the current admitted runtime request are mandatory.
Exact reference equality is a conservative supported subset; arbitrary learned
deviations remain unresolved. A decoded gloss alone is insufficient evidence.

`EpisodeProducer` integrates formation losses, supported decision losses and
compatible serial guidance. It always executes its own live semantic emissions
and delivers the actual Qev result. A matching request is not enough to approve
the continuation: actual status, scalar/interval, selected stable Ref and, when
applicable, exact supported Reason must agree. A set-valued Choice annotation
cannot supervise a single-selection return. Later guidance/continuation targets
remain masked at the first unsupported binding or incompatible return.

Normal native/component producers require loaded reference qualification when a
reference is supplied. An explicit `diagnostic_reference=True` enables engineering
fixtures; `collect_session` rejects their optimizer accumulation. No diagnostic
override can manufacture `ReferenceEvidence`. Actual-reference content binding
is implemented conditionally; no learned request was certified in this work.

## Validation and provenance

**63/63 model-free tests passed in 2.647 s**:

```sh
PYTHONPATH=tests .venv/bin/python -m unittest test_reference_qualification test_episode_producer test_training_inputs test_training_driver test_training_release test_training_schedule test_data_production -q
```

**Nine distinct focused GPU checks passed**, exclusively on RTX 3090 UUID
`GPU-225509c3-d702-6c27-de03-76b5e99b33b1`: five binding/gradient/guard tests in
37.414 s; one new actual-return check and two affected producer regressions in
53.287 s; candidate permutation and its two affected comparison regressions in
18.618 s. The last group repeats two earlier checks following the mapping change.
The frozen auxiliary decoder produced finite latent-input gradient norm
`0.25784578919410706`, with no live R/D adapter gradients or reference parameter
gradients in that isolated fixture. It used prescribed reference-equal content
and an untrained frozen reference, not qualified semantics or learned invocation.

The complete RF diagnostic and later contrast-only diagnostic use the **same**
untrained fingerprint
`411438b8e8e58218b7d9c3d409b5c9e68a8be7ead2b5cdb460ae70aa73b7f9e2`.
The completed 100-case recovery run was not repeated after existing contrast
materialization was added. Both measured source versions are archived under
`data/reference-binding-20261105/{recovery,contrast}-source-snapshot/`, and their
manifests/reports/logs are hash-bound by the receipt. The earlier recovery report's
missing-contrast blocker describes its earlier implementation; the supplemental
100-group measurement resolves that coverage gap, not the failed recovery gate.
Expected diagnostic/preflight exits are **2 / BLOCKED**, not passing qualifications.
The actual diagnostic report was rejected by the qualification loader.

Retained setup failures: the first recovery launch incorrectly required a
three-part vocabulary profile and rejected before executing any recovery case;
the guard was corrected to the already qualified two-part profile. Initial
contrast lookup restricted positives to X records; preflight rejected before
model execution, then exact positives were reused from other selected reviewed
views in the same family. Two temporary schema tests initially omitted required
construction metadata; the corrected complete 63-test run passes. No result,
approval or discarded failed run was relabeled.

Training/development/locked manifest hashes remain unchanged. No scenario data,
model-service calls, agents, optimizer updates, reference fitting, deployment or
live service changes occurred. Original BF16 transformer arithmetic remains
failed; the completed 324-case and 108-case numeric matrices were not rerun.

## Next unfinished work

The learned reference qualification is **blocked by the missing trained frozen
reference artifact**, not a need to generate more data. Producing it requires
later fitting authorization. Once available, run the entire RF plus existing
contrast gate, load its evidence, then enable supported production bindings.
Arbitrary learned slots and promotion/semantic-use behavior still need their
documented evidence; this implementation does not approve them by similarity.
Full participating trainer-memory qualification and observed correction binding
also remain prerequisites before claiming the entire learned flow is ready.
No training or further generation is queued.
