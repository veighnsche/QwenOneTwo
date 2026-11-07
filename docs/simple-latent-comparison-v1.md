# Slice 3: one simpler latent-input comparison

Authorized: one read-only comparison on RTX 3090 UUID
GPU-225509c3-d702-6c27-de03-76b5e99b33b1, existing data, W/reference frozen,
no agents or training. Freeze D/R, codec and all readouts too. No new annotations,
threshold fitting, regression/classifier probes, promotion or service changes.

## Single candidate and hypothesis

Profile **ordered-full-width-means-v1** replaces learned reference pooling with
ordered, parameter-free FP32 means of the **same frozen-W annotation states**.
For a role with n token states and s existing slots, bin i is
`[floor(i*n/s), floor((i+1)*n/s))`; average each bin and apply the existing
FP32 RMS normalization/floor. Every original state contributes to exactly one
bin. The original source includes template tokens; no cropping, new tokenizer
or hidden-state extraction is introduced. Empty bins are rejected.

This bypasses the learned 1024→256 memory projection, attention blocks/queries
and 256→1024 lift. It retains full hidden width during pooling, but **still
compresses the sequence to the same 8/4/2 role-slot counts**. This is the simpler
fixed-capacity comparison selected for slice 3, not an uncompressed sequence
experiment or proof about whether those slot counts are sufficient.

The question is whether this simpler basis works better with the **existing
frozen receiver** on held-out paired tasks. This is a zero-update compatibility
comparison. The receiver previously learned with the control basis; failure of
an unfamiliar replacement cannot establish that its information is inferior,
that the original pooler is correct or that either architecture is unlearnable.
Do not broaden into a variant search or train either path in this slice.

## Controlled construction and cache reuse

Pin parent checkpoint-0188 receiver `94218aae…` and reference `5c6015d…`.
Reuse all **160** pre-pool/pooled annotation captures from the completed slice 2,
bound by file hash, executed source hashes, checkpoint/reference identity,
reviewed input content and role. All 180 existing permitted views must match
their saved records. Cache loading, new means and all model work run on CUDA.

Keep operation, supplied semantics, exact source bytes, evidence states,
candidate identities/order, rubric metadata/direction, codec tags, scalar rows,
budget mechanism, status/Reason decoding and actual return delivery unchanged.
Only semantic role payloads change. Validate nonsemantic encoded request rows
bit-for-bit against the saved wire captures. Canonical record normalization
and all production snapshot, binding, admission and accounting checks remain.

The candidate basis also reaches the existing writer's prepared-draft summary
used for the R-mode component-finish tick. This is a consequence of replacing
the semantic records through the original producer. Do not claim a fixed hidden
prefix at the D request boundary. The prompt, startup/budget, causal lineage,
preparation event count and request lengths remain the same; the actual resulting
state is evaluated. Nothing is inserted into or replayed over an active cache.

Candidate audit entries explicitly name the new basis, originating captured
pre-pool tensor, token bins and **reference_target_compatible=false**. They are
not learned-reference outputs, approved writer targets or semantic certificates.
The diagnostic encoder rejects gradient-enabled use. No training producer or
production default is changed. Request content remains answer-blind. Existing
post-decision Reason semantics/exceptions are unchanged.

## Cases and measurements

- Reuse all **180 X /90 paired** control receipts from slice 2, with current
  source/identity and view checks. E and locked data remain outside this run.
- Execute **three control anchors**, the first canonical X case per operation,
  through the same cached-input adapter. Require exact outputs, full canonical
  ledger, wire vectors and final h against the saved control.
- Execute the **180 candidate cases once**, always decoding/consuming actual
  outputs. Model-result decoding failures remain in denominators and are retained;
  infrastructure/numeric/input-binding violations stop the run and preserve partial
  evidence. There is no elapsed-time cap, automatic retry or alternate candidate.
- Report per-operation delivered correctness, statuses, independent assessed
  heads, Noul Brier, Score interval distance, exact reviewed interval agreement,
  full intervention pairs and paired wins/losses. Keep family/pair dependence visible.
- Verify the 20 identical-permitted-input pairs remain invariant. Report h/wire
  displacement descriptively; distance is not a semantic-fidelity score.
- Count actual new Sessions/positions separately from reused control receipts
  and offline source encoding. No new reference encoding or pooler evaluation
  is necessary; no text invocation or gold result is permitted.

## Acceptance and limits

Acceptance is completion of this one comparison with unchanged frozen identities,
verified cache/source provenance, all cases accounted for, valid request framing,
one prefill/lineage, exact accounting and honest failures. Candidate means must
match an independent GPU FP64 calculation within the existing FP32-reference
relative RMS tolerance 1e−5 (absolute 1e−11 at reference RMS<=1e−6). Validate
all 160 role payloads, finite values and unchanged shapes. No thresholds widen.

Reuse completed transformer/gradient numeric matrices: their implementations
and v3 computation do not change. No gradient or training qualification is
claimed for this intentionally read-only input adapter. Keep all 320 W tensors,
physical tying, live/reference fingerprints, cached inputs and grad=None checks.

Even if existing behavioral thresholds pass, this diagnostic cannot promote
weights, grant ReferenceEvidence or qualify native invocation/answer use.
The next learning/architecture decision must distinguish observed compatibility
from untested adaptation. Leave The Operating Table and live services unchanged.

Entry point (from the repository, with the primary GPU UUID selected):
`PYTHONPATH=. .venv/bin/python -m scripts.compare_latent_inputs --snapshot <pinned>
--output data/<fresh-run> --execute-gpu`. Without `--execute-gpu`, only prepare
the plan and bind metadata; no model library is imported.
