# Reviewed release transport v1

Owner: mechanical serialization and content binding of the reviews, sources,
splits and releases required by [data operations](data-operations.md).
Implemented in `qwen_one_two/training/release.py`. This format does not assign
reviewers, create annotations or approve a dataset. No real reviewed pilot or
training release has been supplied. The tests contain temporary schema markers,
not semantic data or evidence that human review occurred.

## Load boundary

`load_release(directory, expected_hash=...)` reads only local files and returns an
immutable `ReviewedRelease` snapshot. It rejects unapproved states and ambiguous
JSON (duplicate keys, nonfinite constants/numbers). Relative paths must stay
within the release directory, including through symlinks. Every declared record
and asset must match its SHA256. Duplicate IDs/paths reject. Missing ancestors
reject; they must be included as retained audit records.

The release identity is SHA256 of the exact `manifest.json` bytes. Canonical
subdocument hashes use the existing UTF8, sorted-key, compact JSON encoder
`training.inputs.canonical`. Whitespace changes to a complete manifest change
its release identity. No files are edited or approved by loading them.

## Manifest

| Field | Required value |
|---|---|
| `format` | `qot-reviewed-release-v1` |
| `id`, `version` | Nonempty named release, positive integer version |
| `state` | `approved`; drafts reject |
| `purpose` | `pilot`, `training`, `development` or `locked` |
| `records` | Descriptors `{id,path,sha256,selected}`; `selected` is Boolean |
| `assets` | Descriptors `{path,sha256}` for sources, reviews, approval and retained audit evidence |
| `family_splits` | Complete family-to-split mapping, assigned before descendants |
| `split_sha256` | Canonical hash of `family_splits` |
| `exclusions` | `{records:[IDs],families:[IDs],batches:[IDs]}` |
| `approval` | `{artifact_path,artifact_hash}` of release approval evidence |
| `pilot_release_sha256` | For training, exact manifest identity of the prerequisite reviewed pilot |
| `batches` | When applicable, batch ID to `{members:[record IDs],second_review:[record IDs]}` |

`selected` distinguishes usable records from retained ancestors/rejected audit
material. Selected records must be approved for this exact named release/version.
All ancestry is acyclic, remains in the same family/split and agrees with the
split manifest. Selected descendants of excluded, unresolved or quarantined
material reject. A quarantined record with a batch blocks every member of that
batch and their descendants. Rejected material/evidence can remain unselected;
the loader does not delete it to improve review counts.

The complete previous release history and detection of unreported/near-duplicate
relationships remain review responsibilities. Exact duplicate human-input objects
across selected and retained records in different splits reject. This is not a
semantic near-duplicate detector or a check against every release ever authored.

## Record and source minimums

The mandatory record fields in `validate_record_manifest` remain in force:
identity/version/family/split/parents, origin, construction, sources, labels,
reviews and release. Model assistance, replay reuse rights and rollout metadata
retain their existing conditional requirements. Selected records additionally use:

- `author`: actual author identity, distinct from each independent reviewer.
- `release`: exactly `{state:"approved",id:<manifest ID>,version:<manifest version>}`.
- `construction`: includes `scenario_spec`, `template_version`, `seed`,
  `transformation`; values preserve actual provenance.
- `labels.schema_version`: nonempty annotation schema version.
- `human_input.prompt`: nonempty original human task; annotations remain separate.
- `sources`: descriptors `{id,version,path,sha256}` pointing at declared UTF8 assets.
  Bytes and source hashes must match; an identity/version cannot name different
  content. Per-record source IDs are unique.
- For training: `training_category` from reference/writer/receiver/reader/
  interleaved/direct/language_replay/decision_replay. Writer/receiver catalogue
  rows derive operation, reviewed `labels.status` and domain from this record.

Source-coordinate/provenance graphs and role-specific answer blindness are still
checked by the F7/F3 input and clean builders when constructing actual tensors.
This loader verifies source assets and review bindings; it does not interpret
arbitrary annotation prose or turn a release record into a model episode.

## Independent review binding

Each entry in `record.reviews` has the existing reviewer/kind/independence/decision/
record-version fields plus `artifact_path` and `artifact_hash`. The asset bytes
must match that hash and decode to:

| Review artifact field | Required value |
|---|---|
| `format` | `qot-review-v1` |
| `record` | `{id,version,annotation_sha256}` for the exact current annotation |
| `reviewer`, `kind`, `semantics_blind`, `outcome_independent`, `decision` | Match the embedded review entry; approved independent human review required |
| `judgments.semantics`, `judgments.outcome` | Nonempty independent judgments |

`annotation_sha256` hashes the canonical record excluding only `reviews` and
`release`. It includes human input, semantics, labels, source descriptors,
construction, lineage, exposure flags and any other annotation fields. Changing
an annotation therefore invalidates the review even when the file descriptor and
release approval are subsequently updated. Review metadata cannot bind itself.

Pilot, locked, subjective labels and new rubrics require two distinct independent
reviewers. Other records need at least one independent review and a stratified
second-review batch audit. An active batch has at most 200 members; audit size
is `min(size,max(20,ceil(.10*size)))`. Audited members need two independent reviews.
Every represented operation/status/domain/transformation value must occur in
the audit. Selection is still a review procedure: these structural checks do not
prove unbiased sampling. Single-review selected records without a valid batch
reject. Unresolved or rejected current judgments cannot be silently approved.
Adjudication/revision must produce current approved evidence; previous decisions
remain retained audit assets, without being presented as current approvals.

## Release approval and pilot gate

The release approval asset is `qot-release-approval-v1` with `approver`,
`kind:"human"`, `decision:"approved"` and `release_sha256`. This target is the
canonical manifest excluding its `approval` field and the approval asset's own
descriptor. That avoids circular hashing while binding every record, peer asset,
split, exclusion, batch and prerequisite. The complete manifest identity binds
the approval descriptor and its actual bytes as well.

A `pilot` selects exactly 120 distinct root families in development, each marked
`pilot:true` and `fictional:true`, with two independent reviews. Exactly 20 belong
to each domain: refund_eligibility, scheduling, demo_preparation, access_rules,
resource_allocation, issue_prioritization. Repeated identical human inputs reject;
semantic independence still requires actual review. Training selects only train;
development and locked select their respective named splits. Locked material
marked design-exposed or retired through test-driven tuning rejects.

Hashes establish content identity, not the identity/honesty of human reviewers.
There is no signature authority or automatic semantic approval. An unsigned,
fabricated set of mutually consistent files cannot become honest review evidence
by passing these mechanical checks. The CLI explicitly reports
`semantic_authentication:false`.

## Read-only preflight

```bash
.venv/bin/python -m qwen_one_two.training.release /path/to/release \
  --expected-hash <previously recorded manifest SHA256>
```

The command either emits mechanical counts/identity or exits 1 with `REJECTED`.
It imports no model runtime and makes no model/optimizer calls. No real release
is bundled. Data and review artifacts stay under ignored local artifact paths.

Pass `--pilot /path/to/pilot` to check the linked pilot identity and exact
family/input leakage between those two releases. Pilot development families may
be retained as development audit material, but cannot be relabelled as training.
Near-duplicate detection and review of other releases remain external duties.

## Driver and schedule binding

Construct the sampler with `Sampler.from_release(training_release, phase)` and
the native driver with `Driver(window, sampler, training_release, pilot_release,
prerequisites)`. The old raw-record/parent/hash-string driver arguments are
removed; there are no repository callers to migrate. The driver checks the linked
pilot and exact catalogue, including receiver/writer operation, reference status
and domain, before any producer runs. Every window rechecks catalogue integrity.

Each scheduled producer receives a fresh JSON copy of the original record.
Mutating that copy cannot change later repeated exposures. Window reports bind
the training/pilot manifest identities. Producer failure preserves the failed
sample IDs/cursor, aborts the active window and requires explicit resume.

Sampler resume recomputes the identifier-only deterministic schedule to verify
that its counters agree exactly with its window number. This never replays a
model/rollout. Invented, skipped or foreign cursor entries reject transactionally.
Actual Session, graph and optimizer checks retain their separate CUDA evidence.
