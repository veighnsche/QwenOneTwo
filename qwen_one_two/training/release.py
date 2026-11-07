"""Model-free, content-checked ingestion of an externally reviewed release.

Hashes bind declared evidence to bytes. They cannot authenticate a reviewer or
prove honest annotation. This module never generates annotations or approvals.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from .inputs import Source, canonical, check, digest, validate_record_manifest

FORMAT = 'qot-reviewed-release-v1'
REVIEW_POLICY = 'automated-content-v1'
DOMAINS = frozenset({'refund_eligibility', 'scheduling', 'demo_preparation',
                     'access_rules', 'resource_allocation', 'issue_prioritization'})
CATEGORIES = frozenset({'reference', 'writer', 'receiver', 'reader', 'interleaved',
                        'direct', 'language_replay', 'decision_replay'})
_TOKEN = object()


def _object(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result


def decode(data):
    def invalid(value):
        raise ValueError('nonfinite JSON constant: '+value)
    result = json.loads(data, object_pairs_hook=_object, parse_constant=invalid)
    canonical(result)  # Also rejects overflowed JSON numbers, e.g. 1e999.
    return result


def sha256(value):
    check(isinstance(value, str) and len(value) == 64
          and all(c in '0123456789abcdef' for c in value), 'invalid SHA256')
    return value


def annotation_hash(record):
    """Reviews bind every annotation/provenance field except review/release metadata."""
    return digest(canonical({k:v for k,v in record.items() if k not in {'reviews', 'release'}}))


def approval_target(manifest):
    payload = {k:v for k,v in manifest.items() if k != 'approval'}
    # The approval's own bytes cannot hash themselves. The complete manifest
    # identity still binds this descriptor, while the approval binds all peers.
    approval_path = manifest['approval']['artifact_path']
    payload['assets'] = [a for a in manifest['assets'] if a['path'] != approval_path]
    return digest(canonical(payload))


def _path(root, name):
    check(isinstance(name, str) and bool(name), 'missing artifact path')
    relative = PurePosixPath(name)
    check(not relative.is_absolute() and str(relative) == name
          and all(part not in {'.', '..'} for part in relative.parts), 'unsafe artifact path')
    path = root.joinpath(*relative.parts)
    check(path.resolve().is_relative_to(root), 'artifact escapes release directory')
    check(path.is_file(), 'missing release artifact: '+name)
    return path


def _artifact(root, descriptor):
    check(set(descriptor) == {'path', 'sha256'}, 'artifact descriptor fields')
    data = _path(root, descriptor['path']).read_bytes()
    check(digest(data) == sha256(descriptor['sha256']), 'artifact hash mismatch: '+descriptor['path'])
    return data


def _reviews(record, assets):
    for review in record['reviews']:
        path = review.get('artifact_path')
        check(path in assets and digest(assets[path]) == sha256(review['artifact_hash']), 'missing or mismatched review artifact')
        evidence = decode(assets[path])
        check(evidence.get('format') == 'qot-review-v1', 'review artifact format')
        check(evidence.get('record') == {'id':record['id'], 'version':record['version'],
              'annotation_sha256':annotation_hash(record)}, 'review does not bind current annotations')
        for field in ('reviewer', 'kind', 'semantics_blind', 'outcome_independent', 'decision'):
            check(field in review and evidence.get(field) == review[field], 'review evidence mismatch: '+field)
        judgments = evidence.get('judgments', {})
        check(all(isinstance(judgments.get(k), str) and judgments[k].strip()
                  for k in ('semantics', 'outcome')), 'missing independent review judgments')
        if review['kind'] == 'model':
            producer = evidence.get('model_assistance', {})
            check(all(isinstance(producer.get(k), str) and producer[k].strip()
                      for k in ('model', 'session_id')), 'missing review model/session provenance')
            author_session = record.get('model_assistance', {}).get('session_id')
            check(producer['session_id'] != author_session, 'author session cannot review itself')
            for field in ('prompt', 'response'):
                descriptor = producer.get(field, {})
                check(set(descriptor) == {'path', 'sha256'}, 'missing review '+field+' descriptor')
                check(descriptor['path'] in assets and
                      digest(assets[descriptor['path']]) == sha256(descriptor['sha256']),
                      'missing or mismatched review '+field+' bytes')
                check(bool(assets[descriptor['path']].strip()), 'empty review '+field)


def _lineage(records, splits):
    check(set(splits) == {r['family'] for r in records.values()}, 'family split manifest coverage')
    active, done = set(), set()

    def visit(key):
        check(key in records and key not in active, 'missing or cyclic release lineage')
        if key in done:
            return
        active.add(key)
        record = records[key]
        check(record['split'] == splits[record['family']], 'family split manifest mismatch')
        check(all(parent in records for parent in record['parents']), 'missing ancestor in release')
        validate_record_manifest(record, records)
        for parent in record['parents']:
            visit(parent)
        active.remove(key)
        done.add(key)
    for key in records:
        visit(key)


def _batch_audits(manifest, records, selected):
    batches = manifest.get('batches', {})
    for name, batch in batches.items():
        check(set(batch) == {'members', 'second_review'}, 'batch audit fields')
        members, audit = batch['members'], batch['second_review']
        check(0 < len(members) <= 200 and len(set(members)) == len(members)
              and set(members) <= set(records), 'batch size or unknown/duplicate members')
        check(len(set(audit)) == len(audit) and set(audit) <= set(members), 'batch audit membership')
        check(all(records[k].get('batch') == name for k in members), 'batch/record mismatch')
        # Excluded/quarantined batches remain in the audit trail without being
        # required to pass review. Selected members still need a passing audit.
        if not set(members) & selected:
            continue
        # Second review is optional. If a release declares it, verify the claim.
        if not audit:
            continue
        for key in audit:
            record = records[key]
            validate_record_manifest(record, records, for_release=True)
            check(len({r['reviewer'] for r in record['reviews']}) >= 2, 'audit needs two independent reviewers')
        for field in ('operation', 'domain', 'transformation'):
            def stratum(key):
                return records[key]['construction'].get(field) if field == 'transformation' else records[key].get(field)
            check({stratum(k) for k in members} <= {stratum(k) for k in audit}, 'uncovered batch audit stratum: '+field)
        check({records[k]['labels'].get('status') for k in members} <=
              {records[k]['labels'].get('status') for k in audit}, 'uncovered batch audit status')


@dataclass(frozen=True, init=False)
class ReviewedRelease:
    """Immutable byte snapshot; JSON properties return fresh copies to callers."""
    identity: str
    purpose: str
    selected_ids: frozenset[str]
    _manifest: bytes
    _records: MappingProxyType
    _assets: tuple
    _asset_index: MappingProxyType
    _summary: bytes

    def __init__(self, identity, manifest, records, assets, summary, *, selected_ids, _token=None):
        check(_token is _TOKEN, 'use load_release to validate release evidence')
        snapshots = MappingProxyType({key:canonical(record) for key,record in records.items()})
        # Derive only from the loader's fully validated metadata. These small
        # immutable fields avoid reparsing the whole manifest per exposure.
        for name, value in (('identity', identity), ('purpose', summary['purpose']),
                            ('selected_ids', frozenset(selected_ids)),
                            ('_manifest', manifest), ('_records', snapshots),
                            ('_assets', tuple(sorted(assets.items()))),
                            ('_asset_index', MappingProxyType(dict(assets))),
                            ('_summary', canonical(summary))):
            object.__setattr__(self, name, value)

    @property
    def manifest(self):
        return decode(self._manifest)

    @property
    def records(self):
        return {key:self.record(key) for key in self._records}

    def record(self, key):
        """Decode only this record; each exposure receives its own fresh copy."""
        return decode(self._records[key])

    @property
    def summary(self):
        return decode(self._summary)

    def asset(self, path):
        return self._asset_index[path]

    def sampler_rows(self):
        check(self.purpose == 'training', 'sampler requires a training release')
        records = self.records
        rows = []
        for entry in self.manifest['records']:
            if not entry['selected']:
                continue
            record = records[entry['id']]
            category = record.get('training_category')
            check(category in CATEGORIES, 'missing or unknown training category')
            row = {'id':record['id'], 'category':category, 'split':record['split']}
            if category in {'writer', 'receiver'}:
                row.update(operation=record['operation'], status=record['labels']['status'], domain=record['domain'])
            rows.append(row)
        check(bool(rows), 'empty training catalogue')
        return rows


def load_release(path, *, expected_hash=None):
    """Inspect local bytes only. Never write, approve, train or contact a model."""
    root = Path(path).resolve()
    raw = _path(root, 'manifest.json').read_bytes()
    identity = digest(raw)
    if expected_hash is not None:
        check(identity == sha256(expected_hash), 'release manifest hash mismatch')
    manifest = decode(raw)
    required = {'format', 'id', 'version', 'state', 'purpose', 'records', 'assets',
                'family_splits', 'split_sha256', 'exclusions', 'approval'}
    check(required <= set(manifest) and manifest['format'] == FORMAT, 'release format/fields')
    check(isinstance(manifest['id'], str) and manifest['id'].strip() and type(manifest['version']) is int
          and manifest['version'] > 0 and manifest['state'] == 'approved', 'unapproved release identity/state')
    purpose = manifest['purpose']
    check(purpose in {'pilot', 'training', 'development', 'locked'}, 'release purpose')
    check(digest(canonical(manifest['family_splits'])) == sha256(manifest['split_sha256']), 'split manifest hash mismatch')
    assets = {}
    used_paths = {'manifest.json'}
    for entry in manifest['assets']:
        name = entry['path']
        check(name not in used_paths, 'duplicate release artifact path')
        used_paths.add(name)
        assets[name] = _artifact(root, entry)
    records, selected = {}, set()
    for entry in manifest['records']:
        check(set(entry) == {'id', 'path', 'sha256', 'selected'} and type(entry['selected']) is bool, 'record descriptor fields')
        check(entry['path'] not in used_paths and entry['id'] not in records, 'duplicate record/artifact')
        used_paths.add(entry['path'])
        record = decode(_artifact(root, {k:entry[k] for k in ('path', 'sha256')}))
        check(record.get('id') == entry['id'], 'release record identity mismatch')
        records[entry['id']] = record
        if entry['selected']:
            selected.add(entry['id'])
    check(bool(selected), 'empty selected release')
    _lineage(records, manifest['family_splits'])
    exclusions = manifest['exclusions']
    check(set(exclusions) == {'records', 'families', 'batches'}, 'exclusion fields')
    check(set(exclusions['records']) <= set(records) and set(exclusions['families']) <= set(manifest['family_splits'])
          and set(exclusions['batches']) <= set(manifest.get('batches', {})), 'unknown exclusion')
    blocked_batches = set(exclusions['batches']) | {
        r['batch'] for r in records.values() if r.get('quarantined') and r.get('batch') is not None}
    blocked = {k for k, r in records.items() if k in exclusions['records'] or r['family'] in exclusions['families']
               or r.get('batch') in blocked_batches or r.get('quarantined') or r.get('unresolved')
               or r.get('design_excluded')}
    while True:
        descendants = {k for k,r in records.items() if set(r['parents']) & blocked}
        if descendants <= blocked:
            break
        blocked.update(descendants)
    check(not selected & blocked, 'excluded or quarantined lineage selected')
    source_versions, duplicates = {}, {}
    review_count = 0
    for key, record in records.items():
        check(len({s['id'] for s in record['sources']}) == len(record['sources']), 'duplicate source identity in record')
        for source in record['sources']:
            check(set(source) == {'id', 'version', 'path', 'sha256'} and source['path'] in assets, 'source descriptor/asset')
            content = assets[source['path']]
            Source(source['id'], source['version'], content, source['sha256']).validate()
            ref = (source['id'], source['version'])
            check(ref not in source_versions or source_versions[ref] == source['sha256'], 'source identity/version collision')
            source_versions[ref] = source['sha256']
        if key not in selected:
            continue
        validate_record_manifest(record, records, for_training=purpose == 'training', for_release=True)
        check(record['release'] == {'state':'approved', 'id':manifest['id'], 'version':manifest['version']}, 'record/release binding')
        expected_split = 'development' if purpose == 'pilot' else ('train' if purpose == 'training' else purpose)
        check(record['split'] == expected_split, 'release purpose/split mismatch')
        check({'scenario_spec', 'template_version', 'seed', 'transformation'} <= set(record['construction']), 'construction provenance fields')
        check(isinstance(record['labels'].get('schema_version'), str) and record['labels']['schema_version'], 'label schema version')
        human = record.get('human_input')
        check(isinstance(human, dict) and isinstance(human.get('prompt'), str) and human['prompt'].strip(), 'missing human input')
        fingerprint = digest(canonical(human))
        check(fingerprint not in duplicates or duplicates[fingerprint] == record['split'], 'duplicate human input across splits')
        duplicates[fingerprint] = record['split']
        _reviews(record, assets)
        review_count += len(record['reviews'])
    # Check duplicate source tasks across selected and retained audit ancestors.
    for record in records.values():
        if isinstance(record.get('human_input'), dict):
            fingerprint = digest(canonical(record['human_input']))
            check(fingerprint not in duplicates or duplicates[fingerprint] == record['split'], 'duplicate human input across splits')
    _batch_audits(manifest, records, selected)
    # Audited records not selected still need their review evidence checked.
    for batch in manifest.get('batches', {}).values():
        if set(batch['members']) & selected:
            for key in batch['second_review']:
                _reviews(records[key], assets)
    if purpose == 'pilot':
        check(len(selected) == 120 and len({records[k]['family'] for k in selected}) == 120, 'pilot needs 120 independent scenario families')
        check(len(duplicates) == 120, 'pilot repeats identical human inputs')
        check(all(records[k].get('pilot') is True and records[k].get('fictional') is True
                  and records[k]['origin'] in {'authored', 'model_proposal'} and not records[k]['parents'] for k in selected),
              'pilot root/fictional provenance')
        check(Counter(records[k].get('domain') for k in selected) == Counter({d:20 for d in DOMAINS}), 'pilot domain allocation')
    if purpose == 'locked':
        check(not any(records[k].get('design_exposed') or records[k].get('test_driven_tuning') for k in selected),
              'exposed or retired locked material')
    if purpose == 'training':
        sha256(manifest.get('pilot_release_sha256'))
    approval = manifest['approval']
    check(set(approval) == {'artifact_path', 'artifact_hash'} and approval['artifact_path'] in assets, 'release approval artifact')
    data = assets[approval['artifact_path']]
    check(digest(data) == sha256(approval['artifact_hash']), 'release approval hash')
    evidence = decode(data)
    check(evidence.get('format') == 'qot-release-approval-v1'
          and evidence.get('release_sha256') == approval_target(manifest), 'release approval binding')
    check(isinstance(evidence.get('approver'), str) and evidence['approver'].strip()
          and evidence.get('kind') in {'human', 'automated'} and evidence.get('decision') == 'approved', 'release approval decision')
    if evidence['kind'] == 'automated':
        check(evidence.get('policy') == REVIEW_POLICY, 'automated release approval policy')
    summary = {'format':FORMAT, 'release_sha256':identity, 'id':manifest['id'], 'version':manifest['version'],
               'purpose':purpose, 'selected_records':len(selected), 'retained_audit_records':len(records)-len(selected),
               'blocked_records':len(blocked), 'selected_review_artifacts':review_count,
               'selected_domains':dict(Counter(records[k].get('domain', 'unspecified') for k in selected)),
               'selected_operations':dict(Counter(records[k].get('operation', 'unspecified') for k in selected)),
               'selected_statuses':dict(Counter(records[k]['labels'].get('status', 'unlabelled') for k in selected)),
               'selected_categories':dict(Counter(records[k].get('training_category', 'unspecified') for k in selected)),
               'review_policy':REVIEW_POLICY, 'approval_kind':evidence['kind'],
               'selected_review_kinds':dict(Counter(r['kind'] for k in selected for r in records[k]['reviews'])),
               'semantic_authentication':False}
    return ReviewedRelease(identity, raw, records, assets, summary, selected_ids=selected, _token=_TOKEN)


def bind_pilot(training, pilot):
    """Bind the prerequisite and catch exact leakage between these two releases."""
    check(isinstance(training, ReviewedRelease) and isinstance(pilot, ReviewedRelease),
          'loaded training and pilot releases required')
    check(training.manifest['purpose'] == 'training' and pilot.manifest['purpose'] == 'pilot',
          'training/pilot release purpose')
    check(training.manifest['pilot_release_sha256'] == pilot.identity, 'training/pilot release binding')
    train_splits, pilot_splits = training.manifest['family_splits'], pilot.manifest['family_splits']
    check(all(train_splits[k] == pilot_splits[k] for k in set(train_splits) & set(pilot_splits)),
          'pilot/training family split leakage')
    pilot_inputs = {digest(canonical(r['human_input'])):r['split'] for r in pilot.records.values()
                    if isinstance(r.get('human_input'), dict)}
    for record in training.records.values():
        if isinstance(record.get('human_input'), dict):
            fingerprint = digest(canonical(record['human_input']))
            check(fingerprint not in pilot_inputs or pilot_inputs[fingerprint] == record['split'],
                  'pilot/training duplicate input leakage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release')
    parser.add_argument('--expected-hash')
    parser.add_argument('--pilot', help='also check training/pilot identity and cross-release split leakage')
    args = parser.parse_args()
    try:
        release = load_release(args.release, expected_hash=args.expected_hash)
        if args.pilot:
            bind_pilot(release, load_release(args.pilot))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, 'REJECTED: '+str(exc)+'\n')
    print(json.dumps(release.summary, sort_keys=True))


if __name__ == '__main__':
    main()
