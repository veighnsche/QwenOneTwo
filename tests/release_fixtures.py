"""Temporary schema fixtures ONLY. No authored tasks or actual human reviews.

Human/approval markers below test serialization gates. They are not attestations
and must never be used as semantic data, reviewer evidence or release approval.
"""
from copy import deepcopy
from pathlib import Path
from qwen_one_two.training.inputs import canonical, digest
from qwen_one_two.training.release import FORMAT, DOMAINS, annotation_hash, approval_target


def row(key='fixture', *, split='development', **extra):
    result = dict(id=key, version=1, family='family-'+key, split=split, parents=[],
                  origin='authored', construction={'scenario_spec':'schema-only, no scenario',
                  'template_version':'test-only-v1', 'seed':17, 'transformation':'identity'},
                  sources=[], labels={'schema_version':'test-only-v1'}, reviews=[],
                  release={'state':'approved', 'id':'TEST-ONLY', 'version':1},
                  author='TEST-ONLY-author', domain='access_rules',
                  human_input={'prompt':'SCHEMA FIXTURE ONLY: '+key})
    result.update(extra)
    return result


def pilot_rows():
    return [row(f'{domain}-{i}', domain=domain, pilot=True, fictional=True)
            for domain in sorted(DOMAINS) for i in range(20)]


def write_release(root, records=None, *, purpose='development', selected=None, batches=None,
                  exclusions=None, assets=None, review_counts=None, mutate_review=None,
                  pilot_hash='a'*64, review_kind='human', approval_kind='human'):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    records = deepcopy(records if records is not None else [row()])
    assets = dict(assets or {})
    review_counts = review_counts or {}
    for record in records:
        key = record['id']
        record['reviews'] = []
        for index in range(review_counts.get(key, 2)):
            reviewer = f'TEST-ONLY-reviewer-{index}'
            evidence = dict(format='qot-review-v1', record={'id':key, 'version':record['version'],
                            'annotation_sha256':annotation_hash(record)}, reviewer=reviewer,
                            kind=review_kind, semantics_blind=True, outcome_independent=True,
                            decision='approved', judgments={'semantics':'SCHEMA FIXTURE; no review performed',
                                                           'outcome':'SCHEMA FIXTURE; no outcome exists'})
            if review_kind == 'model':
                producer = dict(model='TEST-ONLY-model', session_id='TEST-ONLY-review-session')
                for field in ('prompt', 'response'):
                    path = f'reviews/{key}-{index}-{field}.txt'
                    assets[path] = ('SCHEMA ONLY; no model executed: '+field).encode()
                    producer[field] = dict(path=path, sha256=digest(assets[path]))
                evidence['model_assistance'] = producer
            if mutate_review:
                mutate_review(evidence)
            name = f'reviews/{key}-{index}.json'
            data = canonical(evidence); assets[name] = data
            record['reviews'].append(dict(reviewer=reviewer, kind=review_kind, semantics_blind=True,
                    outcome_independent=True, decision='approved', artifact_path=name,
                    artifact_hash=digest(data), record_version=record['version']))
    manifest = dict(format=FORMAT, id='TEST-ONLY', version=1, state='approved', purpose=purpose,
                    records=[], assets=[], family_splits={r['family']:r['split'] for r in records},
                    exclusions=exclusions or {'records':[], 'families':[], 'batches':[]},
                    approval={'artifact_path':'approval.json', 'artifact_hash':'0'*64})
    if purpose == 'training':
        manifest['pilot_release_sha256'] = pilot_hash
    if batches is not None:
        manifest['batches'] = batches
    manifest['split_sha256'] = digest(canonical(manifest['family_splits']))
    for record in records:
        data = canonical(record); name = 'records/'+record['id']+'.json'
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        manifest['records'].append(dict(id=record['id'], path=name, sha256=digest(data),
                                       selected=selected is None or record['id'] in selected))
    for name, data in assets.items():
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        manifest['assets'].append(dict(path=name, sha256=digest(data)))
    approve(root, manifest, kind=approval_kind)
    return manifest


def approve(root, manifest, *, kind='human'):
    """Test-only consistency helper; never supplies a real approval."""
    evidence = dict(format='qot-release-approval-v1', release_sha256=approval_target(manifest),
                    approver='TEST-ONLY-schema-marker', kind=kind, decision='approved')
    if kind == 'automated':
        evidence['policy'] = 'automated-content-v1'
    data = canonical(evidence); name = manifest['approval']['artifact_path']
    (Path(root)/name).write_bytes(data)
    manifest['approval']['artifact_hash'] = digest(data)
    manifest['assets'] = [a for a in manifest['assets'] if a['path'] != name]
    manifest['assets'].append(dict(path=name, sha256=digest(data)))
    (Path(root)/'manifest.json').write_bytes(canonical(manifest))


def replace_record(root, manifest, key, transform):
    entry = next(e for e in manifest['records'] if e['id'] == key)
    from qwen_one_two.training.release import decode
    target = Path(root)/entry['path']; record = decode(target.read_bytes())
    transform(record); data = canonical(record); target.write_bytes(data)
    entry['sha256'] = digest(data)
    approve(root, manifest)
