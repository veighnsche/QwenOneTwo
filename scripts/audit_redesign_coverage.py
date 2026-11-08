"""Prepare RD-01 metadata evidence; never invokes a model or fits anything."""
import argparse
import json
from pathlib import Path

from qwen_one_two.training.architecture_audit import redesign_manifest
from qwen_one_two.training.inputs import canonical, check, digest
from qwen_one_two.training.release import load_release
from scripts.fit_reference import TRAIN_SHA, DEV_SHA


def write_evidence(output, result, root):
    check(output.is_relative_to(root/'data') and not output.exists(), 'fresh local data directory required')
    # The portable index contains counts and identities; raw row/panel lists stay local.
    inventory = result.pop('inventory')
    panels = result.pop('panels')
    pairs = result.pop('pairs')
    omitted = result.pop('separate_reader_X_ids')
    sources = ('qwen_one_two/training/architecture_audit.py', 'qwen_one_two/training/redesign_coverage.py',
               'qwen_one_two/training/learning_report.py', 'scripts/audit_redesign_coverage.py',
               'scripts/audit_latent_semantics.py', 'qwen_one_two/training/component_producer.py',
               'qwen_one_two/training/binding.py', 'qwen_one_two/training/producer.py',
               'qwen_one_two/training/release.py', 'qwen_one_two/training/inputs.py',
               'qwen_one_two/training/clean.py', 'qwen_one_two/training/behavior_qualification.py',
               'scripts/produce_training_data.py')
    result['source_hashes'] = {name: digest((root/name).read_bytes()) for name in sources}
    result['reused_audits'] = {}
    for name in ('training-x-semantic-audit-v1', 'architecture-baseline-v1'):
        path = root/'docs/evidence'/f'{name}.json'
        result['reused_audits'][str(path.relative_to(root))] = digest(path.read_bytes())
        previous = json.loads(path.read_bytes())
        if 'raw_path' in previous:
            bound = {previous['raw_path']: previous['raw_sha256']}
        else:
            bound = {p: sha for p, sha in previous['raw_hashes'].items() if p.endswith('/report.json')}
        for relative, sha in bound.items():
            check(digest((root/relative).read_bytes()) == sha, 'stale historical audit: '+relative)
        result['reused_audits'].update(bound)
    output.mkdir(parents=True)
    artifacts = {'inventory.jsonl': b''.join(canonical(e)+b'\n' for e in inventory),
                 'panels.json': canonical({'panels': panels, 'pairs': pairs, 'separate_reader_X_ids': omitted})}
    for name, body in artifacts.items():
        (output/name).write_bytes(body)
    result['artifacts'] = {str((output/name).relative_to(root)): digest(body) for name, body in artifacts.items()}
    result['inventory_records'] = len(inventory)
    result['separate_reader_X_records'] = len(omitted)
    (output/'coverage.json').write_bytes(canonical(result)+b'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    check(output.is_relative_to(root/'data') and not output.exists(), 'fresh local data directory required')
    releases = root/'data/training-collections-20261105/releases'
    train = load_release(releases/'train', expected_hash=TRAIN_SHA)
    dev = load_release(releases/'development', expected_hash=DEV_SHA)
    result = write_evidence(output, redesign_manifest(train, dev), root)
    print(json.dumps({k: result[k] for k in ('protocol', 'collections', 'receiver_rows_accounted',
                                           'training_support', 'native_supervision', 'artifacts')}))


if __name__ == '__main__':
    main()
