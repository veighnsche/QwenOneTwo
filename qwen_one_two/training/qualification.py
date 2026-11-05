"""Read-only validation of existing numeric evidence; never executes a model."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from ..codec.qualify_matrix import GPU_UUID, validate_coverage
from .inputs import canonical, check, digest
from .manifest import DIAGNOSTIC_SHA
from .release import _artifact, _path, decode, sha256

_TOKEN = object()
PROFILE = 'b1-full-v3-feedback-fp32-vocab-compensated2-temporal32'
PRECISION = {'frozen_candidate_storage':'BF16', 'feedback_dense_arithmetic':'fp32',
             'native_vocabulary_arithmetic':'2-part BF16 products, FP32 output',
             'reference':'genuine FP32 PyTorch projections; no reference rounding'}


@dataclass(frozen=True, init=False)
class NumericEvidence:
    identity: str
    _summary: bytes

    def __init__(self, identity, summary, *, _token=None):
        check(_token is _TOKEN, 'load existing numeric evidence first')
        object.__setattr__(self, 'identity', identity)
        object.__setattr__(self, '_summary', canonical(summary))

    @property
    def summary(self):
        return decode(self._summary)


def load_numeric_evidence(path, *, expected_hash):
    """Check pinned aggregate, children, axes, invariants and archived sources.

    This checks the recorded source snapshot, not equivalence with subsequently
    edited code. Profile/code changes still require their documented review and
    affected qualification. Passing does not qualify actual A4 trainer memory.
    """
    root = Path(path).resolve()
    raw = _path(root, 'report.json').read_bytes()
    identity = digest(raw)
    check(identity == sha256(expected_hash), 'numeric aggregate hash mismatch')
    report = decode(raw)
    check(report.get('protocol') == 'v3-1024-v1' and report.get('protocol_complete') is True
          and report.get('status') == 'PASSED' and report.get('requested_lengths') == [1024],
          'complete 1024 numeric protocol required')
    check(report.get('gpu_uuid') == GPU_UUID and report.get('precision') == PRECISION
          and report.get('profile') == PROFILE and report.get('checkpoint_sha256') == DIAGNOSTIC_SHA,
          'numeric GPU/profile/checkpoint mismatch')
    check(report.get('all_live_parameters_unchanged') is True
          and report.get('frozen_base_all_groups_pass') is True, 'numeric parameter integrity')
    check(report.get('training_performed') is False and report.get('optimizer_updates') == 0,
          'numeric-only evidence required')
    validate_coverage(report['cases'], 'v3-1024-v1')
    check(all(row['comparison'].get('pass') is True for row in report['cases']), 'numeric case pass markers')
    expected_groups = {f'1024-port{i}' for i in (1,2,3)}
    check(set(report['child_reports']) == expected_groups, 'numeric child group coverage')
    combined = []
    for name, child_hash in report['child_reports'].items():
        child = decode(_artifact(root, {'path':name+'/report.json', 'sha256':child_hash}))
        check(child.get('status') == 'PARTIAL_MATRIX_PASSED' and child.get('requested_checks_passed') is True
              and child.get('all_live_parameters_unchanged') is True, 'numeric child pass/integrity')
        frozen = child.get('frozen_base', {})
        check(frozen.get('pass') is True and frozen.get('base_tied_storage') is True
              and frozen.get('parameters_checked') == 320 and frozen.get('mismatches') == [], 'numeric frozen/tied base')
        for field in ('source_hashes', 'checkpoint_sha256', 'precision', 'profile', 'environment', 'state_memory_reservation'):
            check(child[field] == report[field], 'numeric child identity mismatch: '+field)
        check(all(row['port'] == int(name[-1]) for row in child['cases']), 'foreign child operation')
        combined.extend(dict(row, evidence_group=name) for row in child['cases'])
    key = lambda row:(row['length'],row['port'],row['profile'],row['seed'],row['repeat'])
    check(sorted(combined,key=key) == sorted(report['cases'],key=key), 'aggregate/child trace mismatch')
    cells = set()
    for row in report['cases']:
        summary = row['summary']
        check(summary.get('prefills') == 1 and summary.get('lineage') == 1, 'numeric prefill/lineage')
        check(all(type(summary.get(k)) is int and summary[k] >= 0 for k in ('P','B','A','H'))
              and sum(summary[k] for k in ('P','B','A','H')) == summary.get('total') == 1024,
              'numeric exact position accounting')
        check(summary.get('state') == 'Terminal' and summary.get('held_positions') == 0
              and summary.get('dispatched') == summary.get('consumed') == len(row['calls']),
              'numeric dispatch/consumption settlement')
        cells.update((row['port'], call['status']) for call in row['calls'])
    expected_cells = {(port,status) for port in (1,2,3) for status in ((1,2,3,4,5) if port == 2 else (1,2,3,4))}
    check(cells == expected_cells, 'numeric operation/status coverage')
    snapshot = root/'source-snapshot'
    source_raw = _path(snapshot, 'manifest.json').read_bytes()
    source_manifest = decode(source_raw)
    for name, value in source_manifest.items():
        _artifact(snapshot, {'path':name, 'sha256':value})
    check(bool(report['source_hashes']), 'missing qualified source identity')
    for name, value in report['source_hashes'].items():
        check(source_manifest.get('qwen_one_two/codec/'+name) == value, 'qualified/archive source mismatch')
    return NumericEvidence(identity, {'protocol':'v3-1024-v1', 'protocol_complete':True, 'status':'PASSED',
        'report_hash':identity, 'cases':len(report['cases']), 'gpu_uuid':GPU_UUID, 'profile':PROFILE,
        'child_reports':report['child_reports'], 'source_snapshot_manifest_sha256':digest(source_raw),
        'source_hashes':report['source_hashes'], 'scope':'recorded numeric traces; no A4 trainer memory or semantic certification'},
        _token=_TOKEN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle'); parser.add_argument('--expected-hash', required=True)
    args = parser.parse_args()
    try:
        evidence = load_numeric_evidence(args.bundle, expected_hash=args.expected_hash)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, 'REJECTED: '+str(exc)+'\n')
    print(json.dumps(evidence.summary, sort_keys=True))


if __name__ == '__main__':
    main()
