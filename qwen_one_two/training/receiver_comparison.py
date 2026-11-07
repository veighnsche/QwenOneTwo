"""Model-free binding and loss selection for one matched value-only comparison."""
from .inputs import check
from .manifest import file_hash
from .release import decode
from .receiver_repair import visit_ids
from .two_gpu import ROOT, CHECKPOINT_SHA, RECEIVER, REFERENCE

PROTOCOL = 'receiver-value-only-comparison-v1'
CONTROL = ROOT/'data/receiver-repair-20261107/run'


def value_losses(losses):
    check(set(losses) <= {'status', 'noul', 'score'}, 'receiver components only')
    return {name: value for name, value in losses.items() if name != 'status'}


def load_control(ids):
    """Pin saved observations and all unchanged model/optimizer implementations."""
    evidence_path = ROOT/'docs/evidence/receiver-residual-repair-v1.json'
    evidence = decode(evidence_path.read_bytes())
    check(evidence['status'] == 'COMPLETED_REPAIR_NOT_PROMOTED' and evidence['audit']['audit_pass'], 'completed control required')
    check(file_hash(CONTROL/'plan.json') == evidence['plan_sha256'] and
          file_hash(CONTROL/'report.json') == evidence['report_sha256'], 'control plan/report changed')
    plan = decode((CONTROL/'plan.json').read_bytes())
    report = decode((CONTROL/'report.json').read_bytes())
    check(plan['parent_manifest_sha256'] == CHECKPOINT_SHA and plan['receiver'] == RECEIVER
          and plan['reference'] == REFERENCE and plan['ids'] == ids, 'matched parent/reference/order')
    for name, sha in plan['sources'].items():
        check(file_hash(CONTROL/'source'/name) == sha, 'control source snapshot changed: '+name)
        # Only orchestration gains an explicit comparison mode; model, loss,
        # ownership, optimizer, schedule and numeric implementations must match.
        if name != 'scripts/train_receiver_repair.py':
            check(file_hash(ROOT/name) == sha, 'matched implementation changed: '+name)
    names = ['features.safetensors','features.jsonl','training.jsonl','heads-0000.jsonl',
             'development-E.jsonl','development-X.jsonl','checkpoint-0000/manifest.json']
    initial = decode((CONTROL/'checkpoint-0000/manifest.json').read_bytes())
    names += ['checkpoint-0000/'+name for name in initial['files']]
    for name in names:
        path = CONTROL/name
        check(file_hash(path) == evidence['raw_hashes'][str(path.relative_to(ROOT))], 'matched artifact changed: '+name)
    check(initial['scheduled_windows'] == initial['optimizer_updates'] == 0 and
          initial['sampler']['order'] == ids and initial['sampler']['parent_manifest'] == CHECKPOINT_SHA,
          'original unfitted initialization required')
    windows = [decode(line) for line in (CONTROL/'training.jsonl').read_bytes().splitlines()]
    check(len(windows) == plan['updates'] == 2000, 'complete control schedule')
    empty = []
    for i, window in enumerate(windows, 1):
        check([v['record'] for v in window['visits']] == visit_ids(ids, i), 'control exposure order')
        if not any(value_losses(v['losses']) for v in window['visits']): empty.append(i)
    binding = {'path':str(CONTROL.relative_to(ROOT)), 'evidence_sha256':file_hash(evidence_path),
               'plan_sha256':evidence['plan_sha256'], 'report_sha256':evidence['report_sha256'],
               'initial_manifest_sha256':file_hash(CONTROL/'checkpoint-0000/manifest.json'),
               'empty_scalar_windows':empty, 'expected_optimizer_updates':2000-len(empty),
               'only_intervention':'remove status loss before eligible-count accumulation'}
    return binding, plan, report, initial
