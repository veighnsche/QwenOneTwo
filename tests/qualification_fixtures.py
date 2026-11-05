"""Schema-only report-shaped objects. NO numeric/model execution or results.

Pass markers exercise loader rejection gates and are never measured evidence.
These temporary fixtures must not enter qualification or training artifacts.
"""
import itertools
from pathlib import Path
from qwen_one_two.codec.qualify_matrix import GPU_UUID
from qwen_one_two.training.inputs import canonical, digest
from qwen_one_two.training.manifest import DIAGNOSTIC_SHA
from qwen_one_two.training.qualification import PRECISION, PROFILE


def write_numeric_fixture(root, *, mutate=None):
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    source = b'# TEST ONLY; not model source or a measured result\n'
    snapshot = root/'source-snapshot/qwen_one_two/codec'
    snapshot.mkdir(parents=True,exist_ok=True); (snapshot/'schema.py').write_bytes(source)
    (root/'source-snapshot/manifest.json').write_bytes(canonical({'qwen_one_two/codec/schema.py':digest(source)}))
    report = dict(status='PASSED',protocol='v3-1024-v1',protocol_complete=True,requested_lengths=[1024],
                  gpu_uuid=GPU_UUID,precision=PRECISION,profile=PROFILE,checkpoint_sha256=DIAGNOSTIC_SHA,
                  all_live_parameters_unchanged=True,frozen_base_all_groups_pass=True,training_performed=False,
                  optimizer_updates=0,environment={'fixture':'schema only; no execution'},state_memory_reservation={},
                  source_hashes={'schema.py':digest(source)},cases=[],child_reports={})
    for port in (1,2,3):
        name = f'1024-port{port}'
        statuses = range(1,6) if port == 2 else range(1,5)
        rows = [dict(length=1024,port=port,profile=profile,seed=seed,repeat=repeat,comparison={'pass':True},
                     summary=dict(P=27,B=100,A=891,H=6,total=1024,prefills=1,lineage=1,state='Terminal',
                                  held_positions=0,dispatched=len(statuses),consumed=len(statuses)),
                     calls=[{'status':status} for status in statuses])
                for profile,seed,repeat in itertools.product(('functional','checkpoint','serving','fp32'),(17,29,43),range(3))]
        child = {k:report[k] for k in ('source_hashes','checkpoint_sha256','precision','profile','environment','state_memory_reservation')}
        child.update(status='PARTIAL_MATRIX_PASSED',requested_checks_passed=True,all_live_parameters_unchanged=True,
                     frozen_base={'pass':True,'base_tied_storage':True,'parameters_checked':320,'mismatches':[]},cases=rows)
        if mutate:
            mutate(report,child,port)
        data = canonical(child); target = root/name/'report.json'; target.parent.mkdir(exist_ok=True); target.write_bytes(data)
        report['child_reports'][name] = digest(data)
        report['cases'].extend(dict(row,evidence_group=name) for row in rows)
    raw = canonical(report); (root/'report.json').write_bytes(raw)
    return digest(raw)
