"""Read-only latent receiver evaluation; never fits, repairs or generates data."""
import argparse
import json
from pathlib import Path
from qwen_one_two.training.inputs import check,digest
from qwen_one_two.training.manifest import file_hash,BASE_SHA
from qwen_one_two.training.release import load_release


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--reference-bundle',required=True)
    p.add_argument('--bundle-sha256',required=True)
    p.add_argument('--development-release',required=True)
    p.add_argument('--release-sha256',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--reuse-cases')
    p.add_argument('--reuse-sources',help='source snapshot directory for the saved measurements')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    check(file_hash(Path(a.snapshot)/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base hash')
    check(file_hash(a.checkpoint)==a.checkpoint_sha256,'live checkpoint hash')
    root=Path(a.reference_bundle);raw=(root/'manifest.json').read_bytes()
    check(digest(raw)==a.bundle_sha256,'reference bundle hash');m=json.loads(raw)
    check(m['format']=='qot-frozen-reference-v1' and m['reference_frozen'] is True,'frozen reference bundle')
    check(file_hash(root/'weights.safetensors')==m['files']['weights.safetensors'],'reference weights hash')
    print('Loading existing reviewed development release',flush=True)
    release=load_release(a.development_release,expected_hash=a.release_sha256)
    from qwen_one_two.training.behavior_qualification import suite
    groups,omitted=suite(release)
    from qwen_one_two.training.behavior_qualification import expected,source_hashes
    for group in groups:
        for key in group['ids']:expected(release.record(key))
    reuse=[];reuse_receipt=None
    if a.reuse_cases:
        import ast
        check(a.reuse_sources is not None,'reused measurements require source identities')
        source=Path(a.reuse_sources);old=json.loads((source/'source_hashes.json').read_bytes());current=source_hashes()
        allowed={'qwen_one_two/training/behavior_qualification.py','scripts/qualify_behavior.py'}
        check(set(old)==set(current) and all(old[k]==v for k,v in current.items() if k not in allowed),
              'reused model source drift')
        name='qwen_one_two/training/behavior_qualification.py'
        saved=(source/'behavior_qualification.py').read_bytes()
        check(digest(saved)==old[name],'saved case runner hash')
        def definitions(body):
            return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(body).body
                    if isinstance(node,ast.FunctionDef)}
        prior,now=definitions(saved),definitions(Path(name).read_bytes())
        check(all(prior[k]==now[k] for k in ('evaluate_case','receiver_hash')),'reused forward drift')
        raw=Path(a.reuse_cases).read_bytes();reuse=[json.loads(line) for line in raw.splitlines()]
        reuse_receipt={'cases_sha256':digest(raw),'source_manifest_sha256':file_hash(source/'source_hashes.json'),
                      'cases':len(reuse),'unchanged_case_forward':True,'reused_grades_recomputed':True}

    print(json.dumps({'scheduled':2*len(groups),'omitted_reader':len(omitted)}),flush=True)
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from safetensors import safe_open
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.training.reference import Reference
    from qwen_one_two.training.behavior_qualification import evaluate
    configure()
    with sdpa_kernel(SDPBackend.MATH):
        b,c,_=load(a.snapshot,a.checkpoint)
        reference=Reference(b,c,pooler_kind=m['pooler_kind'])
        with safe_open(root/'weights.safetensors',framework='pt',device='cuda') as tensors:
            reference.load_state_dict({name:tensors.get_tensor('reference.'+name)
                                       for name in reference.state_dict()},strict=True)
        reference.freeze(m['reference_id']);b.requires_grad_(False);c.requires_grad_(False)
        with (out/'cases.jsonl').open('x') as log:
            def emit(case):
                log.write(json.dumps(case)+'\n');log.flush()
                print(json.dumps({'id':case['id'],'score':case['score'],'failure':case['failure']}),flush=True)
            report=evaluate(b,c,reference,release,emit=emit,reuse=reuse)
        report['reused_evidence']=reuse_receipt
        frozen=frozen_check(b,a.snapshot)
        report.update(base_frozen=frozen['pass'] and frozen['base_tied_storage'],base_check=frozen,
            base_sha256=BASE_SHA,checkpoint_file_sha256=a.checkpoint_sha256,
            reference_bundle_sha256=a.bundle_sha256,
            gradients_none=all(p.grad is None for owner in (b,c,reference) for p in owner.parameters()))
        if not report['base_frozen']:
            report['status']='BLOCKED';report['blockers'].append('base W changed')
        check(report['gradients_none'],'unexpected gradients')
    with (out/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('status','blockers','operations','pairs_passed','transport_valid')}),flush=True)
    return 0 if report['status']=='PASSED' else 2


if __name__=='__main__':raise SystemExit(main())
