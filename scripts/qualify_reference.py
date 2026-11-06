"""Read-only RF qualification; explicit diagnostic run, never training/data authoring."""
import argparse
import json
from pathlib import Path
from qwen_one_two.training.release import load_release
from qwen_one_two.training.reference_qualification import suite,relations
from qwen_one_two.training.inputs import digest,check


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--development-release',required=True)
    parser.add_argument('--release-sha256',required=True)
    parser.add_argument('--snapshot')
    parser.add_argument('--diagnostic-checkpoint',default='data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
    parser.add_argument('--reference-bundle')
    parser.add_argument('--bundle-sha256')
    parser.add_argument('--diagnostic-untrained',action='store_true')
    parser.add_argument('--contrast-only',action='store_true',help='measure existing contrasts without repeating completed recovery')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    release=load_release(args.development_release,expected_hash=args.release_sha256)
    ids=suite(release);groups=relations(release)
    check(not (args.diagnostic_untrained and args.reference_bundle),'choose artifact or diagnostic, not both')
    if not args.reference_bundle and not args.diagnostic_untrained:
        report={'status':'BLOCKED','blockers':['no frozen reference artifact supplied; fitting is not authorized'],
                'scheduled_ids':ids,'recovery_scheduled':len(ids),'contrast_groups':len(groups),
                'development_release_sha256':release.identity,'model_work_performed':False,'optimizer_updates':0}
    else:
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from safetensors import safe_open
        from qwen_one_two.g1_proof import configure
        from qwen_one_two.codec.qualify import load
        from qwen_one_two.training.reference import Reference
        from qwen_one_two.training.reference_qualification import evaluate,evaluate_contrasts,source_hashes
        from qwen_one_two.training.manifest import file_hash,BASE_SHA
        check(args.snapshot is not None,'pinned snapshot required')
        check(file_hash(Path(args.snapshot)/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base hash mismatch')
        configure()
        with sdpa_kernel(SDPBackend.MATH):
            backbone,codec,_=load(args.snapshot,args.diagnostic_checkpoint)
            reference=Reference(backbone,codec)
            if args.reference_bundle:
                root=Path(args.reference_bundle);raw=(root/'manifest.json').read_bytes()
                check(args.bundle_sha256 is not None and digest(raw)==args.bundle_sha256,'bundle manifest hash')
                manifest=json.loads(raw)
                check(manifest.get('pooler_kind','shared-v1') in {'shared-v1','target-v2'},'unknown reference pooler kind')
                if manifest.get('pooler_kind','shared-v1')=='target-v2':reference=Reference(backbone,codec,pooler_kind='target-v2')
                check(manifest['format'] in {'qot-training-bundle-v1','qot-frozen-reference-v1'}
                      and manifest['reference_frozen'] is True,'frozen reference bundle required')
                weights=root/'weights.safetensors'
                check(file_hash(weights)==manifest['files']['weights.safetensors'],'bundle weights hash')
                with safe_open(weights,framework='pt',device='cuda') as tensors:
                    expected=reference.state_dict()
                    state={name:tensors.get_tensor('reference.'+name) for name in expected}
                reference.load_state_dict(state,strict=True)
                reference.freeze(manifest['reference_id'])
            else:reference.freeze()
            if args.contrast_only:
                scored=evaluate_contrasts(reference,release)
                report={'protocol':'reference-contrast-v1','status':'EVIDENCE_ONLY',
                    'blockers':['contrast-only evidence cannot qualify a reference'],
                    'reference_sha256':reference.artifact_id,'development_release_sha256':release.identity,
                    'groups':scored,'contrast':{'scheduled':len(scored),'passed':sum(g['passed'] for g in scored)},
                    'source_hashes':source_hashes(),'optimizer_updates':0,
                    'gpu_uuid':'GPU-225509c3-d702-6c27-de03-76b5e99b33b1'}
            else:report=evaluate(reference,release)
            report['diagnostic_untrained']=args.diagnostic_untrained
            if args.diagnostic_untrained:
                report['status']='BLOCKED'
                report['blockers'].append('untrained diagnostic fixture is not a qualified reference artifact')
            report['model_work_performed']=True
    with Path(args.output).open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','blockers')}))
    return 0 if report['status']=='PASSED' else 2


if __name__=='__main__':raise SystemExit(main())
