"""Read-only matched before/after objectives on the saved preflight records."""
import argparse,json
from pathlib import Path
from qwen_one_two.training.inputs import check
from qwen_one_two.training.manifest import file_hash,BASE_SHA,DIAGNOSTIC_SHA


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    check(not Path(a.output).exists(),'never overwrite probe evidence')
    root=Path(a.run);run=json.loads((root/'report.json').read_text())
    check(run['status']=='COMPLETED','finish the declared learning experiment first')
    before=json.loads((root/'preflight.json').read_text())['cases']
    checkpoint=root/f"checkpoint-{run['scheduled_windows']:04d}"
    check(file_hash(Path(a.snapshot)/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base hash')
    check(file_hash('data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')==DIAGNOSTIC_SHA,'initial codec hash')
    from qwen_one_two.training.release import load_release
    release=load_release('data/training-collections-20261105/releases/train',expected_hash=run['identities']['dataset'])
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.training.reference import Reference
    from qwen_one_two.training.receiver_learning import groups,ReceiverProducer
    from qwen_one_two.training.checkpoint import restore_model
    from qwen_one_two.training.behavior_qualification import receiver_hash
    configure()
    with sdpa_kernel(SDPBackend.MATH):
        b,c,_=load(a.snapshot,'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        ref=Reference(b,c,pooler_kind='target-v2');ref.freeze();groups(b,c,ref)
        restore_model(checkpoint,b,c,ref,run['identities'])
        fingerprint=receiver_hash(b,c);check(fingerprint==run['receiver_after'],'final receiver mismatch')
        producer=ReceiverProducer(b,c,ref,release);cases=[]
        with torch.no_grad():
            for initial in before:
                s,losses,_=producer(release.record(initial['record']))
                check(s.runtime.terminal=='Completed','matched objective failed')
                final=s.episode_report
                check(final['input_sha256']==initial['input_sha256'],'matched permitted view changed')
                cases.append({'record':initial['record'],'reference_status':initial['reference_status'],
                    'before_predicted_status':initial['predicted_status'],'after_predicted_status':final['predicted_status'],
                    'before_losses':initial['losses'],'after_losses':final['losses'],
                    'before_eligibility':initial['eligibility'],'after_eligibility':final['eligibility']})
        frozen=frozen_check(b,a.snapshot)
        check(frozen['pass'] and frozen['base_tied_storage'] and ref.fingerprint()==run['reference']
              and receiver_hash(b,c)==fingerprint,'read-only probe changed weights')
        summary={}
        for component in ('status','noul','choice','score','feasibility','alignment','structure'):
            pairs=[(v['before_losses'][component],v['after_losses'][component]) for v in cases
                   if component in v['before_losses'] and component in v['after_losses']]
            summary[component]={'matched_examples':len(pairs),
                'before_mean':sum(x for x,y in pairs)/len(pairs) if pairs else None,
                'after_mean':sum(y for x,y in pairs)/len(pairs) if pairs else None}
    result={'protocol':'receiver-matched-objective-probe-v1','cases':cases,'summary':summary,
        'optimizer_updates_during_probe':0,'receiver':fingerprint,'reference':run['reference'],
        'base_check':frozen,'probe_source_sha256':file_hash(__file__),
        'scope':'fixed preflight training examples, not held-out semantic or autonomous qualification'}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
