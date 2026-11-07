"""Zero-update A/B of literal legacy and fused validation on existing records.

Run on the pinned 3090. Baseline files are preserved source, not reimplemented
math. Parity passes warm both paths; three alternating timed pairs follow per
eight-record batch. No optimizer step, new records, fitting or gate evaluation.
"""
import argparse
import gc
import importlib.util
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch
from qwen_one_two.training.inputs import check,canonical,digest
from qwen_one_two.training.manifest import file_hash,BASE_SHA,DIAGNOSTIC_SHA

LEGACY={'engine.py':'c213615e04515a51161ef3168d4637d3c0f193be2c0bccb7e158ea102d325017',
        'session.py':'3ebce62feb3f0d18b55a893fb2db18e88aeb00fc8dba0808877ce684e3a3534e'}


def legacy(root,name,package):
    path=root/name
    check(file_hash(path)==LEGACY[name],'literal baseline source mismatch')
    spec=importlib.util.spec_from_file_location(package+'._legacy_finite',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
    def emit(**entry):
        print(json.dumps(entry),flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
    check(file_hash(Path(a.snapshot)/'model.safetensors-00001-of-00001.safetensors')==BASE_SHA,'base hash')
    diagnostic='data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors'
    check(file_hash(diagnostic)==DIAGNOSTIC_SHA,'initial codec hash')
    runroot=Path('data/receiver-learning-20261106/continued')
    run=json.loads((runroot/'report.json').read_bytes());checkpoint=runroot/'checkpoint-0188'
    check(file_hash(checkpoint/'manifest.json')=='0cd1d4bc55341e78339c1b0b05dec0b3650573d7501fb3f3d2fc6d2f649b6e58','checkpoint pin')
    from qwen_one_two.training.release import load_release
    release=load_release('data/training-collections-20261105/releases/train',expected_hash=run['identities']['dataset'])
    batches={}
    for line in (runroot/'training.jsonl').read_text().splitlines():
        entry=json.loads(line);n=entry['window']['window']
        if n in (105,111):batches[str(n)]=[e['record'] for e in entry['episodes']]
    check(len(batches)==2 and all(len(v)==8 for v in batches.values()),'two fixed complete windows')
    sources={str(f):file_hash(f) for root in ('qwen_one_two','scripts') for f in Path(root).rglob('*.py')}
    for name in sources:
        target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.codec.qualify import load
    from qwen_one_two.codec.finite import all_finite
    from qwen_one_two.training.reference import Reference
    from qwen_one_two.training.receiver_learning import groups,ReceiverProducer
    from qwen_one_two.training.behavior_qualification import receiver_hash
    from qwen_one_two.training.checkpoint import restore_model
    from qwen_one_two.training.engine import Window
    from qwen_one_two.training.episodes import collect_session
    import qwen_one_two.codec.session as session_module
    configure()
    LegacyWindow=legacy(Path(a.baseline),'engine.py','qwen_one_two.training').Window
    LegacySession=legacy(Path(a.baseline),'session.py','qwen_one_two.codec').Session
    current_session=session_module.Session
    def tensor_hash(t):
        return {'shape':list(t.shape),'dtype':str(t.dtype),
                'bytes_sha256':digest(t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())}
    def tree_hash(x):
        if torch.is_tensor(x):return tensor_hash(x)
        if isinstance(x,dict):return {k:tree_hash(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [tree_hash(v) for v in x]
        return x
    def micro(values):
        old=lambda:all(bool(t.isfinite().all()) for t in values)
        result={}
        for label,fn in [('legacy',old),('fused',lambda:all_finite(values))]:
            for _ in range(5):check(fn(),'finite micro input')
            seconds=[]
            for _ in range(30):
                torch.cuda.synchronize();start=time.perf_counter();check(fn(),'finite micro output')
                torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:fn()
            result[label]={'seconds':seconds,'median_seconds':statistics.median(seconds),
                'host_scalar_decisions':sum(e.count for e in prof.key_averages() if e.key=='aten::_local_scalar_dense')}
        result['tensors']=len(values);result['numel']=sum(t.numel() for t in values)
        return result
    with sdpa_kernel(SDPBackend.MATH):
        b,c,_=load(a.snapshot,diagnostic)
        ref=Reference(b,c,pooler_kind='target-v2');ref.freeze();owned=groups(b,c,ref)
        restore_model(checkpoint,b,c,ref,run['identities']);b.checkpointing=True
        before=receiver_hash(b,c);reference=ref.fingerprint()
        check(before==run['receiver_after'] and reference==run['reference'],'pinned weights')
        base_before=frozen_check(b,a.snapshot);check(base_before['pass'] and base_before['base_tied_storage'],'frozen base')
        producer=ReceiverProducer(b,c,ref,release)
        # Cache only immutable annotation encodings. All D source states remain
        # fresh on every pass, as in the measured training implementation.
        from qwen_one_two.training.component_producer import component_view
        from qwen_one_two.training.binding import annotation
        for ids in batches.values():
            for key in ids:
                row=release.record(key);view,_,_=component_view(release,row)
                for item in view.references:producer.encoder.reference_payload(item)
                if row['supervision'].get('reason') is not None:
                    producer.encoder.reference_payload(annotation('reason',row['supervision']['reason'],row['operation']))
        paths={'legacy':(LegacyWindow,LegacySession),'fused':(Window,current_session)}
        measured=[];parities=[];micro_results={}
        def execute(label,ids,capture=False):
            W,S=paths[label];w=W(owned,'components');episodes=[]
            heads=[];original=c.evaluate
            def evaluate(*args,**kwargs):
                output=original(*args,**kwargs);heads.append(tree_hash(output));return output
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            with patch.object(session_module,'Session',S),patch.object(w.optimizer,'step',side_effect=AssertionError('benchmark forbids updates')):
                with patch.object(c,'evaluate',evaluate if capture else original):
                    for key in ids:
                        s,losses,exposure=producer(release.record(key))
                        check(s.runtime.terminal=='Completed' and losses,'complete eligible benchmark episode')
                        if capture:
                            detail={'report':s.episode_report,'heads':heads.pop(),
                                'losses':tree_hash(losses),'hidden':tensor_hash(s.hidden),
                                'cache':tree_hash(s.state.layers),'snapshots':tree_hash(s.snapshots)}
                        receipt=collect_session(w,s,losses,exposure)
                        episodes.append(detail if capture else {'record':key,'receipt':receipt,'losses':list(losses)})
                        del s,losses,exposure
                w.combine()
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                buffers={k:{i:t.clone() for i,t in v.items()} for k,v in w.buffers.items()} if capture else None
                grads=[None if p.grad is None else p.grad.clone() for p in w.parameters] if capture else None
                stats={'path':label,'seconds':seconds,'eligible':dict(w.counts),'vjp_count':sum(w.counts.values()),
                    'active_parameters':sum(p.grad is not None for p in w.parameters),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'episodes':episodes,
                    'optimizer_updates':w.optimizer_updates,'scheduled_windows':w.scheduled_windows}
                check(not w.optimizer.state and w.optimizer_updates==w.scheduled_windows==0,'no optimizer activity')
                if capture and label=='fused' and not micro_results:
                    micro_results['combined_gradients']=micro([p.grad for p in w.parameters if p.grad is not None])
                    # Actual Session family shapes, recorded from the captured
                    # terminal request; use detached snapshots, not a new run.
                    shapes=[episodes[-1]['hidden']]+[t for pair in episodes[-1]['cache'] for t in pair]
                    values=[torch.ones(t['shape'],device='cuda',dtype=getattr(torch,t['dtype'].split('.')[-1])) for t in shapes]
                    micro_results['causal_state_shapes']=micro(values)
            w.abort();del w
            gc.collect()
            return stats,buffers,grads
        for batch,ids in batches.items():
            baseline,buf0,grad0=execute('legacy',ids,True)
            candidate,buf1,grad1=execute('fused',ids,True)
            check(baseline['episodes']==candidate['episodes'],'output/cache/loss/accounting byte parity')
            check(baseline['eligible']==candidate['eligible'],'eligibility parity')
            check(buf0.keys()==buf1.keys(),'component parity')
            comparisons=0
            for comp in buf0:
                check(buf0[comp].keys()==buf1[comp].keys(),'component None parity')
                for i in buf0[comp]:
                    check(torch.equal(buf0[comp][i],buf1[comp][i]),'component gradient byte parity');comparisons+=1
            for x,y in zip(grad0,grad1):
                check((x is None)==(y is None),'combined None parity')
                if x is not None:check(torch.equal(x,y),'combined gradient byte parity')
            parity={'batch':batch,'passed':True,'component_tensor_comparisons':comparisons,
                'combined_tensor_comparisons':sum(x is not None for x in grad0),
                'baseline':baseline,'fused':candidate}
            parities.append(parity);(out/f'parity-{batch}.json').write_bytes(canonical(parity))
            del baseline,candidate,buf0,buf1,grad0,grad1;gc.collect()
            emit(stage='parity',batch=batch,passed=True,component_tensor_comparisons=comparisons)
            for repeat in range(3):
                for label in (('legacy','fused') if repeat%2==0 else ('fused','legacy')):
                    stats,_,_=execute(label,ids);stats.update(batch=batch,repeat=repeat)
                    measured.append(stats);emit(stage='timed',**stats)
        after=receiver_hash(b,c);base_after=frozen_check(b,a.snapshot)
        check(after==before and ref.fingerprint()==reference,'immutable receiver/reference')
        check(base_after['pass'] and base_after['base_tied_storage'],'immutable tied W')
        check(all(p.grad is None for m in (b,c,ref) for p in m.parameters()),'all gradients cleared')
        summary={}
        for batch in batches:
            med={label:statistics.median(v['seconds'] for v in measured if v['batch']==batch and v['path']==label) for label in paths}
            summary[batch]={**med,'speedup':med['legacy']/med['fused'],
                'reduction_percent':100*(1-med['fused']/med['legacy'])}
        result={'protocol':'fused-finite-v1','torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
            'gpu_uuid':'GPU-225509c3-d702-6c27-de03-76b5e99b33b1',
            'dataset':release.identity,'checkpoint_manifest':file_hash(checkpoint/'manifest.json'),
            'receiver_before':before,'receiver_after':after,'reference':reference,'base_before':base_before,'base_after':base_after,
            'sources':sources,'baseline_sources':LEGACY,'batches':batches,'parities':parities,
            'measurements':measured,'microbenchmarks':micro_results,'summary':summary,'optimizer_updates':0,
            'scope':'Readonly predicates only; no semantic/full-flow qualification or trained update benchmark.'}
        (out/'report.json').write_bytes(canonical(result));emit(stage='complete',summary=summary)


if __name__=='__main__':main()
