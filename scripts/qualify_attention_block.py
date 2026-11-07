"""Focused numeric qualification of larger pure CUDA replay; zero updates."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
from qwen_one_two.training.inputs import canonical,check
from qwen_one_two.training.manifest import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True)
    p.add_argument('--output',required=True);p.add_argument('--execute-gpu',action='store_true')
    a=p.parse_args();check(a.execute_gpu,'explicit --execute-gpu required')
    out=Path(a.output).resolve()
    from qwen_one_two.training.two_gpu import ROOT,source_hashes
    check(out.is_relative_to(ROOT/'data'),'qualification outputs belong in data/')
    out.mkdir(parents=True,exist_ok=False)
    sources=source_hashes()
    sources['scripts/qualify_attention_block.py']=file_hash(Path(__file__))
    for name in sources:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.g1_checks import tensor_error
    from qwen_one_two.codec.qualify import load,trace,compare
    from qwen_one_two.codec.schema import Port
    from qwen_one_two.codec import attention_block,recurrence
    from qwen_one_two.training.reference import state_fingerprint
    configure();started=time.perf_counter()
    report={'protocol':'linear-attention-block-v1','status':'RUNNING','sources':sources,'cases':[],
        'optimizer_updates':0,'training_performed':False,'full_432_matrix_repeated':False,
        'scope':'existing prescribed numeric fixtures; not learned/semantic or full training qualification'}
    def persist():(out/'report.json').write_bytes(canonical(report))
    def clear():
        for d in b.duals:d.evaluation_outputs.clear()
        attention_block.clear_graph_cache();recurrence.clear_graph_cache();gc.collect();torch.cuda.empty_cache()
    with sdpa_kernel(SDPBackend.MATH):
        b,c,_=load(a.snapshot,ROOT/'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        def fingerprint():return state_fingerprint({**dict(b.named_parameters()),**{'codec.'+n:p for n,p in c.named_parameters()}})
        before=fingerprint();report['initial_base']=frozen_check(b,a.snapshot)
        report['environment']={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda}
        try:
            for length in (128,512,1024):
                for port in (1,2,3):
                    baseline=None
                    for mode in ('oracle','functional','checkpoint','serving'):
                        clear();torch.manual_seed(17)
                        b.checkpointing=mode=='checkpoint';b.attention_block_backend='eager' if mode=='oracle' else 'graph-v1'
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                        with torch.set_grad_enabled(mode!='serving'):current=trace(b,c,Port(port),length,mode!='serving')
                        torch.cuda.synchronize()
                        comparison={'pass':True}
                        if baseline is not None:
                            comparison=compare(current,baseline,False)
                            strict=[tensor_error(x,y,1e-6,1e-5,1e-5)
                                for (_,h,states),(_,rh,refs) in zip(current['snapshots'],baseline['snapshots'])
                                for x,y in [(h,rh),*zip(states,refs)]]
                            comparison.update(strict_fp32_forward_pass=all(e['pass'] for e in strict),
                                strict_fp32_worst_rms=max(e['normalized_rms'] for e in strict),
                                strict_fp32_maximum_absolute=max(e['maximum_absolute'] for e in strict),
                                exact_forward_states=all(torch.equal(x,y)
                                    for (_,h,states),(_,rh,refs) in zip(current['snapshots'],baseline['snapshots'])
                                    for x,y in [(h,rh),*zip(states,refs)]))
                            comparison['pass'] &= comparison['strict_fp32_forward_pass']
                        row={'positions':length,'port':port,'mode':mode,'comparison':comparison,
                            'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                            'summary':current['summary'],'calls':current['calls'],'connectivity':current['probe'],
                            'parameter_gradients_compared':mode in ('functional','checkpoint')}
                        report['cases'].append(row);persist()
                        print(json.dumps({k:row[k] for k in ('positions','port','mode','seconds','comparison')}),flush=True)
                        check(comparison['pass'],'full-state/gradient/control numeric check failed')
                        if baseline is None:baseline=current
                        else:del current
                    del baseline;clear()
            report['final_base']=frozen_check(b,a.snapshot);report['weights_unchanged']=before==fingerprint()
            report['gradients_none']=all(p.grad is None for m in (b,c) for p in m.parameters())
            check(report['final_base']['pass'] and report['weights_unchanged'] and report['gradients_none'],'frozen integrity')
            report['status']='PASSED_SCOPED_NUMERIC'
        except BaseException as error:report['status']='FAILED_NUMERIC';report['error']=repr(error);raise
        finally:report['seconds']=time.perf_counter()-started;persist()


if __name__=='__main__':main()
