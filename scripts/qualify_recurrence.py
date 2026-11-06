"""Focused graph-v1 qualification and reference throughput; never optimizes."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path

from qwen_one_two.training.inputs import canonical,digest,check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--lengths',type=int,nargs='+',default=[128,512,1024])
    parser.add_argument('--ports',type=int,nargs='+',default=[1,2,3])
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from safetensors.torch import load_file
    from qwen_one_two.g1_proof import configure
    from qwen_one_two.codec.qualify import load,trace,compare
    from qwen_one_two.codec.schema import Port
    from qwen_one_two.g1_checks import tensor_error,grad_error
    from qwen_one_two.g1_evidence import frozen_check
    from qwen_one_two.training.reference import Reference,state_fingerprint
    from qwen_one_two.training.binding import wire_payload
    from qwen_one_two.training.inputs import SLOTS
    from qwen_one_two.training.release import load_release
    from qwen_one_two.training.component_producer import reference_record
    from scripts.fit_reference import TRAIN_SHA,source_snapshot
    configure();started=time.monotonic()
    snapshot=source_snapshot(output)
    report=dict(protocol='graph-recurrence-v1',profile='v3-precision/recurrence-graph-v1',
                status='RUNNING',optimizer_updates=0,training_performed=False,cases=[],
                full_432_case_matrix_repeated=False,source_hashes=snapshot)
    def persist():
        (output/'report.json').write_bytes(canonical(report))
    with sdpa_kernel(SDPBackend.MATH):
        b,c,_=load(args.snapshot,'data/g1-proof/gate-repair-v2-20261105/diagnostic.safetensors')
        before=state_fingerprint({**dict(b.named_parameters()),**{'codec.'+n:p for n,p in c.state_dict().items()}})
        report['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,
            triton=__import__('triton').__version__,gpu=torch.cuda.get_device_name(0),
            gpu_uuid='GPU-225509c3-d702-6c27-de03-76b5e99b33b1',
            driver=__import__('subprocess').check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader',
                '--id=GPU-225509c3-d702-6c27-de03-76b5e99b33b1'],text=True).strip())
        try:
            for length in args.lengths:
                for port in args.ports:
                    baseline=None
                    for mode in ('eager','functional','checkpoint','serving','fp32'):
                        torch.manual_seed(17)
                        b.checkpointing=mode=='checkpoint'
                        b.set_precision(torch.float32 if mode=='fp32' else torch.bfloat16)
                        b.recurrence_backend='eager' if mode in {'eager','fp32'} else 'graph-v1'
                        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic()
                        with torch.set_grad_enabled(mode!='serving'):
                            current=trace(b,c,Port(port),length,mode!='serving')
                        torch.cuda.synchronize()
                        checks={'pass':True}
                        if baseline is not None:
                            checks=compare(current,baseline,mode=='fp32')
                            # Same FP32 transformer equations: enforce the stricter
                            # FP32 pointwise/RMS limits too, not only BF16 limits.
                            strict=[]
                            for (_,h,states),(_,rh,reference) in zip(current['snapshots'],baseline['snapshots']):
                                strict.extend(tensor_error(x,y,1e-6,1e-5,1e-5) for x,y in [(h,rh),*zip(states,reference)])
                            checks['strict_FP32_forward_pass']=all(x['pass'] for x in strict)
                            checks['strict_FP32_forward_worst_rms']=max(x['normalized_rms'] for x in strict)
                            checks['strict_FP32_forward_maximum_absolute']=max(x['maximum_absolute'] for x in strict)
                            checks['pass'] &= checks['strict_FP32_forward_pass']
                        row=dict(length=length,port=port,mode=mode,seed=17,
                            recurrence_backend=b.recurrence_backend,comparison=checks,
                            seconds=time.monotonic()-t,peak_allocated=torch.cuda.max_memory_allocated(),
                            summary=current['summary'],calls=current['calls'],connectivity=current['probe'])
                        report['cases'].append(row);persist()
                        print(json.dumps({k:row[k] for k in ('length','port','mode','seconds','comparison')}),flush=True)
                        check(checks['pass'],'whole-model numeric/discrete qualification failed')
                        if baseline is None:baseline=current
                        else:del current
                        for layer in b.duals:layer.evaluation_outputs.clear()
                        gc.collect();torch.cuda.empty_cache()
                    del baseline;gc.collect();torch.cuda.empty_cache()
            b.set_precision(torch.bfloat16);b.checkpointing=False
            if args.benchmark:
                ref=Reference(b,c)
                bundle=Path('data/reference-fit-20261106/run/reference-1000')
                manifest=json.loads((bundle/'manifest.json').read_bytes())
                check(digest((bundle/'weights.safetensors').read_bytes())==manifest['files']['weights.safetensors'],'benchmark reference file hash')
                ref.load_state_dict({k.removeprefix('reference.'):v for k,v in load_file(str(bundle/'weights.safetensors'),device='cuda').items()})
                ref_before=ref.fingerprint();check(ref_before==manifest['reference_id'],'benchmark reference identity')
                release=load_release('data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
                annotation,_=reference_record(release,release.record('train-reference-0000'))
                text=annotation.content.decode()
                rendered=b.tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
                b.recurrence_backend='eager'
                states=ref.annotation_states(b.tokenizer.encode(rendered,add_special_tokens=False))
                targets=b.tokenizer.encode(text,add_special_tokens=False)+[248046]
                def episode():
                    slots=ref.pooler(states,*SLOTS[annotation.role])
                    loss=ref.reconstruct_teacher_forced(wire_payload(ref,slots,annotation.role),targets)
                    return loss.detach(),torch.autograd.grad(loss,tuple(ref.parameters()),allow_unused=True)
                timings={};oracle=None;parity=[];kernels={}
                for backend in ('eager','graph-v1'):
                    b.recurrence_backend=backend
                    loss,grads=episode();del grads # warm compilation outside timing
                    times=[]
                    for repeat in range(3):
                        torch.cuda.synchronize();t=time.monotonic()
                        loss,grads=episode();torch.cuda.synchronize();times.append(time.monotonic()-t)
                        if backend=='eager' and repeat==2:oracle=(loss,grads)
                        elif backend=='graph-v1':
                            comparison=grad_error(grads,oracle[1],1e-4,.99999)
                            check(comparison['pass'] and abs(float(loss-oracle[0]))<=1e-4,'trained reference VJP parity')
                            parity.append(dict(repeat=repeat,loss_absolute=float((loss-oracle[0]).abs()),gradient=comparison))
                        del grads
                    timings[backend]=dict(samples_seconds=times,median_seconds=statistics.median(times))
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=False,with_stack=False) as prof:
                        loss,grads=episode();torch.cuda.synchronize()
                    del grads
                    trace_path=output/f'{backend}-trace.json';prof.export_chrome_trace(str(trace_path))
                    events=json.loads(trace_path.read_bytes())['traceEvents']
                    kernels[backend]=dict(cuda_kernels=sum(e.get('cat')=='kernel' and e.get('ph')=='X' for e in events),
                        launch_calls=sum(e.get('name')=='cudaLaunchKernel' for e in events),trace_sha256=digest(trace_path.read_bytes()))
                check(ref.fingerprint()==ref_before,'benchmark changed reference')
                report['benchmark']=dict(sample='train-reference-0000',reference_sha256=ref_before,reference_unchanged=True,
                    timings=timings,kernels=kernels,gradient_parity=parity,
                    speedup=timings['eager']['median_seconds']/timings['graph-v1']['median_seconds'],
                    scope='warm forward/VJP only; excludes accumulation/optimizer; no optimizer object or step')
                print(json.dumps(dict(benchmark=report['benchmark'])),flush=True)
            report['live_unchanged']=before==state_fingerprint({**dict(b.named_parameters()),**{'codec.'+n:p for n,p in c.state_dict().items()}})
            report['frozen_base']=frozen_check(b,args.snapshot)
            check(report['live_unchanged'] and report['frozen_base']['pass'],'weights changed')
            check(all(p.grad is None for p in list(b.parameters())+list(c.parameters())),'parameter gradient accumulation')
            report['status']='PASSED'
        except BaseException as exc:
            report['status']='FAILED';report['error']=repr(exc);raise
        finally:
            report['elapsed_seconds']=time.monotonic()-started;persist()


if __name__=='__main__':main()
